#!/usr/bin/env python3
"""Validate t=0 non-bonded energies for the GMX_vs_DIFFaMD example.

This script compares:
1) Analytical pairwise LJ/Coulomb energies on frame 0.
2) Diff-MD internal energies computed through the same routines used by mdrun.
3) Energies stored in an output H5MD file.

Run from this directory, for example:
    python validate_t0_nonbonded.py \
        --coord output.h5 \
        --config options.one.toml \
        --topol topol.toml \
        --sim-h5 try_one.h5
"""

from __future__ import annotations

import argparse
import math
import tomllib
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np

from diff_md.input_parser import System
from diff_md.neighbor_list import (
    apply_nlist,
    apply_nlist_elec,
    build_neighbor_list_cell,
    exclude_bonded_neighbors,
)
from diff_md.nonbonded import (
    cic_paint,
    ewald_real_space,
    filter_density,
    get_LJ_energy_and_forces,
    get_elec_potential_and_energy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate t=0 non-bonded energies")
    parser.add_argument("--coord", default="output.h5", help="Input coordinate H5")
    parser.add_argument("--config", default="options.one.toml", help="Simulation TOML")
    parser.add_argument("--topol", default="topol.toml", help="Topology TOML")
    parser.add_argument(
        "--sim-h5",
        default="try_one.h5",
        help="Simulation output H5MD used for stored-energy comparison",
    )
    parser.add_argument(
        "--elec-conversion",
        type=float,
        default=138.935458,
        help="Electrostatic conversion factor in kJ mol-1 nm e-2",
    )
    return parser.parse_args()


def pbc_displacement(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    return dr - box * np.round(dr / box)


def lj_energy(r: float, sigma: float, epsilon: float) -> float:
    x = sigma / r
    x6 = x**6
    return 4.0 * epsilon * x6 * (x6 - 1.0)


def build_excluded_pairs_from_topology(topol_toml: Path, nrexcl: int) -> set[tuple[int, int]]:
    with topol_toml.open("rb") as f:
        topol = tomllib.load(f)

    include_files = [Path(p) for p in topol["system"]["include"]]
    molecules = topol["system"]["molecules"]

    mol_defs: dict[str, dict] = {}
    for rel in include_files:
        with rel.open("rb") as f:
            data = tomllib.load(f)
        mol_name = next(iter(data.keys()))
        mol_defs[mol_name] = data[mol_name]

    excluded: set[tuple[int, int]] = set()
    offset = 0

    for mol_name, count in molecules:
        mol = mol_defs[mol_name]
        nat = int(mol["atomnum"])
        adjacency: list[list[int]] = [[] for _ in range(nat)]

        for bond in mol.get("bonds", []):
            i = int(bond[0]) - 1
            j = int(bond[1]) - 1
            adjacency[i].append(j)
            adjacency[j].append(i)

        local_pairs: set[tuple[int, int]] = set()
        for src in range(nat):
            dist = [-1] * nat
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)

            for dst in range(src + 1, nat):
                if dist[dst] != -1 and dist[dst] <= nrexcl:
                    local_pairs.add((src, dst))

        for _ in range(int(count)):
            for i, j in local_pairs:
                excluded.add((offset + i, offset + j))
            offset += nat

    return excluded


def analytical_pairwise_energies(
    coord_h5: Path,
    config_toml: Path,
    topol_toml: Path,
    elec_conversion: float,
) -> tuple[float, float, int, int]:
    with config_toml.open("rb") as f:
        options = tomllib.load(f)

    rc = float(options["simulation"]["rc"])
    rlj = float(options["simulation"]["rlj"])
    nrexcl = int(options["simulation"]["nrexcl"])

    lj_map = {row[0]: (float(row[1]), float(row[2])) for row in options["field"]["LJ_type_param"]}

    with h5py.File(coord_h5, "r") as f:
        positions = np.array(f["coordinates"][0], dtype=float)
        charges = np.array(f["charge"][:], dtype=float)
        names = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in f["names"][:]]
        box = np.array(f.attrs["box"], dtype=float)

    excluded = build_excluded_pairs_from_topology(topol_toml, nrexcl=nrexcl)

    e_lj = 0.0
    e_coul = 0.0
    pair_count = 0

    n_atoms = len(positions)
    for i in range(n_atoms - 1):
        sigma_i, eps_i = lj_map[names[i]]
        qi = charges[i]
        for j in range(i + 1, n_atoms):
            if (i, j) in excluded:
                continue

            sigma_j, eps_j = lj_map[names[j]]
            qj = charges[j]

            rij = pbc_displacement(positions[j] - positions[i], box)
            r = np.linalg.norm(rij)
            if r == 0.0:
                continue

            sigma_ij = 0.5 * (sigma_i + sigma_j)
            eps_ij = math.sqrt(eps_i * eps_j)

            pair_count += 1
            if r < rlj:
                e_lj += lj_energy(r, sigma_ij, eps_ij) - lj_energy(rlj, sigma_ij, eps_ij)
            if r < rc:
                e_coul += elec_conversion * qi * qj / r

    return e_lj, e_coul, pair_count, len(excluded)


def diffmd_internal_energies(coord: Path, config: Path, topol: Path) -> tuple[float, float, float, float]:
    args = SimpleNamespace(
        coord=str(coord),
        config=str(config),
        topol=str(topol),
        no_charges=False,
        database=None,
    )

    system = System.constructor(args, dir=".")
    cfg = system.config
    topo = system.topol
    positions = jnp.mod(system.positions, cfg.box_size)

    use_14_scaling = (
        cfg.ff_family == "amber_like"
        and topo.one_four_pairs is not None
        and (cfg.lj14_scale != 1.0 or cfg.coulomb14_scale != 1.0)
    )
    need_concat_14 = use_14_scaling and cfg.nrexcl < 3

    if topo.excluded_pairs is not None and need_concat_14 and cfg.coulombtype == 1:
        excluded_for_elec = (
            jnp.concatenate((topo.excluded_pairs[0], topo.one_four_pairs[0])),
            jnp.concatenate((topo.excluded_pairs[1], topo.one_four_pairs[1])),
        )
    elif topo.excluded_pairs is not None:
        excluded_for_elec = topo.excluded_pairs
    elif need_concat_14 and cfg.coulombtype == 1:
        excluded_for_elec = topo.one_four_pairs
    else:
        excluded_for_elec = None

    if topo.excluded_pairs is not None and need_concat_14:
        excluded_for_main = (
            jnp.concatenate((topo.excluded_pairs[0], topo.one_four_pairs[0])),
            jnp.concatenate((topo.excluded_pairs[1], topo.one_four_pairs[1])),
        )
    elif topo.excluded_pairs is not None:
        excluded_for_main = topo.excluded_pairs
    elif need_concat_14:
        excluded_for_main = topo.one_four_pairs
    else:
        excluded_for_main = None

    rv = float(cfg.rv)
    skin = float(cfg.skin)
    if skin > 0:
        rv = max(rv, max(float(cfg.rc), float(cfg.rlj)) + skin)

    density = cfg.n_particles / cfg.box_size.prod()
    max_neighbors = int(0.5 * cfg.n_particles * (4.0 * jnp.pi * rv**3 / 3.0) * density)
    max_neighbors += 5000

    neigh_i, neigh_j, _ = build_neighbor_list_cell(positions, cfg.box_size, rv, max_neighbors)
    if excluded_for_main is not None:
        neigh_i, neigh_j = exclude_bonded_neighbors(
            neigh_i,
            neigh_j,
            excluded_for_main[0],
            excluded_for_main[1],
        )

    lj_forces = jnp.zeros_like(positions)
    pair_lj = apply_nlist(
        neigh_i,
        neigh_j,
        positions,
        cfg.box_size,
        cfg.sgm_table,
        cfg.epsl_table,
        system.types,
    )
    e_lj, _ = get_LJ_energy_and_forces(lj_forces, pair_lj, cfg)

    pair_elec = apply_nlist_elec(
        neigh_i,
        neigh_j,
        positions,
        system.charges,
        cfg.box_size,
        cfg.sgm_table,
        cfg.epsl_table,
        system.types,
    )
    excl_elec = (
        apply_nlist_elec(
            excluded_for_elec[0],
            excluded_for_elec[1],
            positions,
            system.charges,
            cfg.box_size,
            cfg.sgm_table,
            cfg.epsl_table,
            system.types,
        )
        if excluded_for_elec is not None
        else None
    )

    e_real, _, _ = ewald_real_space(pair_elec, cfg, excl_elec)
    phi_q = cic_paint(positions, cfg, mass=system.charges)
    phi_q_fourier = filter_density(phi_q, cfg)
    _, e_kspace_minus_self, _ = get_elec_potential_and_energy(phi_q, phi_q_fourier, cfg)
    e_pme = e_real + e_kspace_minus_self

    return float(e_lj), float(e_pme), float(e_real), float(e_kspace_minus_self)


def read_stored_energies(sim_h5: Path) -> tuple[float, float]:
    with h5py.File(sim_h5, "r") as f:
        obs = f["observables"]
        e_lj = float(np.array(obs["LJ_energy"]["value"])[0, 0])
        if "field_q_energy" not in obs:
            raise KeyError("'field_q_energy' not found in observables")
        e_elec = float(np.array(obs["field_q_energy"]["value"])[0, 0])
    return e_lj, e_elec


def main() -> None:
    args = parse_args()

    coord = Path(args.coord)
    config = Path(args.config)
    topol = Path(args.topol)
    sim_h5 = Path(args.sim_h5)

    e_lj_pair, e_coul_pair, n_main_pairs, n_excluded = analytical_pairwise_energies(
        coord_h5=coord,
        config_toml=config,
        topol_toml=topol,
        elec_conversion=args.elec_conversion,
    )

    e_lj_internal, e_elec_internal, e_real, e_kspace_minus_self = diffmd_internal_energies(
        coord=coord,
        config=config,
        topol=topol,
    )

    e_lj_stored, e_elec_stored = read_stored_energies(sim_h5)

    print("=== t=0 Non-bonded Validation ===")
    print(f"coord={coord}, config={config}, topol={topol}, sim_h5={sim_h5}")
    print(f"main_pairs={n_main_pairs}, excluded_pairs={n_excluded}")

    print("\n[LJ]")
    print(f"pairwise_analytical      = {e_lj_pair: .10f}")
    print(f"diffmd_internal          = {e_lj_internal: .10f}")
    print(f"stored_h5                = {e_lj_stored: .10f}")
    print(f"delta(pair - stored)     = {e_lj_pair - e_lj_stored: .10e}")
    print(f"delta(internal - stored) = {e_lj_internal - e_lj_stored: .10e}")

    print("\n[Electrostatics]")
    print(f"pairwise_coulomb_only    = {e_coul_pair: .10f}")
    print(f"pme_internal_total       = {e_elec_internal: .10f}")
    print(f"stored_h5                = {e_elec_stored: .10f}")
    print(f"delta(pair - stored)     = {e_coul_pair - e_elec_stored: .10f}")
    print(f"delta(pme - stored)      = {e_elec_internal - e_elec_stored: .10e}")

    print("\n[PME breakdown]")
    print(f"real_space               = {e_real: .10f}")
    print(f"kspace_minus_self        = {e_kspace_minus_self: .10f}")


if __name__ == "__main__":
    main()
