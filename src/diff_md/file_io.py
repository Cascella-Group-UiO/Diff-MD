import collections
import getpass
import logging
import os
from typing import Any

import h5py
import numpy as np
import tomlkit

from .config import read_toml
from .logger import Logger
from .models import GeneralModel


def _build_unwrap_data(molecules, bonds_atom1, bonds_atom2):
    """Pre-compute data structures for whole-molecule unwrapping.

    Parameters
    ----------
    molecules : array-like, shape (N,)
        Per-atom molecule ID (0-based).
    bonds_atom1, bonds_atom2 : array-like, shape (M,)
        Global bond pair arrays.

    Returns
    -------
    small_anchors : ndarray, shape (S,)   or None
        Anchor atom indices for small molecules (2-3 atoms).
    small_others  : ndarray, shape (S, 2) or None
        Other-atom indices for small molecules (padded with -1).
    large_mol_atoms : list[list[int]]     or None
        Atom lists for molecules with >3 atoms.
    adjacency : list[list[int]]           or None
        Per-atom bond adjacency list.
    """
    if molecules is None:
        return None, None, None, None

    n_atoms = len(molecules)

    mol_to_atoms = {}
    for i, mol in enumerate(np.asarray(molecules).ravel()):
        mol_to_atoms.setdefault(int(mol), []).append(i)

    adjacency = [[] for _ in range(n_atoms)]
    for a, b in zip(np.asarray(bonds_atom1).ravel(),
                     np.asarray(bonds_atom2).ravel()):
        a, b = int(a), int(b)
        adjacency[a].append(b)
        adjacency[b].append(a)

    # Partition molecules into "small" (1-3 atoms, vectorisable) and "large"
    small_anchor_list = []
    small_other_list = []
    large_mol_atoms = []

    for atom_ids in mol_to_atoms.values():
        n = len(atom_ids)
        if n <= 1:
            continue
        if n <= 3:
            anchor = atom_ids[0]
            others = atom_ids[1:] + [-1] * (2 - (n - 1))  # pad to length 2
            small_anchor_list.append(anchor)
            small_other_list.append(others)
        else:
            large_mol_atoms.append(atom_ids)

    small_anchors = np.array(small_anchor_list, dtype=np.intp) if small_anchor_list else None
    small_others = np.array(small_other_list, dtype=np.intp) if small_other_list else None

    return small_anchors, small_others, large_mol_atoms, adjacency


def unwrap_molecules(positions, box_size, small_anchors, small_others,
                     large_mol_atoms, adjacency):
    """Return positions with whole-molecule imaging applied.

    For each molecule a BFS traversal through the bond graph anchors the
    first atom and places every bonded neighbour at the minimum-image
    position relative to its parent.  This guarantees that no molecule is
    split across a periodic boundary in the output.

    Small molecules (2-3 atoms, e.g. water) are handled with a fast
    vectorised path; only larger molecules fall back to the Python BFS.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Possibly-wrapped atomic positions.
    box_size : array-like, shape (3,)
        Periodic box lengths.
    small_anchors : ndarray or None
        Anchor indices for small (2-3 atom) molecules.
    small_others : ndarray or None
        Other-atom indices for small molecules (padded with -1).
    large_mol_atoms : list[list[int]] or None
        Atom lists for molecules with >3 atoms.
    adjacency : list[list[int]] or None
        Per-atom bond adjacency list.

    Returns
    -------
    ndarray, shape (N, 3)
        Unwrapped positions (the input array is **not** modified).
    """
    pos = np.array(positions, dtype=np.float64)
    box = np.asarray(box_size, dtype=np.float64)
    inv_box = 1.0 / box

    # --- fast vectorised path for small molecules (water etc.) -----------
    if small_anchors is not None and small_others is not None:
        anchor_pos = pos[small_anchors]                   # (S, 3)
        for col in range(small_others.shape[1]):           # at most 2 iterations
            idx = small_others[:, col]                     # (S,)
            mask = idx >= 0                                # valid entries
            if not np.any(mask):
                continue
            valid_idx = idx[mask]
            dr = pos[valid_idx] - anchor_pos[mask]
            dr -= box * np.round(dr * inv_box)
            pos[valid_idx] = anchor_pos[mask] + dr

    # --- BFS path for large molecules (protein, lipids, …) ----------------
    if large_mol_atoms:
        for atom_ids in large_mol_atoms:
            atom_set = set(atom_ids)
            visited = set()
            root = atom_ids[0]
            visited.add(root)
            queue = collections.deque([root])

            while queue:
                parent = queue.popleft()
                for child in adjacency[parent]:
                    if child in visited or child not in atom_set:
                        continue
                    dr = pos[child] - pos[parent]
                    dr -= box * np.round(dr * inv_box)
                    pos[child] = pos[parent] + dr
                    visited.add(child)
                    queue.append(child)

    return pos


class OutDataset:
    def __init__(
        self,
        destdir,
        filename,
        double_out=False,
        append=False,
    ):
        if double_out:
            self.float_dtype = "float64"
        else:
            self.float_dtype = "float32"

        os.makedirs(destdir, exist_ok=True)
        path = os.path.join(destdir, f"{filename}.h5")
        if append and os.path.exists(path):
            self.file = h5py.File(path, "a")
            self._is_append = True
        else:
            self.file = h5py.File(path, "w")
            self._is_append = False

        # Populated by store_static when molecule info is available
        self._unwrap_small_anchors = None
        self._unwrap_small_others = None
        self._unwrap_large_mol_atoms = None
        self._unwrap_adjacency = None

    def close_file(self):
        self.file.close()

    def flush(self):
        self.file.flush()


def _to_multiline_lj_array(rows):
    """Re-pack a list-of-lists as a tomlkit multiline array.

    Each inner row is printed on its own line, matching the
    human-readable column layout used in the input TOML files.
    """
    outer = tomlkit.array()
    outer.multiline(True)
    for row in rows:
        inner = tomlkit.array()
        for val in row:
            inner.append(val)
        outer.append(inner)
    return outer


# ── Energy-log helpers (shared between mdrun.py and optimize.py) ──────────────

_ENERGY_LOG_HEADER = (
    "# Diff-MD energy log\n"
    "# step | time_fs | temp_K | E_total_kJmol | E_potential_kJmol | E_kin_kJmol | "
    "E_LJ_kJmol | E_elec_kJmol | E_bond_kJmol | E_angle_kJmol | "
    "E_torsional_kJmol | E_improper_torsional_kJmol | E_cmap_kJmol | P_bar\n"
)


def _safe_float(value) -> float:
    """Convert any scalar (JAX array, numpy scalar, Python float) to float."""
    return float(np.asarray(value))


def format_energy_log_line(
    step,
    time_fs,
    temperature,
    total_energy,
    potential_energy,
    kinetic_energy,
    lj_energy,
    elec_energy,
    bond_energy,
    angle_energy,
    torsional_energy,
    improper_energy,
    cmap_energy=0.0,
    pressure=0.0,
) -> str:
    return (
        f"{step:10d} "
        f"{_safe_float(time_fs):14.3f} "
        f"{_safe_float(temperature):11.4f} "
        f"{_safe_float(total_energy):15.6f} "
        f"{_safe_float(potential_energy):15.6f} "
        f"{_safe_float(kinetic_energy):15.6f} "
        f"{_safe_float(lj_energy):15.6f} "
        f"{_safe_float(elec_energy):15.6f} "
        f"{_safe_float(bond_energy):15.6f} "
        f"{_safe_float(angle_energy):15.6f} "
        f"{_safe_float(torsional_energy):15.6f} "
        f"{_safe_float(improper_energy):15.6f} "
        f"{_safe_float(cmap_energy):15.6f} "
        f"{_safe_float(pressure):15.6f}"
    )


def write_energy_log_from_trj(
    path: str,
    trj: dict,
    config,
    n_print: int,
    append: bool = False,
) -> None:
    """Write an energy.log file from a simulate.py trajectory dict.

    One line per trajectory chunk (every n_print steps).  The step index
    reported is the end-of-chunk step (1-based), so the first line is
    step=n_print, the second is step=2*n_print, etc.

    Note: simulate.py stores torsional + improper + CMAP energies
    combined in ``trj["dihedral energy"]``; the improper and CMAP
    columns are therefore always 0.0 in this code path.  mdrun.py logs
    the per-term split correctly from the live carry.
    Pressure is non-zero only for NPT runs that store ``trj["pressure"]``.
    """
    required = ["temperature", "kinetic energy", "LJ energy"]
    if not all(k in trj for k in required):
        return  # no energy data (n_print==0 or equilibration-only run)

    n_frames = len(trj.get("temperature", []))
    if n_frames == 0:
        return

    outer_ts = _safe_float(config.outer_ts)
    has_pressure = "pressure" in trj and len(trj["pressure"]) == n_frames

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        if not append:
            f.write(_ENERGY_LOG_HEADER)
        for i in range(n_frames):
            step     = (i + 1) * n_print
            time_fs  = step * outer_ts * 1000.0
            T        = _safe_float(trj["temperature"][i])
            kin      = _safe_float(trj["kinetic energy"][i])
            lj       = _safe_float(trj["LJ energy"][i])
            elec     = _safe_float(trj.get("elec energy",     [0.0] * n_frames)[i])
            bond     = _safe_float(trj.get("bond energy",     [0.0] * n_frames)[i])
            ang      = _safe_float(trj.get("angle energy",    [0.0] * n_frames)[i])
            dih      = _safe_float(trj.get("dihedral energy", [0.0] * n_frames)[i])
            pot      = lj + elec + bond + ang + dih
            tot      = pot + kin
            pres     = _safe_float(trj["pressure"][i]) if has_pressure else 0.0
            f.write(
                format_energy_log_line(
                    step, time_fs, T, tot, pot, kin,
                    lj, elec, bond, ang,
                    torsional_energy=dih,  # torsional + improper combined
                    improper_energy=0.0,   # not tracked separately in simulate.py
                    pressure=pres,
                ) + "\n"
            )


def save_params(filename: str, toml: dict[str, Any], params: GeneralModel) -> None:
    assert params.LJ_param is not None

    with open(filename, "w") as outfile:
        model_toml = toml["nn"]["model"]

        if "LJ_type_param" in model_toml and params.lj_mode == "type":
            sigma = np.array(params.lj_sigma_ref)
            epsl = np.array(params.lj_epsilon_ref)

            n_sigma = int(params.n_sigma_train)
            n_epsilon = int(params.n_epsilon_train)

            if n_sigma > 0:
                sigma[np.array(params.lj_sigma_idx)] = np.array(params.LJ_param[:n_sigma])
            if n_epsilon > 0:
                epsl_vals = np.array(params.LJ_param[n_sigma : n_sigma + n_epsilon])
                epsl[np.array(params.lj_epsilon_idx)] = epsl_vals

            for i, row in enumerate(model_toml["LJ_type_param"]):
                t = params.lj_name_to_type[row[0]]
                model_toml["LJ_type_param"][i][1] = float(sigma[t])
                model_toml["LJ_type_param"][i][2] = float(epsl[t])
            model_toml["LJ_type_param"] = _to_multiline_lj_array(
                model_toml["LJ_type_param"]
            )
        elif "LJ_param" in model_toml:
            n_sigma = int(params.n_sigma_train)
            n_epsilon = int(params.n_epsilon_train)

            sigma_rows = []
            epsilon_rows = []
            has_flags = False
            for i, pair in enumerate(model_toml["LJ_param"]):
                if len(pair) > 4:
                    flags = {str(f).lower() for f in pair[4:] if isinstance(f, str)}
                    if flags:
                        has_flags = True
                    if "on_sigma" in flags or "on_sgm" in flags:
                        sigma_rows.append(i)
                    if "on_eps" in flags or "on_epsilon" in flags:
                        epsilon_rows.append(i)

            if not has_flags:
                # No explicit flags: all epsilons trainable
                epsilon_rows = list(range(len(model_toml["LJ_param"])))

            for k, row_idx in enumerate(sigma_rows):
                model_toml["LJ_param"][row_idx][2] = float(params.LJ_param[k])
            for k, row_idx in enumerate(epsilon_rows):
                model_toml["LJ_param"][row_idx][3] = float(params.LJ_param[n_sigma + k])
            model_toml["LJ_param"] = _to_multiline_lj_array(
                model_toml["LJ_param"]
            )
        tomlkit.dump(toml, outfile)


def setup_time_dependent_element(
    name, parent_group, n_frames, shape, dtype, units=None
):
    group = parent_group.create_group(name)
    step = group.create_dataset("step", (n_frames,), "int32", maxshape=(None,), chunks=True)
    time = group.create_dataset("time", (n_frames,), "float32", maxshape=(None,), chunks=True)
    value = group.create_dataset(
        "value",
        (n_frames, *shape),
        dtype,
        maxshape=(None, *shape),
        chunks=True,
    )
    if units is not None:
        value.attrs["unit"] = units
        time.attrs["unit"] = "ps"
    return group, step, time, value


def truncate_time_series(h5md, n_frames: int):
    datasets = [
        "positions_step", "positions_time", "positions",
        "total_energy_step", "total_energy_time", "total_energy",
        "potential_energy_step", "potential_energy_time", "potential_energy",
        "kinetc_energy_step", "kinetc_energy_time", "kinetc_energy",
        "bond_energy_step", "bond_energy_time", "bond_energy",
        "angle_energy_step", "angle_energy_time", "angle_energy",
        "dihedral_energy_step", "dihedral_energy_time", "dihedral_energy",
        "LJ_energy_step", "LJ_energy_time", "LJ_energy",
        "total_momentum_step", "total_momentum_time", "total_momentum",
        "temperature_step", "temperature_time", "temperature",
        "pressure_step", "pressure_time", "pressure",
        "box_step", "box_time", "box_value",
        "field_q_energy_step", "field_q_energy_time", "field_q_energy",
        "velocities_step", "velocities_time", "velocities",
        "forces_step", "forces_time", "forces",
    ]

    for name in datasets:
        if not hasattr(h5md, name):
            continue
        dset = getattr(h5md, name)
        if dset.shape[0] != n_frames:
            dset.resize((n_frames, *dset.shape[1:]))


def reconnect_for_append(
    h5md, config, molecules, bonds_2_atom1, bonds_2_atom2,
    velocity_out=False, force_out=False, charges_present=False,
):
    """Reconnect dataset references from an existing H5MD file opened in
    append mode, resize datasets for the new run, and set up unwrapping data.

    Returns ``(frame_offset, step_offset)`` — the first frame index and
    simulation-step value at which the new run should begin writing.
    """
    f = h5md.file

    # ── Reconnect top-level groups ──────────────────────────────────────
    h5md.h5md_group = f["/h5md"]
    h5md.observables = f["/observables"]
    h5md.connectivity = f["/connectivity"]
    h5md.parameters = f["/parameters"]
    h5md.particles_group = f["/particles"]
    h5md.all_particles = f["/particles/all"]

    # ── Position ────────────────────────────────────────────────────────
    h5md.positions_step = f["particles/all/position/step"]
    h5md.positions_time = f["particles/all/position/time"]
    h5md.positions = f["particles/all/position/value"]

    # ── Velocity / Force ────────────────────────────────────────────────
    if velocity_out and "velocity" in f["particles/all"]:
        h5md.velocities_step = f["particles/all/velocity/step"]
        h5md.velocities_time = f["particles/all/velocity/time"]
        h5md.velocities = f["particles/all/velocity/value"]
    if force_out and "force" in f["particles/all"]:
        h5md.forces_step = f["particles/all/force/step"]
        h5md.forces_time = f["particles/all/force/time"]
        h5md.forces = f["particles/all/force/value"]

    # ── Observable datasets ─────────────────────────────────────────────
    _obs_groups = [
        ("total_energy",     "total_energy"),
        ("kinetic_energy",   "kinetc_energy"),   # note: historical typo preserved
        ("potential_energy",  "potential_energy"),
        ("bond_energy",       "bond_energy"),
        ("angle_energy",      "angle_energy"),
        ("dihedral_energy",   "dihedral_energy"),
        ("LJ_energy",         "LJ_energy"),
        ("total_momentum",    "total_momentum"),
        ("temperature",       "temperature"),
        ("pressure",          "pressure"),
    ]
    for h5_name, attr_base in _obs_groups:
        if h5_name in f["observables"]:
            setattr(h5md, f"{attr_base}_step", f[f"observables/{h5_name}/step"])
            setattr(h5md, f"{attr_base}_time", f[f"observables/{h5_name}/time"])
            setattr(h5md, attr_base,           f[f"observables/{h5_name}/value"])

    if charges_present and "field_q_energy" in f["observables"]:
        h5md.field_q_energy_step = f["observables/field_q_energy/step"]
        h5md.field_q_energy_time = f["observables/field_q_energy/time"]
        h5md.field_q_energy = f["observables/field_q_energy/value"]

    # ── Box edges ───────────────────────────────────────────────────────
    h5md.box_step = f["particles/all/box/edges/step"]
    h5md.box_time = f["particles/all/box/edges/time"]
    h5md.box_value = f["particles/all/box/edges/value"]

    # ── Determine frame_offset and step_offset ──────────────────────────
    # Scan backwards through position data to find the last valid frame.
    # A frame is invalid if all positions are zero OR any are NaN.
    pos_data = h5md.positions
    n_existing = pos_data.shape[0]
    frame_offset = None
    for _fi in range(n_existing - 1, -1, -1):
        _frame_pos = np.array(pos_data[_fi])
        if np.all(_frame_pos == 0.0) or np.any(np.isnan(_frame_pos)):
            continue
        frame_offset = _fi + 1
        break
    if frame_offset is None:
        frame_offset = 1  # only frame 0 (step=0) was expected

    step_data = np.array(h5md.positions_step)
    step_offset = int(step_data[frame_offset - 1])

    # ── Resize all time-dependent datasets for the new run ──────────────
    n_new_frames = config.n_steps // config.n_print
    if np.mod(config.n_steps - 1, config.n_print) != 0:
        n_new_frames += 1
    if np.mod(config.n_steps, config.n_print) == 1:
        n_new_frames += 1
    if n_new_frames == config.n_steps:
        n_new_frames += 1
    new_total = frame_offset + n_new_frames

    _datasets_to_resize = [
        "positions_step", "positions_time", "positions",
        "total_energy_step", "total_energy_time", "total_energy",
        "potential_energy_step", "potential_energy_time", "potential_energy",
        "kinetc_energy_step", "kinetc_energy_time", "kinetc_energy",
        "bond_energy_step", "bond_energy_time", "bond_energy",
        "angle_energy_step", "angle_energy_time", "angle_energy",
        "dihedral_energy_step", "dihedral_energy_time", "dihedral_energy",
        "LJ_energy_step", "LJ_energy_time", "LJ_energy",
        "total_momentum_step", "total_momentum_time", "total_momentum",
        "temperature_step", "temperature_time", "temperature",
        "pressure_step", "pressure_time", "pressure",
        "box_step", "box_time", "box_value",
        "field_q_energy_step", "field_q_energy_time", "field_q_energy",
        "velocities_step", "velocities_time", "velocities",
        "forces_step", "forces_time", "forces",
    ]
    for name in _datasets_to_resize:
        if not hasattr(h5md, name):
            continue
        dset = getattr(h5md, name)
        if dset.shape[0] < new_total:
            dset.resize((new_total, *dset.shape[1:]))

    # ── Pre-compute unwrapping data ─────────────────────────────────────
    if getattr(config, "unwrap_output", True):
        small_anchors, small_others, large_mol_atoms, adjacency = _build_unwrap_data(
            molecules, bonds_2_atom1, bonds_2_atom2,
        )
        h5md._unwrap_small_anchors = small_anchors
        h5md._unwrap_small_others = small_others
        h5md._unwrap_large_mol_atoms = large_mol_atoms
        h5md._unwrap_adjacency = adjacency

    Logger.rank0.info(
        "Append mode: continuing from frame %d (step %d). "
        "Datasets resized to %d frames.",
        frame_offset, step_offset, new_total,
    )
    return frame_offset, step_offset


def store_static(
    h5md,
    names,
    types,
    indices,
    config,
    bonds_2_atom1,
    bonds_2_atom2,
    topol_mols,
    molecules=None,
    velocity_out=False,
    force_out=False,
    charges=None,
    resnames=None,
):
    dtype = h5md.float_dtype

    h5md_group = h5md.file.create_group("/h5md")
    h5md.h5md_group = h5md_group
    h5md.observables = h5md.file.create_group("/observables")
    h5md.connectivity = h5md.file.create_group("/connectivity")
    h5md.parameters = h5md.file.create_group("/parameters")

    h5md_group.attrs["version"] = np.array([1, 1], dtype=int)
    author_group = h5md_group.create_group("author")
    author_group.attrs["name"] = np.bytes_(getpass.getuser())
    creator_group = h5md_group.create_group("creator")
    creator_group.attrs["name"] = np.bytes_("Diff-MD")

    creator_group.attrs["version"] = np.bytes_("0.0")

    h5md.particles_group = h5md.file.create_group("/particles")
    h5md.all_particles = h5md.particles_group.create_group("all")
    mass = h5md.all_particles.create_dataset("mass", (config.n_particles, 1), dtype)
    mass[...] = np.asarray(config.mass)

    if charges is not None:
        charge_vals = np.asarray(charges).ravel()
        charge = h5md.all_particles.create_dataset(
            "charge", (config.n_particles,), dtype="float32"
        )
        charge[...] = charge_vals

    box = h5md.all_particles.create_group("box")
    box.attrs["dimension"] = 3
    box.attrs["boundary"] = np.array(
        [np.bytes_(s) for s in 3 * ["periodic"]], dtype="S8"
    )

    n_frames = config.n_steps // config.n_print
    if np.mod(config.n_steps - 1, config.n_print) != 0:
        n_frames += 1
    if np.mod(config.n_steps, config.n_print) == 1:
        n_frames += 1
    if n_frames == config.n_steps:
        n_frames += 1

    species = h5md.all_particles.create_dataset(
        "species", (config.n_particles,), dtype="i"
    )
    (
        _,
        h5md.positions_step,
        h5md.positions_time,
        h5md.positions,
    ) = setup_time_dependent_element(
        "position",
        h5md.all_particles,
        n_frames,
        (config.n_particles, 3),
        dtype,
        units="nm",
    )
    if velocity_out:
        (
            _,
            h5md.velocities_step,
            h5md.velocities_time,
            h5md.velocities,
        ) = setup_time_dependent_element(
            "velocity",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="nm ps-1",
        )
    if force_out:
        (
            _,
            h5md.forces_step,
            h5md.forces_time,
            h5md.forces,
        ) = setup_time_dependent_element(
            "force",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="kJ mol-1 nm-1",
        )
    (
        _,
        h5md.total_energy_step,
        h5md.total_energy_time,
        h5md.total_energy,
    ) = setup_time_dependent_element(
        "total_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.kinetc_energy_step,
        h5md.kinetc_energy_time,
        h5md.kinetc_energy,
    ) = setup_time_dependent_element(
        "kinetic_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.potential_energy_step,
        h5md.potential_energy_time,
        h5md.potential_energy,
    ) = setup_time_dependent_element(  # noqa: E501
        "potential_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.bond_energy_step,
        h5md.bond_energy_time,
        h5md.bond_energy,
    ) = setup_time_dependent_element(
        "bond_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.angle_energy_step,
        h5md.angle_energy_time,
        h5md.angle_energy,
    ) = setup_time_dependent_element(
        "angle_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.dihedral_energy_step,
        h5md.dihedral_energy_time,
        h5md.dihedral_energy,
    ) = setup_time_dependent_element(
        "dihedral_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.LJ_energy_step,
        h5md.LJ_energy_time,
        h5md.LJ_energy,
    ) = setup_time_dependent_element(
        "LJ_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )    
    if charges is not None:
        (
            _,
            h5md.field_q_energy_step,
            h5md.field_q_energy_time,
            h5md.field_q_energy,
        ) = setup_time_dependent_element(
            "field_q_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
        )
        # (
        #     _,
        #     h5md.elec_ener_real_step,
        #     h5md.elec_ener_real_time,
        #     h5md.elec_ener_real,
        # ) = setup_time_dependent_element(
        #     "q_energy_real", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
        # )
        # (
        #     _,
        #     h5md.elec_ener_fourrier_step,
        #     h5md.elec_ener_fourrier_time,
        #     h5md.elec_ener_fourrier,
        # ) = setup_time_dependent_element(
        #     "q_energy_fourrier", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
        # )

    (
        _,
        h5md.total_momentum_step,
        h5md.total_momentum_time,
        h5md.total_momentum,
    ) = setup_time_dependent_element(  # noqa: E501
        "total_momentum",
        h5md.observables,
        n_frames,
        (3,),
        dtype,
        units="nm g ps-1 mol-1",
    )
    # (
    #     _,
    #     h5md.angular_momentum_step,
    #     h5md.angular_momentum_time,
    #     h5md.angular_momentum,
    # ) = setup_time_dependent_element(  # noqa: E501
    #     "angular_momentum",
    #     h5md.observables,
    #     n_frames,
    #     (3,),
    #     dtype,
    #     units="nm+2 g ps-1 mol-1",
    # )
    # (
    #     _,
    #     h5md.torque_step,
    #     h5md.torque_time,
    #     h5md.torque,
    # ) = setup_time_dependent_element(  # noqa: E501
    #     "torque",
    #     h5md.observables,
    #     n_frames,
    #     (3,),
    #     dtype,
    #     units="kJ nm+2 mol-1",
    # )
    (
        _,
        h5md.temperature_step,
        h5md.temperature_time,
        h5md.temperature,
    ) = setup_time_dependent_element(
        "temperature", h5md.observables, n_frames, (3,), dtype, units="K"
    )

    (
        _,
        h5md.pressure_step,
        h5md.pressure_time,
        h5md.pressure,
    ) = setup_time_dependent_element(
        "pressure", h5md.observables, n_frames, (3,), dtype, units="bar"
    )
    (
        _,
        h5md.box_step,
        h5md.box_time,
        h5md.box_value,
    ) = setup_time_dependent_element(
        "edges", box, n_frames, (3, 3), "float32", units="nm"
    )

    species[:] = types[:]

    # Store flat restart-compatible datasets so that the output h5 can be
    # used directly as a coordinate input file for restart runs.
    h5md.file.create_dataset("indices", data=indices)
    h5md.file.create_dataset("types", data=types)
    h5md.file.create_dataset("names", data=names)
    h5md.file.create_dataset("masses", data=np.asarray(config.mass).ravel())
    if molecules is not None:
        h5md.file.create_dataset("molecules", data=molecules)
    if resnames is not None:
        h5md.file.create_dataset("resnames", data=resnames)
    if charges is not None:
        h5md.file.create_dataset("charge", data=np.asarray(charges).ravel())
    h5md.file.attrs["box"] = np.asarray(config.box_size)

    h5md.parameters.attrs["config.toml"] = np.bytes_(str(config))
    vmd_group = h5md.parameters.create_group("vmd_structure")
    index_of_species = vmd_group.create_dataset(
        "indexOfSpecies", (config.n_types,), "i"
    )
    index_of_species[:] = np.array(list(range(config.n_types)))

    # VMD-h5mdplugin maximum name/type name length is 16 characters (for
    # whatever reason [VMD internals?]).
    name_dataset = vmd_group.create_dataset("name", (config.n_types,), "S16")

    if molecules is not None:
        resid_dataset = vmd_group.create_dataset("resid", (config.n_particles,), "i")
        resid_dataset[:] = molecules

        unique_mols = np.unique(molecules)
        resname_dataset = vmd_group.create_dataset("resname", (len(unique_mols),), "S8")

        prev = 0
        for resname, n in topol_mols:
            resname_dataset[(unique_mols >= prev) & (unique_mols < prev + n)] = (
                np.bytes_(resname)
            )
            prev += n


    _, name_idx = np.unique(names, return_index=True)
    unique_names = names[np.sort(name_idx)]

    for i, n in enumerate(unique_names):
        name_dataset[i] = np.bytes_(n.decode("utf-8")[:16])


    total_bonds = len(bonds_2_atom1)
    bonds_from = vmd_group.create_dataset("bond_from", (total_bonds,), "i")
    bonds_to = vmd_group.create_dataset("bond_to", (total_bonds,), "i")
    for i in range(total_bonds):
        a = bonds_2_atom1[i]
        b = bonds_2_atom2[i]
        bonds_from[i] = indices[a] + 1
        bonds_to[i] = indices[b] + 1

    # Pre-compute unwrapping data for whole-molecule imaging
    if getattr(config, "unwrap_output", True):
        small_anchors, small_others, large_mol_atoms, adjacency = _build_unwrap_data(
            molecules, bonds_2_atom1, bonds_2_atom2,
        )
        h5md._unwrap_small_anchors = small_anchors
        h5md._unwrap_small_others = small_others
        h5md._unwrap_large_mol_atoms = large_mol_atoms
        h5md._unwrap_adjacency = adjacency
        n_small = len(small_anchors) if small_anchors is not None else 0
        n_large = len(large_mol_atoms) if large_mol_atoms else 0
        if n_small or n_large:
            Logger.rank0.info(
                f"Whole-molecule unwrapping enabled for output "
                f"({n_small} small + {n_large} large molecules, {total_bonds} bonds)."
            )
    else:
        Logger.rank0.info("Whole-molecule unwrapping disabled (unwrap_output = false).")


def store_data(
    h5md,
    step,
    frame,
    indices,
    positions,
    velocities,
    forces,
    temperature,
    pressure,
    kinetic_energy,
    bond2_energy,
    bond3_energy,
    bond4_energy,
    LJ_energy,
    field_q_energy,
    # elec_ener_real,
    # elec_ener_fourrier,
    config,
    velocity_out=False,
    force_out=False,
    charge_out=False,
    dump_per_particle=False,
):
    for dset in (
        h5md.positions_step,
        h5md.total_energy_step,
        h5md.potential_energy,
        h5md.kinetc_energy_step,
        h5md.bond_energy_step,
        h5md.angle_energy_step,
        h5md.dihedral_energy_step,
        h5md.LJ_energy_step,
        h5md.total_momentum_step,
        # h5md.angular_momentum_step,
        # h5md.torque_step,
        h5md.temperature_step,
        h5md.pressure_step,
        h5md.box_step,
    ):
        dset[frame] = step

    for dset in (
        h5md.positions_time,
        h5md.total_energy_time,
        h5md.potential_energy_time,
        h5md.kinetc_energy_time,
        h5md.bond_energy_time,
        h5md.angle_energy_time,
        h5md.dihedral_energy_time,
        h5md.LJ_energy_time,
        h5md.total_momentum_time,
        # h5md.angular_momentum_time,
        # h5md.torque_time,
        h5md.temperature_time,
        h5md.pressure_time,
        h5md.box_time,
    ):
        dset[frame] = step * config.outer_ts

    if velocity_out:
        h5md.velocities_step[frame] = step
        h5md.velocities_time[frame] = step * config.outer_ts
    if force_out:
        h5md.forces_step[frame] = step
        h5md.forces_time[frame] = step * config.outer_ts
    if charge_out:
        h5md.field_q_energy_step[frame] = step
        h5md.field_q_energy_time[frame] = step * config.outer_ts
        # h5md.elec_ener_real_step[frame] = step
        # h5md.elec_ener_real_time[frame] = step * config.outer_ts
        # h5md.elec_ener_fourrier_step[frame] = step
        # h5md.elec_ener_fourrier_time[frame] = step * config.outer_ts

    ind_sort = np.argsort(indices)
    # positions, velocities and forces are already np.ndarrays
    # Apply whole-molecule unwrapping so no molecule is split across PBC
    out_positions = positions
    if h5md._unwrap_adjacency is not None:
        out_positions = unwrap_molecules(
            positions, config.box_size,
            h5md._unwrap_small_anchors, h5md._unwrap_small_others,
            h5md._unwrap_large_mol_atoms, h5md._unwrap_adjacency,
        )
    h5md.positions[frame, indices[ind_sort]] = out_positions[ind_sort]

    if velocity_out:
        h5md.velocities[frame, indices[ind_sort]] = velocities[ind_sort]
    if force_out:
        h5md.forces[frame, indices[ind_sort]] = forces[ind_sort]
    if charge_out:
        h5md.field_q_energy[frame] = field_q_energy
        # h5md.elec_ener_real[frame] = elec_ener_real
        # h5md.elec_ener_fourrier[frame] = elec_ener_fourrier


    potential_energy = (
        bond2_energy + bond3_energy + bond4_energy + LJ_energy + field_q_energy
    )

    total_momentum = np.sum(config.mass * velocities, axis=0)
    # angular_momentum = config.mass * np.sum(np.cross(positions, velocities), axis=0)
    # torque = config.mass * np.sum(np.cross(positions, forces), axis=0)

    h5md.total_energy[frame] = kinetic_energy + potential_energy
    h5md.potential_energy[frame] = potential_energy
    h5md.kinetc_energy[frame] = kinetic_energy
    h5md.bond_energy[frame] = bond2_energy
    h5md.angle_energy[frame] = bond3_energy
    h5md.dihedral_energy[frame] = bond4_energy
    h5md.LJ_energy[frame] = LJ_energy
    h5md.total_momentum[frame, :] = total_momentum
    # h5md.angular_momentum[frame, :] = angular_momentum
    # h5md.torque[frame, :] = torque
    h5md.temperature[frame] = temperature
    h5md.pressure[frame] = pressure
    for d in range(3):
        h5md.box_value[frame, d, d] = config.box_size[d]

    fmt_ = [
        "step",
        "time",
        "temp",
        "tot E",
        "kin E",
        "pot E",
        "LJ E",
        "Elec E",
        "bond E",
        "ang E",
        "dih E",
        "Px",
        "Py",
        "Pz",
    ]
    fmt_ = np.array(fmt_)

    # create mask to show only energies != 0
    en_array = np.array(
        [
            LJ_energy,
            field_q_energy,
            bond2_energy,
            bond3_energy,
            bond4_energy,
        ]
    )
    mask = np.full_like(fmt_, True, dtype=bool)
    mask[range(6, 11)] = en_array != 0.0

    divide_by = 1.0
    if dump_per_particle:
        for i in range(3, 9):
            fmt_[i] = fmt_[i][:-2] + "E/N"
        fmt_[-1] += "/N"
        divide_by = config.n_particles
    total_energy = kinetic_energy + potential_energy

    header_ = fmt_[mask].shape[0] * "{:>13}"
    header = header_.format(*fmt_[mask])

    data_fmt = f'{"{:13}"}{(fmt_[mask].shape[0] -1 ) * "{:13.5g}" }'
    all_data = (
        step,
        config.outer_ts * step,
        temperature,
        total_energy / divide_by,
        kinetic_energy / divide_by,
        potential_energy / divide_by,
        LJ_energy / divide_by,
        field_q_energy / divide_by,
        bond2_energy / divide_by,
        bond3_energy / divide_by,
        bond4_energy / divide_by,
        total_momentum[0] / divide_by,
        total_momentum[1] / divide_by,
        total_momentum[2] / divide_by,
    )
    data = data_fmt.format(*[val for i, val in enumerate(all_data) if mask[i]])
    Logger.rank0.log(logging.INFO, ("\n" + header + "\n" + data))


def write_full_trajectory(
    h5md,
    trj_dict,
    indices,
    config,
    velocity_out=False,
    force_out=False,
    charge_out=True,
):
    # Extract exactly the 11 frame-level keys that this writer expects.
    # Extra keys (e.g. "pressure" from NPT runs) are intentionally skipped
    # so the unpacking does not break.
    _expected_keys = [
        "angle energy", "bond energy", "dihedral energy", "elec energy",
        "LJ energy", "forces", "kinetic energy", "temperature",
        "positions", "velocities", "box",
    ]
    _frame_values = [trj_dict[k] for k in _expected_keys if k in trj_dict]
    for frame, (  # type: ignore
        angle_energy,
        bond_energy,
        dih_energy,
        elec_energy,
        LJ_energy,
        forces,
        kinetic_energy,
        temperature,
        positions,
        velocities,
        box_sizes,
    ) in enumerate(zip(*_frame_values)):
        for dset in (
            h5md.positions_step,
            h5md.total_energy_step,
            h5md.potential_energy,
            h5md.kinetc_energy_step,
            h5md.bond_energy_step,
            h5md.angle_energy_step,
            h5md.dihedral_energy_step,
            h5md.LJ_energy_step,
            h5md.total_momentum_step,
            h5md.temperature_step,
            h5md.box_step,
        ):
            dset[frame] = frame

        for dset in (
            h5md.positions_time,
            h5md.total_energy_time,
            h5md.potential_energy_time,
            h5md.kinetc_energy_time,
            h5md.bond_energy_time,
            h5md.angle_energy_time,
            h5md.dihedral_energy_time,
            h5md.LJ_energy_time,
            h5md.total_momentum_time,
            h5md.temperature_time,
            h5md.box_time,
        ):
            dset[frame] = frame * config.outer_ts

        if velocity_out:
            h5md.velocities_step[frame] = frame
            h5md.velocities_time[frame] = frame * config.outer_ts
        if force_out:
            h5md.forces_step[frame] = frame
            h5md.forces_time[frame] = frame * config.outer_ts
        if charge_out:
            h5md.field_q_energy_step[frame] = frame
            h5md.field_q_energy_time[frame] = frame * config.outer_ts
            # h5md.elec_ener_real_step[frame] = frame
            # h5md.elec_ener_real_time[frame] = frame * config.outer_ts
            # h5md.elec_ener_fourrier_step[frame] = frame
            # h5md.elec_ener_fourrier_time[frame] = frame * config.outer_ts

        ind_sort = np.argsort(indices)
        # Apply whole-molecule unwrapping so no molecule is split across PBC
        frame_positions = np.asarray(positions)
        if h5md._unwrap_adjacency is not None:
            frame_positions = unwrap_molecules(
                frame_positions, box_sizes,
                h5md._unwrap_small_anchors, h5md._unwrap_small_others,
                h5md._unwrap_large_mol_atoms, h5md._unwrap_adjacency,
            )
        h5md.positions[frame, indices[ind_sort]] = frame_positions[ind_sort]

        if velocity_out:
            h5md.velocities[frame, indices[ind_sort]] = np.asarray(velocities)[ind_sort]
        if force_out:
            h5md.forces[frame, indices[ind_sort]] = np.asarray(forces)[ind_sort]
        if charge_out:
            h5md.field_q_energy[frame] = elec_energy


        potential_energy = (
            bond_energy + angle_energy + dih_energy + LJ_energy + elec_energy
        )

        total_momentum = np.sum(config.mass * np.asarray(velocities), axis=0)

        h5md.total_energy[frame] = kinetic_energy + potential_energy
        h5md.potential_energy[frame] = potential_energy
        h5md.kinetc_energy[frame] = kinetic_energy
        h5md.bond_energy[frame] = bond_energy
        h5md.angle_energy[frame] = angle_energy
        h5md.dihedral_energy[frame] = dih_energy
        h5md.LJ_energy[frame] = LJ_energy
        h5md.total_momentum[frame, :] = total_momentum
        h5md.temperature[frame] = temperature
        for d in range(3):
            h5md.box_value[frame, d, d] = box_sizes[d]
