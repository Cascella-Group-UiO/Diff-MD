import os
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array

from .config import read_toml, Config
from .logger import Logger
from .models import GeneralModel

# from .config import config, n_particles, nrexcl 
@struct.dataclass
class Topology:
    # molecules_flag: bool = struct.field(pytree_node=False, default=False)
    molecules: dict[str, int] = struct.field(pytree_node=False, default=False)
    bonds: int = struct.field(pytree_node=False, default=False)
    angles: int = struct.field(pytree_node=False, default=False)
    dihedrals: int = struct.field(pytree_node=False, default=False)
    # ``proper_torsions`` is a count (like ``dihedrals``) that gates a SECOND
    # bonded path used only when ``ff_family == "martini"`` and the chain
    # TOML mixes amber-like proper-torsion entries (``[i,j,k,l,funct,
    # phi0_deg,k,multiplicity]``) alongside Martini coefficient-format
    # dihedrals.  The actual data lives in ``bonds_pt`` below.  For
    # ff_family in ("amber_like", "amber19sb") this stays 0 — those
    # families already pack amber-style propers into ``bonds_4``.
    proper_torsions: int = struct.field(pytree_node=False, default=0)
    impropers: int = struct.field(pytree_node=False, default=False)
    cmaps: int = struct.field(pytree_node=False, default=False)
    # ``True`` when every angle in ``bonds_3`` carries GROMACS function type 1
    # (or the angle table is empty).  Used by ``force.get_angle_energy_and_forces``
    # as a Python-time switch to a single-branch harmonic kernel, so atomistic
    # FFs (which the parser already restricts to type 1) compile a graph that
    # is bit-identical to the pre-2026-05-13 legacy.
    angle_uses_only_harmonic: bool = struct.field(pytree_node=False, default=True)
    restraints: Optional[Array] = None
    bonds_2: Optional[tuple[Array, ...]] = None
    bonds_3: Optional[tuple[Array, ...]] = None
    bonds_4: Optional[tuple[Array, ...]] = None
    bonds_pt: Optional[tuple[Array, ...]] = None
    bonds_d: Optional[tuple[Array, ...]] = None
    bonds_impr: Optional[tuple[Array, ...]] = None
    # CMAP backbone correction (only populated for ff_family=='amber19sb'
    # when the chain TOML carries a [cmap] section).  ``bonds_cmap`` is
    # ``(atom1, atom2, atom3, atom4, atom5, grid_id)`` with all six
    # arrays of shape ``(n_cmap,)``.  ``cmap_coefs`` is the stacked
    # bicubic-coefficient bank, shape ``(n_grids, G, G, 4, 4)``,
    # carried as a JAX leaf so ``coefs[grid_id, i, j]`` gathers
    # cleanly inside jit.
    bonds_cmap: Optional[tuple[Array, ...]] = None
    cmap_coefs: Optional[Array] = None
    excluded_pairs: Optional[tuple[Array, ...]] = None
    one_four_pairs: Optional[tuple[Array, ...]] = None


def get_topol(
    file_path: str, molecules: np.ndarray, config: Config, model: Optional[GeneralModel] = None
) -> Topology:
    try:
        toml_topol = read_toml(file_path)
        topol_atoms = 0

        # Check if we have single "itp" files and add their keys to topol
        if os.path.dirname(file_path) == "":
            file_path = "./" + file_path
        if "include" in toml_topol["system"]:
            for file in toml_topol["system"]["include"]:
                path = f"{os.path.dirname(file_path)}/{file}"
                itps = read_toml(path)
                for mol, itp in itps.items():
                    toml_topol[mol] = itp
                    for mol_name in toml_topol["system"]["molecules"]:
                        if mol_name[0] == mol:
                            topol_atoms += mol_name[1] * toml_topol[mol]["atomnum"]
    except Exception as e:
        Logger.rank0.error(f"Unable to parse topology '{file_path}'.", exc_info=e)
        exit()

    if topol_atoms != len(molecules):
        Logger.rank0.error(
            f"Number of particles defined in '{file_path}' ({topol_atoms}) does not match"
            f"number of particles present in the coordinate file ({len(molecules)})."
        )
        exit()

    topol = prepare_bonds(molecules, toml_topol, config, training=model)
    Logger.rank0.info(
        f"Topology file '{file_path}' parsed successfully.",
    )
    return topol


def find_excluded_pairs(i_atoms, j_atoms, nrexcl, num_particles):
    adjacency = [set() for _ in range(num_particles)]
    for i, j in zip(i_atoms, j_atoms):
        ii = int(i)
        jj = int(j)
        adjacency[ii].add(jj)
        adjacency[jj].add(ii)

    pairs_i = []
    pairs_j = []
    for i in range(num_particles):
        visited = {i}
        frontier = {i}
        for _ in range(1, nrexcl + 1):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= visited
            for j in next_frontier:
                if j > i:
                    pairs_i.append(i)
                    pairs_j.append(j)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

    # Keep index arrays integer-typed even when lists are empty.
    return (
        jnp.asarray(pairs_i, dtype=jnp.int32),
        jnp.asarray(pairs_j, dtype=jnp.int32),
    )


def find_pairs_at_bond_level(i_atoms, j_atoms, bond_level, num_particles):
    adjacency = [set() for _ in range(num_particles)]
    for i, j in zip(i_atoms, j_atoms):
        ii = int(i)
        jj = int(j)
        adjacency[ii].add(jj)
        adjacency[jj].add(ii)

    pairs_i = []
    pairs_j = []
    for i in range(num_particles):
        visited = {i}
        frontier = {i}
        for level in range(1, bond_level + 1):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= visited
            if level == bond_level:
                for j in next_frontier:
                    if j > i:
                        pairs_i.append(i)
                        pairs_j.append(j)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

    # Keep index arrays integer-typed even when lists are empty.
    return (
        jnp.asarray(pairs_i, dtype=jnp.int32),
        jnp.asarray(pairs_j, dtype=jnp.int32),
    )


def prepare_index_based_bonds(molecules, topol, config, training):
    bonds = []
    angles = []
    dihedrals = []
    proper_torsions = []
    impropers = []
    cmaps = []
    restraints = []
    explicit_exclusions = set()


    ff_family = str(config.ff_family).lower()
    different_molecules = np.unique(molecules) 
    for mol in different_molecules: # Iterate trough all individual molecules in the system
        resid = mol + 1 # Offset the index  
        top_summary = topol["system"]["molecules"] # Array with name of molecules and number of such molecules in the system. Similar to the molecules section in GROMACS topol file. 
        resname = None
        test_mol_number = 0
        for molname in top_summary: 
            test_mol_number += molname[1]
            if resid <= test_mol_number:
                resname = molname[0]
                break

        if resname is None:
            break

        # resnames += resname * topol[resname]["n_atoms"]

        # Take index of restrained atoms    
        if "restraints" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for i in topol[resname]["restraints"][0]:
                index_i = i - 1 + first_id  
                restraints.append(index_i)   

        if "bonds" in topol[resname]: # Information in topol[resname] is added in get_topol
            first_id = np.where(molecules == mol)[0][0]
            for bond in topol[resname]["bonds"]:
                index_i = bond[0] - 1 + first_id
                index_j = bond[1] - 1 + first_id
                # bond[2] is the bond type, inherited by the itp format,
                # we don't use it
                if training.bonds:
                    bonds.append([index_i, index_j])
                    continue
                equilibrium = bond[3]
                strength = bond[4]
                bonds.append([index_i, index_j, equilibrium, strength])

        if "exclusions" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for row in topol[resname]["exclusions"]:
                if len(row) < 2:
                    continue

                base = int(row[0]) - 1 + first_id
                for partner in row[1:]:
                    other = int(partner) - 1 + first_id
                    if other == base:
                        continue
                    i_ex, j_ex = (base, other) if base < other else (other, base)
                    explicit_exclusions.add((i_ex, j_ex))

        if "angles" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for angle in topol[resname]["angles"]:
                index_i = angle[0] - 1 + first_id
                index_j = angle[1] - 1 + first_id
                index_k = angle[2] - 1 + first_id
                # GROMACS function-type column (column 4 in the itp row,
                # index 3 here).  Supported: 1 = harmonic in theta,
                # 2 = G96 cosine, 10 = restricted bending (ReB).
                # Anything else is a hard error — better to fail at parse
                # time than silently treat it as harmonic and produce
                # wrong physics.
                angle_type = int(angle[3])
                if ff_family in ("amber_like", "amber19sb") and angle_type != 1:
                    Logger.rank0.error(
                        f"Angle type {angle_type} in molecule '{resname}' is not "
                        f"valid for atomistic ff_family='{ff_family}'. Atomistic "
                        "families currently accept only type 1 harmonic angles; "
                        "types 2 and 10 are intended for Martini/CG topologies. "
                        f"Got row: {angle}."
                    )
                    exit()
                if angle_type not in (1, 2, 10):
                    Logger.rank0.error(
                        f"Angle type {angle_type} in molecule '{resname}' is not "
                        f"supported. Got row: {angle}. Supported types: "
                        f"1 (harmonic), 2 (G96 cosine), 10 (restricted bending / ReB)."
                    )
                    exit()
                if training.angles:
                    angles.append([index_i, index_j, index_k, angle_type])
                    continue
                equilibrium = np.radians(angle[4])
                strength = angle[5]
                angles.append([index_i, index_j, index_k, angle_type,
                               equilibrium, strength])

        if "dihedrals" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            seen_dihedral_quads = set()  # Track unique atom quadruplets for training dedup
            for dih in topol[resname]["dihedrals"]:
                index_i = dih[0] - 1 + first_id
                index_j = dih[1] - 1 + first_id
                index_k = dih[2] - 1 + first_id
                index_l = dih[3] - 1 + first_id

                if len(dih) < 5:
                    Logger.rank0.error(
                        f"Invalid dihedral entry for molecule '{resname}': {dih}."
                    )
                    exit()

                # Auto-detect amber-style proper-torsion entries inside a
                # Martini topology.  Martini's native dihedral format has a
                # coefficient *list* at dih[4] or dih[5]; amber-style entries
                # are flat scalars.  When ff_family == "martini" and the
                # entry is not coefficient-shaped, reuse the amber_like
                # parser below but route propers into ``proper_torsions``
                # so they coexist with native Martini coefficient dihedrals.
                if ff_family == "martini":
                    coeff_at_5 = len(dih) >= 6 and isinstance(dih[5], list)
                    coeff_at_4 = len(dih) >= 5 and isinstance(dih[4], list)
                    looks_amber_like = not (coeff_at_5 or coeff_at_4)
                else:
                    looks_amber_like = False

                use_amber_parser = ff_family == "amber_like" or (
                    ff_family == "martini" and looks_amber_like
                )
                proper_target = (
                    proper_torsions if ff_family == "martini" else dihedrals
                )

                if use_amber_parser:
                    # GROMACS/ITP-like formats:
                    #   Proper legacy: [i, j, k, l, funct, phi0_deg, k, multiplicity]
                    #   Improper legacy: [i, j, k, l, funct, phi0_deg, k] with funct in {2, 4}
                    #   Protein-style: [i, j, k, l, funct, improper_bool, phi0_deg, k, multiplicity]
                    funct = int(dih[4])

                    has_improper_flag = len(dih) >= 9 and isinstance(dih[5], (bool, np.bool_))
                    if has_improper_flag:
                        is_improper = bool(dih[5])
                        if is_improper:
                            equilibrium = np.radians(dih[6])
                            strength = dih[7]
                            multiplicity = int(dih[8]) if len(dih) >= 9 else 1
                            is_periodic = funct == 4
                            impropers.append(
                                [
                                    index_i,
                                    index_j,
                                    index_k,
                                    index_l,
                                    equilibrium,
                                    strength,
                                    multiplicity,
                                    is_periodic,
                                ]
                            )
                            continue

                        if training.dihedrals:
                            quad = (index_i, index_j, index_k, index_l)
                            if quad not in seen_dihedral_quads:
                                seen_dihedral_quads.add(quad)
                                proper_target.append([index_i, index_j, index_k, index_l])
                            continue

                        phase = np.radians(dih[6])
                        strength = dih[7]
                        multiplicity = int(dih[8])
                        proper_target.append(
                            [index_i, index_j, index_k, index_l, phase, strength, multiplicity]
                        )
                        continue

                    if funct in (2, 4):
                        if len(dih) < 7:
                            Logger.rank0.error(
                                "Improper dihedral entry requires [i,j,k,l,funct,phi0_deg,k] "
                                "with funct in {2,4}. "
                                f"Got: {dih}."
                            )
                            exit()
                        equilibrium = np.radians(dih[5])
                        strength = dih[6]
                        multiplicity = int(dih[7]) if (funct == 4 and len(dih) >= 8) else 1
                        is_periodic = funct == 4
                        impropers.append(
                            [
                                index_i,
                                index_j,
                                index_k,
                                index_l,
                                equilibrium,
                                strength,
                                multiplicity,
                                is_periodic,
                            ]
                        )
                        continue

                    if len(dih) < 8:
                        Logger.rank0.error(
                            "Proper dihedral entry requires "
                            "[i,j,k,l,funct,phi0_deg,k,multiplicity] or "
                            "[i,j,k,l,funct,improper_bool,phi0_deg,k,multiplicity]. "
                            f"Got: {dih}."
                        )
                        exit()

                    if training.dihedrals:
                        quad = (index_i, index_j, index_k, index_l)
                        if quad not in seen_dihedral_quads:
                            seen_dihedral_quads.add(quad)
                            proper_target.append([index_i, index_j, index_k, index_l])
                        continue

                    phase = np.radians(dih[5])
                    strength = dih[6]
                    multiplicity = int(dih[7])
                    proper_target.append(
                        [index_i, index_j, index_k, index_l, phase, strength, multiplicity]
                    )
                    continue

                # Legacy / Martini-compatible formats:
                #   [i, j, k, l, type, coeff_matrix]  -> proper
                #   [i, j, k, l, coeff_matrix]         -> proper
                #   [i, j, k, l, 2, phi0_deg, k]       -> improper
                if len(dih) >= 6 and dih[4] == 2:
                    if len(dih) < 7:
                        Logger.rank0.error(
                            f"Improper dihedral entry requires [i,j,k,l,2,phi0_deg,k]. Got: {dih}."
                        )
                        exit()
                    equilibrium = np.radians(dih[5])
                    strength = dih[6]
                    impropers.append(
                        [index_i, index_j, index_k, index_l, equilibrium, strength]
                    )
                    continue

                if len(dih) >= 6 and isinstance(dih[5], list):
                    coeff = dih[5]
                elif len(dih) >= 5 and isinstance(dih[4], list):
                    coeff = dih[4]
                else:
                    Logger.rank0.error(
                        "Proper dihedral entry must include a coefficient matrix, "
                        f"either as [i,j,k,l,type,coeff] or [i,j,k,l,coeff]. Got: {dih}."
                    )
                    exit()

                if training.dihedrals:
                    # coeffs are defined inside the simulator ==>
                    # we just need the indices and then filter out by type
                    quad = (index_i, index_j, index_k, index_l)
                    if quad not in seen_dihedral_quads:
                        seen_dihedral_quads.add(quad)
                        dihedrals.append([index_i, index_j, index_k, index_l])
                    continue

                dihedrals.append([index_i, index_j, index_k, index_l, coeff, 0])
            if (
                not training.dihedrals
                and ff_family != "amber_like"
                and len(dihedrals) > 0
            ):
                # TODO: provide 'is_last' in the topology?
                dihedrals[-1][-1] = 1

        if "impropers" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for impr in topol[resname]["impropers"]:
                # Supported formats:
                #   [i, j, k, l, phi0_deg, k_improper]
                #   [i, j, k, l, funct, phi0_deg, k_improper]
                #   [i, j, k, l, funct, phi0_deg, k_improper, multiplicity]
                if len(impr) < 6:
                    Logger.rank0.error(
                        f"Invalid improper entry for molecule '{resname}': {impr}."
                    )
                    exit()

                index_i = impr[0] - 1 + first_id
                index_j = impr[1] - 1 + first_id
                index_k = impr[2] - 1 + first_id
                index_l = impr[3] - 1 + first_id

                if len(impr) >= 7:
                    funct = int(impr[4])
                    equilibrium = np.radians(impr[5])
                    strength = impr[6]
                    multiplicity = int(impr[7]) if (funct == 4 and len(impr) >= 8) else 1
                    is_periodic = funct == 4
                else:
                    equilibrium = np.radians(impr[4])
                    strength = impr[5]
                    multiplicity = 1
                    is_periodic = False
                impropers.append(
                    [
                        index_i,
                        index_j,
                        index_k,
                        index_l,
                        equilibrium,
                        strength,
                        multiplicity,
                        is_periodic,
                    ]
                )

        # CMAP backbone correction (ff19SB-style).  Only consumed when
        # ``ff_family == "amber19sb"`` AND the run-time options.toml
        # provided a ``[cmap]`` block (i.e., ``config.cmap_grid_bank``
        # is not None).  Each entry is
        #   [i, j, k, l, m, residue_name]
        # with 1-indexed atom IDs (same convention as ``dihedrals``).
        # The residue_name keys into ``config.cmap_grid_bank`` to pick
        # the right 24x24 energy grid.
        if (
            ff_family == "amber19sb"
            and getattr(config, "cmap_grid_bank", None) is not None
            and "cmap" in topol[resname]
        ):
            first_id = np.where(molecules == mol)[0][0]
            grid_bank = config.cmap_grid_bank
            for cm in topol[resname]["cmap"]:
                if len(cm) < 6:
                    Logger.rank0.error(
                        f"Invalid cmap entry for molecule '{resname}': {cm}. "
                        "Expected [i, j, k, l, m, residue_name]."
                    )
                    exit()
                index_i = int(cm[0]) - 1 + first_id
                index_j = int(cm[1]) - 1 + first_id
                index_k = int(cm[2]) - 1 + first_id
                index_l = int(cm[3]) - 1 + first_id
                index_m = int(cm[4]) - 1 + first_id
                residue_name = str(cm[5])
                if residue_name not in grid_bank.grids:
                    Logger.rank0.error(
                        f"CMAP residue '{residue_name}' (in molecule "
                        f"'{resname}') has no grid in [cmap.grids].  "
                        f"Known: {sorted(grid_bank.grids)}."
                    )
                    exit()
                cmaps.append(
                    [index_i, index_j, index_k, index_l, index_m, residue_name]
                )

    return bonds, angles, dihedrals, proper_torsions, impropers, cmaps, restraints, explicit_exclusions


def prepare_bonds(molecules, topol, config, training=None):
    if training is None:
        training = GeneralModel()
    (
        bonds,
        angles,
        dihedrals,
        proper_torsions,
        impropers,
        cmaps,
        restraints,
        explicit_exclusions,
    ) = prepare_index_based_bonds(molecules, topol, config, training)
    # Bonds
    n_bonds = len(bonds)
    bonds_atom1 = np.zeros(n_bonds, dtype=int)
    bonds_atom2 = np.zeros(n_bonds, dtype=int)
    if not training.bonds:
        bonds_equilibrium = np.zeros(n_bonds, dtype=np.float64)
        bonds_strength = np.zeros(n_bonds, dtype=np.float64)
    for i, b in enumerate(bonds):
        bonds_atom1[i] = b[0]
        bonds_atom2[i] = b[1]
        if not training.bonds:
            bonds_equilibrium[i] = b[2]
            bonds_strength[i] = b[3]
    # Angles
    n_angles = len(angles)
    angles_atom1 = np.zeros(n_angles, dtype=int)
    angles_atom2 = np.zeros(n_angles, dtype=int)
    angles_atom3 = np.zeros(n_angles, dtype=int)
    # angles_type is always built (it is structural metadata, not a
    # learnable parameter), so it rides through both training and
    # non-training topology paths.
    angles_type = np.zeros(n_angles, dtype=np.int32)
    if not training.angles:
        angles_equilibrium = np.zeros(n_angles, dtype=np.float64)
        angles_strength = np.zeros(n_angles, dtype=np.float64)
    for i, b in enumerate(angles):
        angles_atom1[i] = b[0]
        angles_atom2[i] = b[1]
        angles_atom3[i] = b[2]
        angles_type[i] = b[3]
        if not training.angles:
            angles_equilibrium[i] = b[4]
            angles_strength[i] = b[5]
    # Python-time static flag used by force.get_angle_energy_and_forces to
    # pick a single-branch harmonic kernel when no row uses G96/ReB.  Cast
    # to a plain Python bool so the value is hashable and the JIT cache
    # key splits cleanly (the 2026-04-23 hashable-metadata rule).
    angle_uses_only_harmonic = bool(np.all(angles_type == 1)) if n_angles > 0 else True
    # Dihedrals
    n_dihedrals = len(dihedrals)
    dihedrals_atom1 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom2 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom3 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom4 = np.zeros(n_dihedrals, dtype=int)
    if not training.dihedrals:
        if config.ff_family == "amber_like":
            dihedrals_phase = np.zeros(n_dihedrals, dtype=np.float64)
            dihedrals_strength = np.zeros(n_dihedrals, dtype=np.float64)
            dihedrals_multiplicity = np.zeros(n_dihedrals, dtype=np.int32)
        else:
            dihedrals_coeffs = np.zeros((n_dihedrals, 6, 5), dtype=np.float64)
            dihedrals_last = np.zeros((n_dihedrals), dtype=int)
    for i, b in enumerate(dihedrals):
        dihedrals_atom1[i] = b[0]
        dihedrals_atom2[i] = b[1]
        dihedrals_atom3[i] = b[2]
        dihedrals_atom4[i] = b[3]
        if not training.dihedrals:
            if config.ff_family == "amber_like":
                dihedrals_phase[i] = b[4]
                dihedrals_strength[i] = b[5]
                dihedrals_multiplicity[i] = b[6]
            else:
                dihedrals_coeffs[i][: len(b[4]), : len(b[4][0])] = b[4]
                dihedrals_last[i] = b[5]
    # TODO: dipole reconstruction triplets, right now it only works if there is a single protein
    # use `dihedrals_last` variable to get all the dipole triplets
    if dihedrals:
        dipole_atom1 = np.append(dihedrals_atom1, dihedrals_atom2[-1])
        dipole_atom2 = np.append(dihedrals_atom2, dihedrals_atom3[-1])
        dipole_atom3 = np.append(dihedrals_atom3, dihedrals_atom4[-1])
    else:
        dipole_atom1, dipole_atom2, dipole_atom3 = [], [], []
    # Amber-like proper torsions in Martini systems (separate bucket so
    # they coexist with native Martini coefficient-format dihedrals).
    n_proper_torsions = len(proper_torsions)
    pt_atom1 = np.zeros(n_proper_torsions, dtype=int)
    pt_atom2 = np.zeros(n_proper_torsions, dtype=int)
    pt_atom3 = np.zeros(n_proper_torsions, dtype=int)
    pt_atom4 = np.zeros(n_proper_torsions, dtype=int)
    if not training.dihedrals:
        pt_phase = np.zeros(n_proper_torsions, dtype=np.float64)
        pt_strength = np.zeros(n_proper_torsions, dtype=np.float64)
        pt_multiplicity = np.zeros(n_proper_torsions, dtype=np.int32)
    for i, b in enumerate(proper_torsions):
        pt_atom1[i] = b[0]
        pt_atom2[i] = b[1]
        pt_atom3[i] = b[2]
        pt_atom4[i] = b[3]
        if not training.dihedrals:
            pt_phase[i] = b[4]
            pt_strength[i] = b[5]
            pt_multiplicity[i] = b[6]
    # Improper dihedrals
    n_impropers = len(impropers)
    improper_atom1 = np.zeros(n_impropers, dtype=int)
    improper_atom2 = np.zeros(n_impropers, dtype=int)
    improper_atom3 = np.zeros(n_impropers, dtype=int)
    improper_atom4 = np.zeros(n_impropers, dtype=int)
    improper_equilibrium = np.zeros(n_impropers, dtype=np.float64)
    improper_strength = np.zeros(n_impropers, dtype=np.float64)
    improper_multiplicity = np.ones(n_impropers, dtype=np.int32)
    improper_periodic = np.zeros(n_impropers, dtype=bool)
    for i, b in enumerate(impropers):
        improper_atom1[i] = b[0]
        improper_atom2[i] = b[1]
        improper_atom3[i] = b[2]
        improper_atom4[i] = b[3]
        improper_equilibrium[i] = b[4]
        improper_strength[i] = b[5]
        if len(b) >= 8:
            improper_multiplicity[i] = int(b[6])
            improper_periodic[i] = bool(b[7])

    bonds_2 = (
        (
            jnp.array(bonds_atom1, dtype=jnp.int32),
            jnp.array(bonds_atom2, dtype=jnp.int32),
        )
        if training.bonds
        else (
            jnp.array(bonds_atom1, dtype=jnp.int32),
            jnp.array(bonds_atom2, dtype=jnp.int32),
            jnp.array(bonds_equilibrium),
            jnp.array(bonds_strength),
        )
    )
    # bonds_3 layout — angles_type rides as the last element so the
    # unpacking `*topol.bonds_3` at every call site fans out cleanly
    # into get_angle_energy_and_forces' positional signature
    # (atom1, atom2, atom3, theta_0, k, angle_type).
    bonds_3 = (
        (
            jnp.array(angles_atom1, dtype=jnp.int32),
            jnp.array(angles_atom2, dtype=jnp.int32),
            jnp.array(angles_atom3, dtype=jnp.int32),
            jnp.array(angles_type, dtype=jnp.int32),
        )
        if training.angles
        else (
            jnp.array(angles_atom1, dtype=jnp.int32),
            jnp.array(angles_atom2, dtype=jnp.int32),
            jnp.array(angles_atom3, dtype=jnp.int32),
            jnp.array(angles_equilibrium),
            jnp.array(angles_strength),
            jnp.array(angles_type, dtype=jnp.int32),
        )
    )
    bonds_4 = (
        (
            jnp.array(dihedrals_atom1, dtype=jnp.int32),
            jnp.array(dihedrals_atom2, dtype=jnp.int32),
            jnp.array(dihedrals_atom3, dtype=jnp.int32),
            jnp.array(dihedrals_atom4, dtype=jnp.int32),
        )
        if training.dihedrals
        else (
            jnp.array(dihedrals_atom1, dtype=jnp.int32),
            jnp.array(dihedrals_atom2, dtype=jnp.int32),
            jnp.array(dihedrals_atom3, dtype=jnp.int32),
            jnp.array(dihedrals_atom4, dtype=jnp.int32),
            jnp.array(dihedrals_phase),
            jnp.array(dihedrals_strength),
            jnp.array(dihedrals_multiplicity, dtype=jnp.int32),
        )
        if config.ff_family == "amber_like"
        else (
            jnp.array(dihedrals_atom1, dtype=jnp.int32),
            jnp.array(dihedrals_atom2, dtype=jnp.int32),
            jnp.array(dihedrals_atom3, dtype=jnp.int32),
            jnp.array(dihedrals_atom4, dtype=jnp.int32),
            jnp.array(dihedrals_coeffs),
            jnp.array(dihedrals_last),
        )
    )
    if n_proper_torsions > 0:
        bonds_pt = (
            (
                jnp.array(pt_atom1, dtype=jnp.int32),
                jnp.array(pt_atom2, dtype=jnp.int32),
                jnp.array(pt_atom3, dtype=jnp.int32),
                jnp.array(pt_atom4, dtype=jnp.int32),
            )
            if training.dihedrals
            else (
                jnp.array(pt_atom1, dtype=jnp.int32),
                jnp.array(pt_atom2, dtype=jnp.int32),
                jnp.array(pt_atom3, dtype=jnp.int32),
                jnp.array(pt_atom4, dtype=jnp.int32),
                jnp.array(pt_phase),
                jnp.array(pt_strength),
                jnp.array(pt_multiplicity, dtype=jnp.int32),
            )
        )
    else:
        bonds_pt = None
    bonds_dip = (
        jnp.array(dipole_atom1, dtype=jnp.int32),
        jnp.array(dipole_atom2, dtype=jnp.int32),
        jnp.array(dipole_atom3, dtype=jnp.int32),
    )
    impropers = (
        jnp.array(improper_atom1, dtype=jnp.int32),
        jnp.array(improper_atom2, dtype=jnp.int32),
        jnp.array(improper_atom3, dtype=jnp.int32),
        jnp.array(improper_atom4, dtype=jnp.int32),
        jnp.array(improper_equilibrium),
        jnp.array(improper_strength),
        jnp.array(improper_multiplicity, dtype=jnp.int32),
        jnp.array(improper_periodic),
    )
    restraints = (
        jnp.array(restraints, dtype=jnp.int32)
    )

    if n_bonds != 0 and config.nrexcl > 0:
        excluded_pairs = find_excluded_pairs(
            bonds_atom1, bonds_atom2, config.nrexcl, config.n_particles
        )
    else:
        excluded_pairs = None
    if explicit_exclusions:
        explicit_arr = np.array(sorted(explicit_exclusions), dtype=np.int32)
        explicit_pairs = (
            jnp.asarray(explicit_arr[:, 0], dtype=jnp.int32),
            jnp.asarray(explicit_arr[:, 1], dtype=jnp.int32),
        )

        if excluded_pairs is None:
            excluded_pairs = explicit_pairs
        else:
            merged = np.concatenate(
                [
                    np.column_stack(
                        [np.asarray(excluded_pairs[0]), np.asarray(excluded_pairs[1])]
                    ),
                    explicit_arr,
                ],
                axis=0,
            )
            merged = np.unique(merged, axis=0)
            excluded_pairs = (
                jnp.asarray(merged[:, 0], dtype=jnp.int32),
                jnp.asarray(merged[:, 1], dtype=jnp.int32),
            )

    if (
        n_bonds != 0
        and config.ff_family == "amber_like"
        and (config.lj14_scale != 1.0 or config.coulomb14_scale != 1.0)
    ):
        one_four_pairs = find_pairs_at_bond_level(
            bonds_atom1, bonds_atom2, 3, config.n_particles
        )
    else:
        one_four_pairs = None

    # CMAP arrays.  ``bonds_cmap`` is built only when there is at
    # least one CMAP entry; otherwise we leave the optional fields as
    # ``None`` so the simulate/mdrun guards (``topol.cmaps``) skip the
    # kernel entirely.
    n_cmaps = len(cmaps)
    if n_cmaps > 0:
        cmap_atom1 = np.zeros(n_cmaps, dtype=np.int32)
        cmap_atom2 = np.zeros(n_cmaps, dtype=np.int32)
        cmap_atom3 = np.zeros(n_cmaps, dtype=np.int32)
        cmap_atom4 = np.zeros(n_cmaps, dtype=np.int32)
        cmap_atom5 = np.zeros(n_cmaps, dtype=np.int32)
        cmap_grid_id = np.zeros(n_cmaps, dtype=np.int32)
        # Lazy import to avoid a circular dependency at module load.
        from .cmap import stack_grid_bank

        coefs_np, name_to_id = stack_grid_bank(config.cmap_grid_bank.grids)
        for i, entry in enumerate(cmaps):
            cmap_atom1[i] = entry[0]
            cmap_atom2[i] = entry[1]
            cmap_atom3[i] = entry[2]
            cmap_atom4[i] = entry[3]
            cmap_atom5[i] = entry[4]
            cmap_grid_id[i] = name_to_id[entry[5]]
        bonds_cmap = (
            jnp.asarray(cmap_atom1, dtype=jnp.int32),
            jnp.asarray(cmap_atom2, dtype=jnp.int32),
            jnp.asarray(cmap_atom3, dtype=jnp.int32),
            jnp.asarray(cmap_atom4, dtype=jnp.int32),
            jnp.asarray(cmap_atom5, dtype=jnp.int32),
            jnp.asarray(cmap_grid_id, dtype=jnp.int32),
        )
        cmap_coefs = jnp.asarray(coefs_np, dtype=jnp.float32)
    else:
        bonds_cmap = None
        cmap_coefs = None

    return Topology(
        # molecules_flag=True,
        molecules=topol["system"]["molecules"],
        bonds_2=bonds_2,
        bonds_3=bonds_3,
        bonds_4=bonds_4,
        bonds_pt=bonds_pt,
        bonds_d=bonds_dip,
        bonds_impr=impropers,
        bonds_cmap=bonds_cmap,
        cmap_coefs=cmap_coefs,
        bonds=n_bonds,
        angles=n_angles,
        dihedrals=n_dihedrals,
        proper_torsions=n_proper_torsions,
        impropers=n_impropers,
        cmaps=n_cmaps,
        angle_uses_only_harmonic=angle_uses_only_harmonic,
        restraints=restraints,
        excluded_pairs=excluded_pairs,
        one_four_pairs=one_four_pairs,
    )


# nn stuff
def get_bond_parameters(model, types: np.ndarray, atom_1, atom_2):
    # types needs to be a numpy array
    strength = []
    equilibrium = []
    for i, j in zip(atom_1, atom_2):
        eq, st = model.bonds[(types[i], types[j])]
        strength.append(st)
        equilibrium.append(eq)
    return (atom_1, atom_2, jnp.array(strength), jnp.array(equilibrium))


def get_dihedral_parameters(model, types: np.ndarray, atom_1, atom_2, atom_3, atom_4):
    # types needs to be a numpy array
    coeffs = []
    last = []
    for i, j, k, l in zip(atom_1, atom_2, atom_3, atom_4):
        last.append(0)
        coeff = model.dihedrals[(types[i], types[j], types[k], types[l])]
        coeffs.append(coeff)
    if len(last) > 0:
        last[-1] = 1  # single protein
    return atom_1, atom_2, atom_3, atom_4, jnp.array(coeffs), jnp.array(last)


def get_bonded_parameters(model, types, topol):
    if hasattr(model, "bonds") and model.bonds:
        bonds_2 = get_bond_parameters(model, types, *topol.bonds_2)
    else:
        bonds_2 = topol.bonds_2
    if hasattr(model, "dihedrals") and model.dihedrals:
        bonds_4 = get_dihedral_parameters(model, types, *topol.bonds_4)
    else:
        bonds_4 = topol.bonds_4

    return Topology(
        # molecules_flag=True,
        molecules=topol.molecules,
        bonds_2=bonds_2,
        bonds_3=topol.bonds_3,
        bonds_4=bonds_4,
        bonds_pt=topol.bonds_pt,
        bonds_d=topol.bonds_d,
        bonds_impr=topol.bonds_impr,
        bonds_cmap=topol.bonds_cmap,
        cmap_coefs=topol.cmap_coefs,
        bonds=topol.bonds,
        angles=topol.angles,
        dihedrals=topol.dihedrals,
        proper_torsions=topol.proper_torsions,
        impropers=topol.impropers,
        cmaps=topol.cmaps,
        angle_uses_only_harmonic=topol.angle_uses_only_harmonic,
        restraints=topol.restraints,
        excluded_pairs=topol.excluded_pairs,
        one_four_pairs=topol.one_four_pairs,
    )
