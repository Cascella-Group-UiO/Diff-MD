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
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_array


@struct.dataclass
class Topology:
    # molecules_flag: bool = struct.field(pytree_node=False, default=False)
    molecules: dict[str, int] = struct.field(pytree_node=False, default=False)
    bonds: int = struct.field(pytree_node=False, default=False)
    angles: int = struct.field(pytree_node=False, default=False)
    dihedrals: int = struct.field(pytree_node=False, default=False)
    impropers: int = struct.field(pytree_node=False, default=False)
    restraints: Array = struct.field(pytree_node=False, default=False)
    bonds_2: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))
    bonds_3: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))
    bonds_4: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))
    bonds_d: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))
    bonds_impr: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))
    excluded_pairs: tuple[Array, ...] = struct.field(pytree_node=False, default=([], []))


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
    adjacency_matrix = np.zeros((num_particles, num_particles), dtype=int)
    for i, j in zip(i_atoms, j_atoms):
      adjacency_matrix[i][j] = 1
      adjacency_matrix[j][i] = 1

    dist_matrix, _ = shortest_path(csr_array(adjacency_matrix), directed=False, return_predecessors=True)
    pairs_i = []
    pairs_j = []
    for bond_level in range(1, nrexcl + 1):
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                if dist_matrix[i][j] == bond_level:
                    pairs_i.append(i)
                    pairs_j.append(j)

    return (jnp.asarray(pairs_i), jnp.asarray(pairs_j))


def prepare_index_based_bonds(molecules, topol, training):
    bonds = []
    angles = []
    dihedrals = []
    impropers = []
    restraints = []

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

        if "angles" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for angle in topol[resname]["angles"]:
                index_i = angle[0] - 1 + first_id
                index_j = angle[1] - 1 + first_id
                index_k = angle[2] - 1 + first_id
                # angle[3] is the angle type, inherited by the itp format
                # we don't use it
                if training.angles:
                    angles.append([index_i, index_j, index_k])
                    continue
                equilibrium = np.radians(angle[4])
                strength = angle[5]
                angles.append([index_i, index_j, index_k, equilibrium, strength])

        if "dihedrals" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for angle in topol[resname]["dihedrals"]:
                index_i = angle[0] - 1 + first_id
                index_j = angle[1] - 1 + first_id
                index_k = angle[2] - 1 + first_id
                index_l = angle[3] - 1 + first_id
                d_type = angle[4]
                # Parse impropers
                if d_type == 2:
                    equilibrium = np.radians(angle[4])
                    strength = angle[5]
                    impropers.append(
                        [index_i, index_j, index_k, index_l, equilibrium, strength]
                    )
                else:
                    if training.dihedrals:
                        # coeffs are defined inside the simulator ==>
                        # we just need the indices and then filter out by type
                        dihedrals.append([index_i, index_j, index_k, index_l])
                        continue
                    coeff = angle[5]
                    dihedrals.append([index_i, index_j, index_k, index_l, coeff, 0])
            if not training.dihedrals:
                # TODO: provide 'is_last' in the topology?
                dihedrals[-1][-1] = 1
            
    return bonds, angles, dihedrals, impropers, restraints


def prepare_bonds(molecules, topol, config, training=None):
    if training is None:
        training = GeneralModel()
    bonds, angles, dihedrals, impropers, restraints = prepare_index_based_bonds(
        molecules, topol, training
    )
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
    if not training.angles:
        angles_equilibrium = np.zeros(n_angles, dtype=np.float64)
        angles_strength = np.zeros(n_angles, dtype=np.float64)
    for i, b in enumerate(angles):
        angles_atom1[i] = b[0]
        angles_atom2[i] = b[1]
        angles_atom3[i] = b[2]
        if not training.angles:
            angles_equilibrium[i] = b[3]
            angles_strength[i] = b[4]
    # Dihedrals
    n_dihedrals = len(dihedrals)
    dihedrals_atom1 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom2 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom3 = np.zeros(n_dihedrals, dtype=int)
    dihedrals_atom4 = np.zeros(n_dihedrals, dtype=int)
    if not training.dihedrals:
        dihedrals_coeffs = np.zeros((n_dihedrals, 6, 5), dtype=np.float64)
        dihedrals_last = np.zeros((n_dihedrals), dtype=int)
    for i, b in enumerate(dihedrals):
        dihedrals_atom1[i] = b[0]
        dihedrals_atom2[i] = b[1]
        dihedrals_atom3[i] = b[2]
        dihedrals_atom4[i] = b[3]
        if not training.dihedrals:
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
    # Improper dihedrals
    n_impropers = len(impropers)
    improper_atom1 = np.zeros(n_impropers, dtype=int)
    improper_atom2 = np.zeros(n_impropers, dtype=int)
    improper_atom3 = np.zeros(n_impropers, dtype=int)
    improper_atom4 = np.zeros(n_impropers, dtype=int)
    improper_equilibrium = np.zeros(n_impropers, dtype=np.float64)
    improper_strength = np.zeros(n_impropers, dtype=np.float64)
    for i, b in enumerate(impropers):
        improper_atom1[i] = b[0]
        improper_atom2[i] = b[1]
        improper_atom3[i] = b[2]
        improper_atom4[i] = b[3]
        improper_equilibrium[i] = b[4]
        improper_strength[i] = b[5]

    bonds_2 = (
        (
            jnp.array(bonds_atom1),
            jnp.array(bonds_atom2),
        )
        if training.bonds
        else (
            jnp.array(bonds_atom1),
            jnp.array(bonds_atom2),
            jnp.array(bonds_equilibrium),
            jnp.array(bonds_strength),
        )
    )
    bonds_3 = (
        (
            jnp.array(angles_atom1),
            jnp.array(angles_atom2),
            jnp.array(angles_atom3),
        )
        if training.angles
        else (
            jnp.array(angles_atom1),
            jnp.array(angles_atom2),
            jnp.array(angles_atom3),
            jnp.array(angles_equilibrium),
            jnp.array(angles_strength),
        )
    )
    bonds_4 = (
        (
            jnp.array(dihedrals_atom1),
            jnp.array(dihedrals_atom2),
            jnp.array(dihedrals_atom3),
            jnp.array(dihedrals_atom4),
        )
        if training.dihedrals
        else (
            jnp.array(dihedrals_atom1),
            jnp.array(dihedrals_atom2),
            jnp.array(dihedrals_atom3),
            jnp.array(dihedrals_atom4),
            jnp.array(dihedrals_coeffs),
            jnp.array(dihedrals_last),
        )
    )
    bonds_dip = (
        jnp.array(dipole_atom1),
        jnp.array(dipole_atom2),
        jnp.array(dipole_atom3),
    )
    impropers = (
        jnp.array(improper_atom1),
        jnp.array(improper_atom2),
        jnp.array(improper_atom3),
        jnp.array(improper_atom4),
        jnp.array(improper_equilibrium),
        jnp.array(improper_strength),
    )
    restraints = (
        jnp.array(restraints)
    )

    if n_bonds != 0:
        excluded_pairs = find_excluded_pairs(bonds_atom1, bonds_atom2, config.nrexcl, config.n_particles)
    else:
        excluded_pairs = None

    return Topology(
        # molecules_flag=True,
        molecules=topol["system"]["molecules"],
        bonds_2=bonds_2,
        bonds_3=bonds_3,
        bonds_4=bonds_4,
        bonds_d=bonds_dip,
        bonds_impr=impropers,
        bonds=n_bonds,
        angles=n_angles,
        dihedrals=n_dihedrals,
        impropers=n_impropers,
        restraints=restraints,
        excluded_pairs = excluded_pairs
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
    last[-1] = 1  # single protein
    return atom_1, atom_2, atom_3, atom_4, jnp.array(coeffs), jnp.array(last)


def get_bonded_parameters(model, types, topol):
    if hasattr(model, "bonds") and model.bonds:
        bonds_2 = get_bond_parameters(model, types, *topol.bonds_2)
    else:
        bonds_2 = topol.bonds_2
    # if hasattr(model, "angles") and model.angles:
    #     bonds_3 = get_angle_parameters(model, types, *topol.bonds_2)
    # else:
    #     bonds_3 = topol.bonds_3
    if hasattr(model, "dihedrals") and model.dihedrals:
        bonds_4 = get_dihedral_parameters(model, types, *topol.bonds_4)
    else:
        bonds_4 = topol.bonds_4

    return Topology(
        # molecules_flag=True,
        bonds_2=bonds_2,
        bonds_3=topol.bonds_3,
        bonds_4=bonds_4,
        bonds_d=topol.bonds_d,
        # impropers,
        bonds=topol.bonds,
        angles=topol.angles,
        dihedrals=topol.dih,
    )
