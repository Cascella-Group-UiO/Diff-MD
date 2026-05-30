from dataclasses import dataclass

import numpy as np

from .topol_utils import Topol


@dataclass
class System:
    n_mol: int
    names: np.ndarray
    resnames: np.ndarray
    types: np.ndarray
    molecules: np.ndarray
    masses: np.ndarray
    charges: np.ndarray


def get_system_properties(
    molecule_list: dict[str, int], topol: dict[str, Topol]
) -> System:
    names, resnames, molecules, masses, charges = [], [], [], [], []
    exclude = ()
    n_mol = 0
    for molname, molnum in molecule_list.items():
        names += molnum * [
            atom.atomtype if atom.resname not in exclude else atom.atomname
            for atom in topol[molname].atoms
        ]

        resnames += molnum * [atom.resname for atom in topol[molname].atoms]

        for i in range(molnum):
            molecules += [(n_mol + i)] * topol[molname].n_atoms
        atom_masses = [atom.mass for atom in topol[molname].atoms]
        masses += atom_masses * molnum
        n_mol += molnum

        charges += molnum * [atom.charge for atom in topol[molname].atoms]

    names = np.array(names, dtype="S10")
    resnames = np.array(resnames, dtype="S10")
    molecules = np.array(molecules)
    masses = np.array(masses)
    charges = np.array(charges)

    _, idx = np.unique(names, return_index=True)
    unique_names = names[np.sort(idx)]
    name_to_type = {name: t for t, name in enumerate(unique_names)}
    types = np.array([name_to_type[name] for name in names])

    return System(n_mol, names, resnames, types, molecules, masses, charges)


# Known single-bead / single-atom solvent and ion residue names. Used by
# derive_thermo_groups to bucket bead types into thermostat groups.
SOLVENT_RESNAMES: frozenset[str] = frozenset({
    "W", "SW", "TW", "SOL", "HOH", "WAT", "TIP3", "SPC",
})
ION_RESNAMES: frozenset[str] = frozenset({
    "NA", "NA+", "CL", "CL-", "K", "K+", "CA", "CA+", "MG", "MG2+",
    "LI", "LI+", "RB", "RB+", "CS", "CS+", "ZN", "ZN2+", "ION",
})


def derive_thermo_groups(system: System) -> list[list[str]]:
    """Bucket unique atomtypes into thermostat-coupling groups.

    Heuristic per molecule species:

    * **solvent** — residue name in :data:`SOLVENT_RESNAMES`, or single-atom
      molecule with zero charge.
    * **ion**     — residue name in :data:`ION_RESNAMES`, or single-atom
      molecule with non-zero charge.
    * **polymer** — everything else (multi-atom molecules).

    Returns one list of atomtype names per non-empty bucket, in order
    ``[polymer, solvent, ion]``. Names are str (decoded from bytes).
    """
    names = np.asarray(system.names)
    resnames = np.asarray(system.resnames)
    molecules = np.asarray(system.molecules)
    charges = np.asarray(system.charges)

    def _s(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)

    # Per-molecule atom count
    _, mol_sizes = np.unique(molecules, return_counts=True)
    mol_size_lookup = dict(zip(*np.unique(molecules, return_counts=True)))

    polymer: list[str] = []
    solvent: list[str] = []
    ion: list[str] = []
    seen: set[str] = set()

    for atom_name, atom_resname, mol_id, charge in zip(names, resnames, molecules, charges):
        n = _s(atom_name)
        if n in seen:
            continue
        seen.add(n)
        r = _s(atom_resname).strip()
        size = int(mol_size_lookup[int(mol_id)])
        if r in SOLVENT_RESNAMES or (size == 1 and abs(float(charge)) < 1e-9):
            solvent.append(n)
        elif r in ION_RESNAMES or (size == 1 and abs(float(charge)) >= 1e-9):
            ion.append(n)
        else:
            polymer.append(n)

    return [group for group in (polymer, solvent, ion) if group]
