import argparse
import collections
import os
import re
import shutil
import sys
import textwrap
from argparse import RawDescriptionHelpFormatter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .gro_utils import GroAtom, load_gro


NAME_TO_ELEMENT = {
    "OW": "O",
    "HW": "H",
    "MW": "X",
    "CA": "C",
    "CT": "C",
    "CX": "C",
    "3C": "C",
    "2C": "C",
    "C8": "C",
    "C*": "C",
    "CB": "C",
    "CN": "C",
    "CO": "C",
    "CW": "C",
    "NZ": "N",
    "N3": "N",
    "NA": "N",
    "O2": "O",
    "OH": "O",
    "H1": "H",
    "H": "H",
    "H4": "H",
    "HA": "H",
    "HC": "H",
    "HP": "H",
    "HO": "H",
    "S": "S",
    "SZ": "S",
}
COMMON_TWO_LETTER = {
    "CL": "Cl",
    "NA": "Na",
    "MG": "Mg",
    "ZN": "Zn",
    "FE": "Fe",
    "LI": "Li",
    "CS": "Cs",
    "RB": "Rb",
}
PERIODIC_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}


def _normalize_letters(token: str) -> str:
    letters = "".join(re.findall(r"[A-Za-z]+", token))
    if not letters:
        return ""
    if len(letters) == 1:
        return letters[0].upper()
    return letters[0].upper() + letters[1:].lower()


def _guess_element_from_token(token: str) -> str | None:
    normalized = _normalize_letters(token)
    if not normalized:
        return None
    if normalized in PERIODIC_SYMBOLS:
        return normalized
    if len(normalized) >= 2:
        two_letter = normalized[:2]
        if two_letter in PERIODIC_SYMBOLS:
            return two_letter
    one_letter = normalized[0]
    if one_letter in PERIODIC_SYMBOLS:
        return one_letter
    return None


def _infer_element(name: str) -> str:
    cleaned = name.strip().replace("\x00", "")
    if cleaned in NAME_TO_ELEMENT:
        return NAME_TO_ELEMENT[cleaned]

    base_token = cleaned.split("_", 1)[0]
    if base_token in NAME_TO_ELEMENT:
        return NAME_TO_ELEMENT[base_token]

    guessed = _guess_element_from_token(base_token)
    if guessed:
        return guessed

    letters = "".join(re.findall(r"[A-Za-z]+", cleaned)).upper()
    if len(letters) >= 2 and letters[:2] in COMMON_TWO_LETTER:
        return COMMON_TWO_LETTER[letters[:2]]

    guessed = _guess_element_from_token(cleaned)
    if guessed:
        return guessed
    return "H"


def _build_unwrap_data(molecules, bonds_atom1, bonds_atom2):
    """Mirror Diff-MD whole-molecule unwrapping precomputation."""
    if molecules is None:
        return None, None, None, None

    n_atoms = len(molecules)

    mol_to_atoms = {}
    for atom_idx, mol in enumerate(np.asarray(molecules).ravel()):
        mol_to_atoms.setdefault(int(mol), []).append(atom_idx)

    adjacency = [[] for _ in range(n_atoms)]
    for atom1, atom2 in zip(np.asarray(bonds_atom1).ravel(), np.asarray(bonds_atom2).ravel()):
        atom1 = int(atom1)
        atom2 = int(atom2)
        adjacency[atom1].append(atom2)
        adjacency[atom2].append(atom1)

    small_anchor_list = []
    small_other_list = []
    large_mol_atoms = []

    for atom_ids in mol_to_atoms.values():
        n_atoms_mol = len(atom_ids)
        if n_atoms_mol <= 1:
            continue
        if n_atoms_mol <= 3:
            anchor = atom_ids[0]
            others = atom_ids[1:] + [-1] * (2 - (n_atoms_mol - 1))
            small_anchor_list.append(anchor)
            small_other_list.append(others)
        else:
            large_mol_atoms.append(atom_ids)

    small_anchors = np.array(small_anchor_list, dtype=np.intp) if small_anchor_list else None
    small_others = np.array(small_other_list, dtype=np.intp) if small_other_list else None
    return small_anchors, small_others, large_mol_atoms, adjacency


def _read_toml(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


@dataclass
class ExpectedTopologyMetadata:
    names: np.ndarray
    atom_names: np.ndarray
    resnames: np.ndarray
    molecules: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    types: np.ndarray
    bonds_i: np.ndarray
    bonds_j: np.ndarray


@dataclass
class TemplateMetadata:
    indices: np.ndarray
    names: np.ndarray
    resnames: np.ndarray
    molecules: np.ndarray
    masses: np.ndarray
    charges: np.ndarray | None
    types: np.ndarray
    bond_rows_i: np.ndarray | None
    bond_rows_j: np.ndarray | None


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            decoded.append(value.decode("utf-8").replace("\x00", "").strip())
        else:
            decoded.append(str(value).replace("\x00", "").strip())
    return np.asarray(decoded, dtype=object)


def _load_toml_topology_bundle(topol_path: str | Path):
    topol_path = Path(topol_path)
    toml_topol = _read_toml(str(topol_path))
    system_block = toml_topol.get("system", {})
    if "molecules" not in system_block:
        raise ValueError(f"Topology TOML '{topol_path}' is missing [system].molecules.")

    molecule_tables = {
        key: value for key, value in toml_topol.items() if key != "system"
    }
    for inc_file in system_block.get("include", []):
        inc_path = topol_path.parent / inc_file
        included = _read_toml(str(inc_path))
        molecule_tables.update(included)
    return system_block["molecules"], molecule_tables


def _build_expected_topology_metadata(topol_path: str | Path) -> ExpectedTopologyMetadata:
    molecule_summary, molecule_tables = _load_toml_topology_bundle(topol_path)

    names = []
    atom_names = []
    resnames = []
    molecules = []
    masses = []
    charges = []
    molecule_index = 0

    for molname, mol_count in molecule_summary:
        if molname not in molecule_tables:
            raise ValueError(
                f"Topology TOML '{topol_path}' declares molecule '{molname}' but no TOML table was found for it."
            )
        atoms = molecule_tables[molname].get("atoms")
        if atoms is None:
            raise ValueError(f"Molecule '{molname}' is missing an 'atoms' table in '{topol_path}'.")

        for _ in range(int(mol_count)):
            for atom in atoms:
                if len(atom) < 8:
                    raise ValueError(
                        f"Molecule '{molname}' has an atom entry with fewer than 8 fields: {atom}"
                    )
                names.append(str(atom[1]))
                resnames.append(str(atom[3]))
                atom_names.append(str(atom[4]))
                molecules.append(molecule_index)
                charges.append(float(atom[6]))
                masses.append(float(atom[7]))
            molecule_index += 1

    names_arr = np.asarray(names, dtype=object)
    _, idx = np.unique(names_arr, return_index=True)
    unique_names = names_arr[np.sort(idx)]
    name_to_type = {name: type_index for type_index, name in enumerate(unique_names)}
    types = np.asarray([name_to_type[name] for name in names_arr], dtype=np.int32)
    molecules_arr = np.asarray(molecules, dtype=np.int32)
    bonds_i, bonds_j = _load_bonds_from_toml(str(topol_path), molecules_arr)
    return ExpectedTopologyMetadata(
        names=names_arr,
        atom_names=np.asarray(atom_names, dtype=object),
        resnames=np.asarray(resnames, dtype=object),
        molecules=molecules_arr,
        masses=np.asarray(masses, dtype=np.float64),
        charges=np.asarray(charges, dtype=np.float64),
        types=types,
        bonds_i=bonds_i,
        bonds_j=bonds_j,
    )


def _normalize_template_bond_rows(infile: h5py.File, n_atoms: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if "parameters/vmd_structure/bond_from" not in infile or "parameters/vmd_structure/bond_to" not in infile:
        return None, None

    bond_from = np.asarray(infile["parameters/vmd_structure/bond_from"], dtype=np.int64).ravel()
    bond_to = np.asarray(infile["parameters/vmd_structure/bond_to"], dtype=np.int64).ravel()
    if bond_from.size == 0 or bond_to.size == 0:
        return None, None

    if "indices" in infile:
        atom_indices = np.asarray(infile["indices"], dtype=np.int64).ravel()
    elif "particles/all/indices" in infile:
        atom_indices = np.asarray(infile["particles/all/indices"], dtype=np.int64).ravel()
    else:
        atom_indices = np.arange(n_atoms, dtype=np.int64)

    if np.array_equal(atom_indices, np.arange(n_atoms, dtype=np.int64)):
        if np.min(bond_from) >= 1 and np.min(bond_to) >= 1 and np.max(bond_from) <= n_atoms and np.max(bond_to) <= n_atoms:
            return bond_from.astype(np.int64) - 1, bond_to.astype(np.int64) - 1

    row_for_index = {int(atom_index): row for row, atom_index in enumerate(atom_indices)}

    def _try_map(values: np.ndarray):
        try:
            return np.asarray([row_for_index[int(value)] for value in values], dtype=np.int64)
        except KeyError:
            return None

    direct_from = _try_map(bond_from)
    direct_to = _try_map(bond_to)
    if direct_from is not None and direct_to is not None:
        return direct_from, direct_to

    if np.min(bond_from) >= 1 and np.min(bond_to) >= 1:
        shifted_from = _try_map(bond_from - 1)
        shifted_to = _try_map(bond_to - 1)
        if shifted_from is not None and shifted_to is not None:
            return shifted_from, shifted_to

    if np.max(bond_from) < n_atoms and np.max(bond_to) < n_atoms:
        return bond_from.astype(np.int64), bond_to.astype(np.int64)

    raise ValueError("Template H5 contains embedded bonds that could not be mapped onto atom rows.")


def _load_template_metadata(template_h5: str | Path) -> TemplateMetadata:
    template_h5 = Path(template_h5)
    with h5py.File(template_h5, "r") as infile:
        required = ["indices", "names", "types", "molecules", "masses", "resnames"]
        missing = [name for name in required if name not in infile]
        if missing:
            raise ValueError(
                f"Template H5 '{template_h5}' is missing required restart metadata datasets: {missing}"
            )

        names = _decode_string_array(np.asarray(infile["names"]))
        resnames = _decode_string_array(np.asarray(infile["resnames"]))
        indices = np.asarray(infile["indices"], dtype=np.int64).ravel()
        types = np.asarray(infile["types"], dtype=np.int32).ravel()
        molecules = np.asarray(infile["molecules"], dtype=np.int32).ravel()
        masses = np.asarray(infile["masses"], dtype=np.float64).ravel()
        charges = (
            np.asarray(infile["charge"], dtype=np.float64).ravel()
            if "charge" in infile
            else None
        )

        n_atoms = len(indices)
        if not (len(names) == len(resnames) == len(types) == len(molecules) == len(masses) == n_atoms):
            raise ValueError(f"Template H5 '{template_h5}' has inconsistent per-atom metadata lengths.")

        bond_rows_i, bond_rows_j = _normalize_template_bond_rows(infile, n_atoms)
        return TemplateMetadata(
            indices=indices,
            names=names,
            resnames=resnames,
            molecules=molecules,
            masses=masses,
            charges=charges,
            types=types,
            bond_rows_i=bond_rows_i,
            bond_rows_j=bond_rows_j,
        )


def _sorted_bond_pairs(bonds_i: np.ndarray, bonds_j: np.ndarray) -> np.ndarray:
    if bonds_i.size == 0 and bonds_j.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.column_stack([bonds_i, bonds_j]).astype(np.int64)
    pairs.sort(axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _fail_mismatch(label: str, index: int, expected, observed) -> None:
    raise ValueError(
        f"{label} mismatch at atom {index}: expected '{expected}', observed '{observed}'."
    )


def _validate_template_matches_topology(
    template_h5: str | Path,
    topol_path: str | Path,
) -> ExpectedTopologyMetadata:
    numeric_tol = 1e-5
    expected = _build_expected_topology_metadata(topol_path)
    template = _load_template_metadata(template_h5)

    n_atoms = len(expected.names)
    if len(template.indices) != n_atoms:
        raise ValueError(
            f"Template H5 '{template_h5}' atom count ({len(template.indices)}) does not match topology TOML '{topol_path}' ({n_atoms})."
        )

    expected_indices = np.arange(n_atoms, dtype=np.int64)
    if not np.array_equal(template.indices, expected_indices):
        raise ValueError("Template H5 indices are not the expected contiguous 0-based atom ordering.")

    for atom_index, (expected_name, observed_name) in enumerate(zip(expected.names, template.names)):
        if expected_name != observed_name:
            _fail_mismatch("Template atom type", atom_index, expected_name, observed_name)
    for atom_index, (expected_resname, observed_resname) in enumerate(zip(expected.resnames, template.resnames)):
        if expected_resname != observed_resname:
            _fail_mismatch("Template residue name", atom_index, expected_resname, observed_resname)

    if not np.array_equal(expected.molecules, template.molecules):
        mismatch_idx = int(np.flatnonzero(expected.molecules != template.molecules)[0])
        _fail_mismatch(
            "Template molecule index",
            mismatch_idx,
            expected.molecules[mismatch_idx],
            template.molecules[mismatch_idx],
        )

    if not np.allclose(expected.masses, template.masses, atol=numeric_tol, rtol=0.0):
        mismatch_idx = int(np.flatnonzero(~np.isclose(expected.masses, template.masses, atol=numeric_tol, rtol=0.0))[0])
        _fail_mismatch(
            "Template mass",
            mismatch_idx,
            expected.masses[mismatch_idx],
            template.masses[mismatch_idx],
        )

    if template.charges is not None and not np.allclose(expected.charges, template.charges, atol=numeric_tol, rtol=0.0):
        mismatch_idx = int(np.flatnonzero(~np.isclose(expected.charges, template.charges, atol=numeric_tol, rtol=0.0))[0])
        _fail_mismatch(
            "Template charge",
            mismatch_idx,
            expected.charges[mismatch_idx],
            template.charges[mismatch_idx],
        )

    if not np.array_equal(expected.types, template.types):
        mismatch_idx = int(np.flatnonzero(expected.types != template.types)[0])
        _fail_mismatch(
            "Template type index",
            mismatch_idx,
            expected.types[mismatch_idx],
            template.types[mismatch_idx],
        )

    if template.bond_rows_i is not None and template.bond_rows_j is not None:
        expected_pairs = _sorted_bond_pairs(expected.bonds_i, expected.bonds_j)
        template_pairs = _sorted_bond_pairs(template.bond_rows_i, template.bond_rows_j)
        if expected_pairs.shape != template_pairs.shape or not np.array_equal(expected_pairs, template_pairs):
            raise ValueError(
                f"Template H5 '{template_h5}' embedded bonds do not match topology TOML '{topol_path}'."
            )

    return expected


def _validate_gro_order(gro_path: str | Path, expected: ExpectedTopologyMetadata) -> tuple[np.ndarray, np.ndarray]:
    atoms, box = load_gro(str(gro_path))
    if len(atoms) != len(expected.names):
        raise ValueError(
            f"GRO file '{gro_path}' atom count ({len(atoms)}) does not match the validated topology/template atom count ({len(expected.names)})."
        )

    coordinates = np.empty((len(atoms), 3), dtype=np.float32)
    velocities = np.empty((len(atoms), 3), dtype=np.float32)
    for atom_index, atom in enumerate(atoms):
        if atom.resname != expected.resnames[atom_index]:
            _fail_mismatch("GRO residue name", atom_index, expected.resnames[atom_index], atom.resname)
        if atom.atom_name != expected.atom_names[atom_index]:
            _fail_mismatch("GRO atom name", atom_index, expected.atom_names[atom_index], atom.atom_name)
        coordinates[atom_index] = np.asarray([atom.x, atom.y, atom.z], dtype=np.float32)
        velocities[atom_index] = np.asarray([atom.vx, atom.vy, atom.vz], dtype=np.float32)

    if box.shape[0] < 3:
        raise ValueError(f"GRO file '{gro_path}' does not contain three box lengths.")
    box_lengths = np.asarray(box[:3], dtype=np.float32)
    if not np.all(np.isfinite(box_lengths)) or not np.all(box_lengths > 0.0):
        raise ValueError(f"GRO file '{gro_path}' has an invalid box: {box_lengths}.")
    return coordinates, velocities, box_lengths


def _update_template_clone(
    output_h5: str | Path,
    coordinates: np.ndarray,
    box_lengths: np.ndarray,
    velocities: np.ndarray,
) -> None:
    with h5py.File(output_h5, "r+") as outfile:
        outfile.attrs["box"] = np.asarray(box_lengths, dtype=outfile.attrs["box"].dtype if "box" in outfile.attrs else np.float32)

        if "coordinates" in outfile:
            outfile["coordinates"][0, :, :] = np.asarray(coordinates, dtype=outfile["coordinates"].dtype)
        if "velocities" in outfile:
            outfile["velocities"][0, :, :] = np.asarray(velocities, dtype=outfile["velocities"].dtype)

        if "particles/all/position/value" in outfile:
            outfile["particles/all/position/value"][0, :, :] = np.asarray(
                coordinates, dtype=outfile["particles/all/position/value"].dtype
            )
        if "particles/all/velocity/value" in outfile:
            outfile["particles/all/velocity/value"][0, :, :] = np.asarray(
                velocities, dtype=outfile["particles/all/velocity/value"].dtype
            )
        if "particles/all/box/edges/value" in outfile:
            box_dataset = outfile["particles/all/box/edges/value"]
            if box_dataset.ndim == 3 and box_dataset.shape[1:] == (3, 3):
                box_matrix = np.zeros((3, 3), dtype=box_dataset.dtype)
                box_matrix[np.arange(3), np.arange(3)] = np.asarray(box_lengths, dtype=box_dataset.dtype)
                box_dataset[0, :, :] = box_matrix
            elif box_dataset.ndim == 2 and box_dataset.shape[1] == 3:
                box_dataset[0, :] = np.asarray(box_lengths, dtype=box_dataset.dtype)
            elif box_dataset.ndim == 1 and box_dataset.shape[0] == 3:
                box_dataset[:] = np.asarray(box_lengths, dtype=box_dataset.dtype)
            else:
                raise ValueError(
                    f"Unsupported H5MD box dataset shape {box_dataset.shape} in '{output_h5}'."
                )


def gro_to_h5_from_template(
    gro_dir: Path,
    template_h5: Path,
    topol_path: Path,
    output_dir: Path,
    gro_pattern: str = "*.gro",
    overwrite: bool = False,
    use_gro_velocities: bool = False,
) -> list[Path]:
    expected = _validate_template_matches_topology(template_h5, topol_path)
    gro_paths = sorted(path for path in gro_dir.glob(gro_pattern) if path.is_file())
    if not gro_paths:
        raise ValueError(f"No GRO files matching '{gro_pattern}' were found in '{gro_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for gro_path in gro_paths:
        coordinates, gro_velocities, box_lengths = _validate_gro_order(gro_path, expected)
        output_path = output_dir / f"{gro_path.stem}.h5"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file '{output_path}'. Pass --overwrite to replace it."
            )
        shutil.copy2(template_h5, output_path)
        if use_gro_velocities:
            velocities = gro_velocities
        else:
            velocities = np.zeros_like(coordinates, dtype=np.float32)
        _update_template_clone(output_path, coordinates, box_lengths, velocities)
        written.append(output_path)
    return written


def _load_bonds_from_toml(topol_path: str, molecules: np.ndarray):
    """Load bond connectivity from a TOML topology, mirroring diff_md/topology.py."""
    topol_dir = os.path.dirname(topol_path) or "."
    toml_topol = _read_toml(topol_path)

    if "include" in toml_topol.get("system", {}):
        for inc_file in toml_topol["system"]["include"]:
            inc_path = os.path.join(topol_dir, inc_file)
            itps = _read_toml(inc_path)
            for mol_name, mol_data in itps.items():
                toml_topol[mol_name] = mol_data

    top_summary = toml_topol["system"]["molecules"]

    bonds_i = []
    bonds_j = []
    different_molecules = np.unique(molecules)
    for mol in different_molecules:
        resid = mol + 1
        test_mol_number = 0
        resname = None
        for mol_entry in top_summary:
            test_mol_number += mol_entry[1]
            if resid <= test_mol_number:
                resname = mol_entry[0]
                break
        if resname is None:
            continue
        if resname not in toml_topol or "bonds" not in toml_topol[resname]:
            continue
        first_id = int(np.where(molecules == mol)[0][0])
        for bond in toml_topol[resname]["bonds"]:
            bonds_i.append(bond[0] - 1 + first_id)
            bonds_j.append(bond[1] - 1 + first_id)

    return (
        np.array(bonds_i, dtype=np.int64),
        np.array(bonds_j, dtype=np.int64),
    )


def _unwrap_molecules_same_as_diff_md(
    positions,
    box_size,
    small_anchors,
    small_others,
    large_mol_atoms,
    adjacency,
):
    """Mirror Diff-MD whole-molecule imaging for XYZ export."""
    pos = np.array(positions, dtype=np.float64)
    box = np.asarray(box_size, dtype=np.float64)
    inv_box = 1.0 / box

    if small_anchors is not None and small_others is not None:
        anchor_pos = pos[small_anchors]
        for col in range(small_others.shape[1]):
            idx = small_others[:, col]
            mask = idx >= 0
            if not np.any(mask):
                continue
            valid_idx = idx[mask]
            dr = pos[valid_idx] - anchor_pos[mask]
            dr -= box * np.round(dr * inv_box)
            pos[valid_idx] = anchor_pos[mask] + dr

    if large_mol_atoms:
        for atom_ids in large_mol_atoms:
            atom_set = set(atom_ids)
            visited = {atom_ids[0]}
            queue = collections.deque([atom_ids[0]])

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


def _resolve_atom_elements(infile: h5py.File) -> list[str]:
    if "parameters/vmd_structure/name" in infile:
        raw_unique_names = infile["parameters/vmd_structure/name"][:]
        unique_names = [
            name.decode("utf-8").strip().replace("\x00", "")
            for name in raw_unique_names
        ]
        species_indices = infile["particles/all/species"][:]
        return [_infer_element(unique_names[idx]) for idx in species_indices]

    if "names" in infile:
        raw_names = infile["names"][:]
        return [
            _infer_element(
                name.decode("utf-8").strip().replace("\x00", "")
                if isinstance(name, bytes) else str(name).strip()
            )
            for name in raw_names
        ]

    if "particles/all/names" in infile:
        raw_names = infile["particles/all/names"][:]
        return [
            _infer_element(
                name.decode("utf-8").strip().replace("\x00", "")
                if isinstance(name, bytes) else str(name).strip()
            )
            for name in raw_names
        ]

    if "particles/all/species" in infile:
        species_indices = infile["particles/all/species"][:]
        return [f"X{idx}" for idx in species_indices]

    raise KeyError(
        "Cannot determine atom names/elements. Expected "
        "'parameters/vmd_structure/name', 'names', or 'particles/all/names'."
    )


def _resolve_coords_dataset(infile: h5py.File):
    if "coordinates" in infile:
        return infile["coordinates"]
    if "particles/all/position/value" in infile:
        return infile["particles/all/position/value"]
    raise KeyError(
        "No coordinate data found. Expected 'coordinates' or "
        "'particles/all/position/value'."
    )


def _resolve_box_for_frame(infile: h5py.File, frame_idx: int) -> np.ndarray:
    if "particles/all/box/edges/value" in infile:
        box_raw = np.asarray(infile["particles/all/box/edges/value"][frame_idx])
        if box_raw.ndim == 2 and box_raw.shape == (3, 3):
            return np.diag(box_raw)
        if box_raw.ndim == 1 and box_raw.shape[0] == 3:
            return box_raw
    if "box" in infile.attrs:
        return np.asarray(infile.attrs["box"], dtype=np.float64)
    raise KeyError(
        "Cannot unwrap molecules: no box information found in H5 file."
    )


def _resolve_unwrap_state(infile: h5py.File, topol_path: str | None = None):
    molecules = None
    if "molecules" in infile:
        molecules = np.asarray(infile["molecules"])
    elif "parameters/vmd_structure/resid" in infile:
        molecules = np.asarray(infile["parameters/vmd_structure/resid"])

    if molecules is None:
        return None, "missing molecule IDs ('molecules' or 'parameters/vmd_structure/resid')"

    # --- Try bond topology from the H5 file first --------------------------
    if (
        "parameters/vmd_structure/bond_from" in infile
        and "parameters/vmd_structure/bond_to" in infile
    ):
        bond_from = np.asarray(infile["parameters/vmd_structure/bond_from"], dtype=np.int64).ravel()
        bond_to = np.asarray(infile["parameters/vmd_structure/bond_to"], dtype=np.int64).ravel()
        if bond_from.size > 0 and bond_to.size > 0:
            if min(int(bond_from.min()), int(bond_to.min())) >= 1:
                bond_from = bond_from - 1
                bond_to = bond_to - 1

            if "indices" in infile:
                atom_indices = np.asarray(infile["indices"], dtype=np.int64).ravel()
            elif "particles/all/indices" in infile:
                atom_indices = np.asarray(infile["particles/all/indices"], dtype=np.int64).ravel()
            else:
                atom_indices = np.arange(len(molecules), dtype=np.int64)

            row_for_index = {int(ai): row for row, ai in enumerate(atom_indices)}
            try:
                bf_rows = np.array([row_for_index[int(ai)] for ai in bond_from], dtype=np.int64)
                bt_rows = np.array([row_for_index[int(ai)] for ai in bond_to], dtype=np.int64)
                return _build_unwrap_data(molecules, bf_rows, bt_rows), None
            except KeyError:
                pass  # fall through to TOML

    # --- Fall back to TOML topology -----------------------------------------
    if topol_path is not None:
        try:
            bonds_i, bonds_j = _load_bonds_from_toml(topol_path, molecules)
            if bonds_i.size > 0:
                return _build_unwrap_data(molecules, bonds_i, bonds_j), None
        except Exception as exc:
            return None, f"failed to parse topology '{topol_path}': {exc}"

    if topol_path is None:
        return None, (
            "missing bond topology in H5 file and no --topol provided"
        )
    return None, "no bonds found in topology file"


def h5_to_xyz_species(
    h5_path: Path,
    xyz_path: Path,
    unit_scale: float = 10.0,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    unwrap_molecules: bool = False,
    topol_path: str | None = None,
) -> int:
    with h5py.File(h5_path, "r") as infile:
        atom_elements = _resolve_atom_elements(infile)
        coords_dataset = _resolve_coords_dataset(infile)

        unwrap_state = None
        if unwrap_molecules:
            unwrap_state, unwrap_reason = _resolve_unwrap_state(infile, topol_path)
            if unwrap_state is None:
                print(
                    f"Warning: --unwrap-molecules requested for '{h5_path}' but exact "
                    f"whole-molecule unwrapping is unavailable: {unwrap_reason}. "
                    "Writing raw coordinates instead.",
                    file=sys.stderr,
                )

        n_frames, n_atoms, _ = coords_dataset.shape
        frame_stop = n_frames if stop is None else min(stop, n_frames)

        written_frames = 0
        with xyz_path.open("w", encoding="utf-8") as outfile:
            for frame_idx in range(start, frame_stop, stride):
                raw_frame = coords_dataset[frame_idx]
                if np.all(raw_frame == 0) or np.any(np.isnan(raw_frame)):
                    print(f"Reached end of valid data at frame {frame_idx}. Stopping.")
                    break

                if unwrap_state is not None:
                    frame_box = _resolve_box_for_frame(infile, frame_idx)
                    raw_frame = _unwrap_molecules_same_as_diff_md(
                        raw_frame,
                        frame_box,
                        *unwrap_state,
                    )

                frame_coords = raw_frame * unit_scale
                outfile.write(f"{n_atoms}\n")
                outfile.write(f"Frame {frame_idx}\n")
                for atom_idx, symbol in enumerate(atom_elements):
                    x, y, z = frame_coords[atom_idx]
                    outfile.write(f"{symbol:5} {x:12.6f} {y:12.6f} {z:12.6f}\n")
                written_frames += 1

    return written_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trajectory conversion utilities for Diff-MD H5 files and same-system GRO frame sets. "
            "Default mode converts H5 trajectories to XYZ; use --gro-to-h5 to clone a template H5 "
            "for a directory of GRO frames after strict topology/order validation."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
              diffmd-h5toxyz -f simulation.h5 -o trajectory.xyz

              diffmd-h5toxyz -f simulation.h5 --unwrap-molecules --topol topol.toml \\
                  --start 100 --stride 10 -o unwrapped.xyz

              diffmd-h5toxyz --gro-to-h5 --gro-dir starts --template-h5 finish.h5 \\
                  --topol topol.toml --output-dir starts_h5 --overwrite
            """
        ),
    )
    parser.add_argument("-f", "--file", help="Input H5 trajectory for the default H5-to-XYZ mode.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output XYZ path. Defaults to the input stem with .xyz extension.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Coordinate scale factor applied before writing XYZ (default: 10.0).",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame to export.")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--unwrap-molecules",
        action="store_true",
        help=(
            "Apply the same whole-molecule unwrapping algorithm used by Diff-MD "
            "trajectory writing. Requires molecule IDs and bond topology "
            "(from H5 file or --topol)."
        ),
    )
    parser.add_argument(
        "--topol",
        default=None,
        help=(
            "Path to a TOML topology file (topol.toml) for bond connectivity. "
            "Used by --unwrap-molecules when the H5 file lacks embedded bond data, and required "
            "by --gro-to-h5 for template/TOML validation."
        ),
    )
    parser.add_argument(
        "--gro-to-h5",
        action="store_true",
        help="Convert a directory of same-system GRO frames into cloned Diff-MD H5 inputs using a template H5.",
    )
    parser.add_argument(
        "--gro-dir",
        default=None,
        help="Directory containing GRO frames to convert in --gro-to-h5 mode.",
    )
    parser.add_argument(
        "--gro-pattern",
        default="*.gro",
        help="Glob used to select GRO frames inside --gro-dir (default: *.gro).",
    )
    parser.add_argument(
        "--template-h5",
        default=None,
        help="Template H5 file whose metadata and layout will be cloned in --gro-to-h5 mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for converted H5 files in --gro-to-h5 mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files in --gro-to-h5 mode.",
    )
    parser.add_argument(
        "--use-gro-velocities",
        action="store_true",
        help="Use velocities stored in GRO atom lines when present; otherwise velocities are zeroed in --gro-to-h5 mode.",
    )
    return parser


def run_h5_to_xyz(args) -> int:
    if not args.file:
        raise SystemExit("ERROR: H5-to-XYZ mode requires -f / --file.")

    input_path = Path(args.file)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".xyz")
    written_frames = h5_to_xyz_species(
        input_path,
        output_path,
        unit_scale=args.scale,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        unwrap_molecules=args.unwrap_molecules,
        topol_path=args.topol,
    )
    print(f"Wrote {written_frames} frame(s) to {output_path}")
    return 0


def run_gro_to_h5(args) -> int:
    if args.gro_dir is None:
        raise SystemExit("ERROR: --gro-to-h5 requires --gro-dir.")
    if args.template_h5 is None:
        raise SystemExit("ERROR: --gro-to-h5 requires --template-h5.")
    if args.topol is None:
        raise SystemExit("ERROR: --gro-to-h5 requires --topol with the legacy topol.toml path.")

    gro_dir = Path(args.gro_dir)
    template_h5 = Path(args.template_h5)
    topol_path = Path(args.topol)
    output_dir = Path(args.output_dir) if args.output_dir else gro_dir.with_name(f"{gro_dir.name}_h5")

    try:
        written = gro_to_h5_from_template(
            gro_dir,
            template_h5,
            topol_path,
            output_dir,
            gro_pattern=args.gro_pattern,
            overwrite=args.overwrite,
            use_gro_velocities=args.use_gro_velocities,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Validated template '{template_h5}' against topology '{topol_path}'.")
    print(f"Converted {len(written)} GRO frame(s) into {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.gro_to_h5:
        return run_gro_to_h5(args)
    return run_h5_to_xyz(args)


if __name__ == "__main__":
    raise SystemExit(main())