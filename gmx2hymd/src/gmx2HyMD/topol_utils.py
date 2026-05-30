import os
from typing import Callable, TypeVar

import parse


AUTO_INCLUDE_ITP_MAP = {
    "spce.itp": "ions_spce.itp",
    "spc.itp": "ions_spc.itp",
    "tip3p.itp": "ions_tip3p.itp",
    "tip4pew.itp": "ions_tip4pew.itp",
    "opc.itp": "ions_opc.itp",
    "opc3.itp": "ions_opc3.itp",
}


def _append_auto_include_itps(itp_paths: list[str]) -> list[str]:
    extra_paths = []
    known_paths = set(itp_paths)

    for itp_path in itp_paths:
        base_name = os.path.basename(itp_path)
        auto_name = AUTO_INCLUDE_ITP_MAP.get(base_name)
        if auto_name is None:
            continue

        auto_path = os.path.join(os.path.dirname(itp_path), auto_name)
        if os.path.exists(auto_path) and auto_path not in known_paths:
            extra_paths.append(auto_path)
            known_paths.add(auto_path)

    return itp_paths + extra_paths


def _classify_dihedral_block(lines: list[str], header_index: int):
    for idx in range(header_index - 1, -1, -1):
        stripped = lines[idx].strip().lower()
        if not stripped:
            continue
        if stripped.startswith("["):
            break
        if stripped.startswith(";"):
            if "improper" in stripped:
                return True
            if "proper" in stripped:
                return False
            continue
        break
    return None


class BondedDefaults:
    def __init__(self):
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}


def _section_name(line: str):
    stripped = line.strip()
    if stripped.startswith("[") and "]" in stripped:
        return stripped.strip("[] ").lower()
    return None


def _skip_or_tokens(line: str):
    stripped = line.strip()
    if not stripped or stripped.startswith(";") or stripped.startswith("#"):
        return None
    tokens = line.split(";")[0].split()
    return tokens or None


def _parse_bonded_defaults(itp_paths: list[str]) -> BondedDefaults:
    defaults = BondedDefaults()
    for itp_path in itp_paths:
        try:
            with open(itp_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
        except OSError:
            continue

        current_sec = None
        ifdef = False
        inner_ifdef = False
        for line in lines:
            section = _section_name(line)
            if section is not None:
                current_sec = section
                continue

            stripped = line.strip()
            if stripped.startswith("#if"):
                if ifdef:
                    inner_ifdef = True
                else:
                    ifdef = True
                continue
            if stripped.startswith("#endif"):
                if inner_ifdef:
                    inner_ifdef = False
                else:
                    ifdef = False
                continue
            if ifdef:
                continue

            tokens = _skip_or_tokens(line)
            if tokens is None:
                continue

            try:
                if current_sec == "bondtypes" and len(tokens) >= 5:
                    a, b = tokens[0], tokens[1]
                    func = int(tokens[2])
                    value = (float(tokens[3]), float(tokens[4]))
                    defaults.bonds[(a, b, func)] = value
                    defaults.bonds[(b, a, func)] = value
                elif current_sec == "angletypes" and len(tokens) >= 6:
                    a, b, c = tokens[0], tokens[1], tokens[2]
                    func = int(tokens[3])
                    value = (float(tokens[4]), float(tokens[5]))
                    defaults.angles[(a, b, c, func)] = value
                    defaults.angles[(c, b, a, func)] = value
                elif current_sec == "dihedraltypes" and len(tokens) >= 7:
                    a, b, c, d = tokens[0], tokens[1], tokens[2], tokens[3]
                    func = int(tokens[4])
                    value = (
                        float(tokens[5]),
                        float(tokens[6]),
                        int(float(tokens[7])) if len(tokens) > 7 else 1,
                    )
                    defaults.dihedrals.setdefault((a, b, c, d, func), []).append(value)
            except ValueError:
                continue
    return defaults


def _atom_type(atom_lookup: dict[int, "ItpAtom"], atom_idx: int) -> str:
    return atom_lookup[atom_idx].atomtype


def _dihedral_key_matches(query, candidate) -> bool:
    return all(c == "X" or q == c for q, c in zip(query, candidate))


def _lookup_dihedral_defaults(defaults: BondedDefaults, atom_types, func: int):
    forward = (*atom_types, func)
    reverse = (*reversed(atom_types), func)
    if forward in defaults.dihedrals:
        return defaults.dihedrals[forward]
    if reverse in defaults.dihedrals:
        return defaults.dihedrals[reverse]

    best_rows = None
    best_specificity = -1
    for key, rows in defaults.dihedrals.items():
        if key[4] != func:
            continue
        candidate = key[:4]
        if not (
            _dihedral_key_matches(atom_types, candidate)
            or _dihedral_key_matches(tuple(reversed(atom_types)), candidate)
        ):
            continue
        specificity = sum(part != "X" for part in candidate)
        if specificity > best_specificity:
            best_specificity = specificity
            best_rows = rows
    return best_rows


def _resolve_bonded_parameters(atoms, bonds, angles, dihedrals, defaults: BondedDefaults):
    atom_lookup = {atom.index: atom for atom in atoms}

    for bond in bonds:
        if bond._has_parameters:
            continue
        key = (
            _atom_type(atom_lookup, bond.atom1),
            _atom_type(atom_lookup, bond.atom2),
            bond.func,
        )
        params = defaults.bonds.get(key)
        if params is not None:
            bond.length, bond.strength = params

    for angle in angles:
        if angle._has_parameters:
            continue
        key = (
            _atom_type(atom_lookup, angle.atom1),
            _atom_type(atom_lookup, angle.atom2),
            _atom_type(atom_lookup, angle.atom3),
            angle.func,
        )
        params = defaults.angles.get(key)
        if params is not None:
            angle.length, angle.strength = params

    resolved_dihedrals = []
    for dihedral in dihedrals:
        if dihedral._has_parameters:
            resolved_dihedrals.append(dihedral)
            continue
        atom_types = (
            _atom_type(atom_lookup, dihedral.atom1),
            _atom_type(atom_lookup, dihedral.atom2),
            _atom_type(atom_lookup, dihedral.atom3),
            _atom_type(atom_lookup, dihedral.atom4),
        )
        params = _lookup_dihedral_defaults(defaults, atom_types, dihedral.func)
        if params is None:
            resolved_dihedrals.append(dihedral)
            continue
        for phi0, strength, mult in params:
            resolved = ItpDihedral(
                dihedral.atom1,
                dihedral.atom2,
                dihedral.atom3,
                dihedral.atom4,
                dihedral.func,
                phi0,
                strength,
                mult,
            )
            resolved.is_improper = dihedral.is_improper
            resolved_dihedrals.append(resolved)
    return resolved_dihedrals


class Topol:
    def __init__(self, atoms, bonds=None, angles=None, dihedrals=None, cmap=None):
        if bonds is None:
            bonds = []
        if angles is None:
            angles = []
        if dihedrals is None:
            dihedrals = []
        if cmap is None:
            cmap = []

        self.n_atoms = len(atoms)
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        # CMAP entries (only populated by --amber19sb).  Each entry is
        #   [c_prev, n, ca, c, n_next, residue_name]
        # with 1-indexed atom IDs.
        self.cmap = cmap
        self.dihedrals = dihedrals


class ItpAtom:
    def __init__(self, *args):
        # GMX format: nr type resnr residue atom cgnr charge mass
        self.index    = int(args[0])
        self.atomtype = args[1]
        self.resnr    = int(args[2])
        self.resname  = args[3]
        self.atomname = args[4]
        self.cgnr     = int(args[5])
        # GMX charge is usually the 7th column (index 6)
        self.charge   = float(args[6])
        self.mass     = float(args[7]) if len(args) > 7 else None

class ItpBond:
    def __init__(self, *args):
        # GMX format: ai aj funct b0 kb
        self.atom1    = int(args[0])
        self.atom2    = int(args[1])
        self.func     = int(args[2])
        self._has_parameters = len(args) > 3
        self.length   = float(args[3]) if len(args) > 3 else 0.0
        self.strength = float(args[4]) if len(args) > 4 else 0.0

class ItpAngle:
    def __init__(self, *args):
        # GMX format: ai aj ak funct th0 cth
        self.atom1    = int(args[0])
        self.atom2    = int(args[1])
        self.atom3    = int(args[2])
        self.func     = int(args[3])
        self._has_parameters = len(args) > 4
        self.length   = float(args[4]) if len(args) > 4 else 0.0
        self.strength = float(args[5]) if len(args) > 5 else 0.0

class ItpDihedral:
    def __init__(self, *args):
        # GMX format: ai aj ak al funct phi0 cp mult
        self.atom1    = int(args[0])
        self.atom2    = int(args[1])
        self.atom3    = int(args[2])
        self.atom4    = int(args[3])
        self.func     = int(args[4])
        
        # Categorize Dihedral vs Improper
        # GROMACS function types 2 and 4 are usually impropers
        self.is_improper = self.func in (2, 4)
        self._has_parameters = len(args) > 5
        
        # Pull parameters safely
        self.length   = float(args[5]) if len(args) > 5 else 0.0
        self.strength = float(args[6]) if len(args) > 6 else 0.0
        self.mult     = int(args[7])   if len(args) > 7 else 1


ItpSection = TypeVar("ItpSection", int, ItpAtom, ItpBond, ItpAngle, ItpDihedral)


def load_itp_section(data: list[str], class_or_fun: Callable) -> list[ItpSection]:
    """ """
    output_list = []
    ifdef = False
    inner_ifdef = False
    for line in data:
        # CHECK: what's the best way to order these if-statements?
        # break if next section starts
        if line.startswith("["):
            break

        # Skip #ifdef blocks
        if line.startswith("#if"):
            if ifdef:
                inner_ifdef = True
            else:
                ifdef = True
            continue
        if line.startswith("#endif"):
            if inner_ifdef:
                continue
            ifdef = False
            continue
        if ifdef:
            continue

        # Skip comments or empty lines
        if line in ("\n", " \n") or line.startswith(";"):
            continue
        demoline = line.split(";")[0].split()
        if not demoline:
            continue

        # CHECK: I feel like classes only add complexity without benifits here
        output_list.append(class_or_fun(*demoline))
    return output_list


def parse_itp(
    itp_file: str,
    molecule_list: dict[str, int],
    elec_label: bool,
    atomtype_masses: dict[str, float],
    bonded_defaults: BondedDefaults | None = None,
) -> tuple[dict[str, tuple[ItpAtom, ItpBond, ItpAngle, ItpDihedral]], bool]:
    # tuple[dict[str, tuple[]]] :
    atoms_list, bonds_list, angles_list, dih_list = [], [], [], []

    print("Reading ITP file:", itp_file)
    with open(itp_file, "r") as infile:
        lines = infile.readlines()

    current_sec = None
    for line in lines:
        if line.startswith("["):
            current_sec = line.strip("[] \n")
            continue

        if current_sec == "atomtypes":
            demoline = line.split(";")[0].split()
            if len(demoline) >= 3:
                try:
                    atomtype_masses[str(demoline[0])] = float(demoline[2])
                except ValueError:
                    continue

    moltype_idx = []
    for i, line in enumerate(lines):
        if line.startswith("[ moleculetype") or line.startswith("[moleculetype"):
            moltype_idx.append(i + 1)

    if not moltype_idx:
        raise ValueError("Missing [ moleculetype ] section in {itp_file}.")

    molecules = {}
    for i, idx in enumerate(moltype_idx):
        try:
            end = moltype_idx[i + 1]
        except IndexError:
            end = None
        sel_lines = lines[idx:end]

        molname = load_itp_section(sel_lines, lambda x, _: x)[0]
        if molname in molecule_list:
            sections = {}
            dihedral_sections = []
            for j, line in enumerate(sel_lines):
                pattern = parse.parse("[{}]\n", line)
                if pattern is not None:
                    sec_name = pattern[0].strip()
                    if sec_name == "dihedrals":
                        dihedral_sections.append(
                            (j + 1, _classify_dihedral_block(sel_lines, j))
                        )
                    else:
                        sections[sec_name] = j + 1

            if "atoms" in sections:
                start = sections["atoms"]
                atoms_list = load_itp_section(sel_lines[start:], ItpAtom)
                for atom in atoms_list:
                    if atom.mass is None:
                        atom.mass = atomtype_masses.get(atom.atomtype, 72.0)
                if any([atom.charge for atom in atoms_list]):
                    elec_label = True
            if "bonds" in sections:
                start = sections["bonds"]
                bonds_list = load_itp_section(sel_lines[start:], ItpBond)
            if "angles" in sections:
                start = sections["angles"]
                angles_list = load_itp_section(sel_lines[start:], ItpAngle)
            if dihedral_sections:
                dih_list = []
                for start, forced_improper in dihedral_sections:
                    block = load_itp_section(sel_lines[start:], ItpDihedral)
                    for d in block:
                        if forced_improper is not None:
                            d.is_improper = forced_improper
                            if d.func == 1:
                                d.func = 4 if forced_improper else 9
                        dih_list.append(d)
            if bonded_defaults is not None:
                dih_list = _resolve_bonded_parameters(
                    atoms_list, bonds_list, angles_list, dih_list, bonded_defaults
                )
            molecules[molname] = (atoms_list, bonds_list, angles_list, dih_list)
    return molecules, elec_label


def load_top_params(
    molecule_list: dict[str, int],
    itp_paths: list[str],
    elec_label: bool,
    bonded_itp_paths: list[str] | None = None,
) -> tuple[dict[str, Topol], int, bool]:
    topol = {}
    atomtype_masses = {}
    bonded_defaults = _parse_bonded_defaults(bonded_itp_paths or itp_paths)
    for itp in itp_paths:
        # FIXME: if file with LJ params do something else?
        # if os.path.basename(itp) in [
        # "martini.itp",
        # "martini_v2.2.itp",
        params, elec_label = parse_itp(
            itp, molecule_list, elec_label, atomtype_masses, bonded_defaults
        )
        for molname in params:
            topol[molname] = Topol(*params[molname])

    topol_atoms = 0
    print("System composition:")
    for molname in molecule_list:
        n_mol = molecule_list[molname]
        # if molname in ["W", "NA", "CL"] and molname not in topol:
        #     topol[molname] = Topol(*single_bead_itp(molname))
        topol_atoms += n_mol * topol[molname].n_atoms
        print(f"    {molname:10}\t{n_mol:>6}")
    return topol, topol_atoms, elec_label


def load_top(filename: str) -> tuple[dict[str, int], list[str]]:
    """
    Lookup top file's [ molecules ] section and included itps
    """
    itp_molecules = {}
    itp_paths = []
    with open(filename, "r") as infile:
        lines = infile.readlines()

    try:
        idx = lines.index("[ molecules ]\n")
    except ValueError:
        try:
            idx = lines.index("[molecules]\n")
        except:
            raise ValueError(f"[ molecules ] section missing in {filename}.")

    for line in lines[:idx]:
        if line.startswith("#"):
            line_wo_comment = line.split(";")[0]
            path = parse.parse("#include {}\n", line_wo_comment)
            if path != None:
                itp_paths.append(
                    f"{os.path.dirname(os.path.abspath(filename))}/{path[0][1:-1]}"
                )  # strip sorrounding quotes

    itp_paths = _append_auto_include_itps(itp_paths)

    for line in lines[idx + 1 :]:
        if line == "\n":
            # cut off at the first empty line
            break
        elif line.startswith(";"):
            # Skip comments
            continue
        else:
            # make sure only the first two elements are used
            molname, molnum = line.split()[:2]
            if molname in itp_molecules:
                itp_molecules[molname] += int(molnum)
            else:
                itp_molecules[molname] = int(molnum)
    print(f"TOP file {filename} loaded... ")
    return itp_molecules, itp_paths
