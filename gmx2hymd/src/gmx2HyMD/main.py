import argparse
import os
import textwrap
import time
from argparse import RawDescriptionHelpFormatter

import numpy as np

from .cmap_utils import (
    build_cmap_entries_for_chain,
    cmap_grids_to_toml_str,
    parse_cmap_itp,
)
from .ff_utils import detect_ff_family, parse_ff_tree
from .gro_utils import load_gro
from .h5_utils import three_to_one, write_coordinates
from .lj_utils import build_lj_param, format_lj_param_toml
from .pdb_utils import load_pdb
from .system_properties import derive_thermo_groups, get_system_properties
from .toml_utils import write_topology
from .topol_utils import load_top, load_top_params


def parse_ligand_itp(itp_path: str) -> list:
    """Extract custom Lennard-Jones parameters from a ligand ITP file."""
    custom_lj_lines = []
    if not os.path.exists(itp_path):
        print(f"WARNING: Ligand itp file '{itp_path}' not found. Skipping ligand params.")
        return custom_lj_lines

    with open(itp_path, "r", encoding="utf-8") as infile:
        in_atomtypes = False
        for line in infile:
            line = line.split(";")[0].strip()
            if not line:
                continue

            if line.startswith("["):
                in_atomtypes = line == "[ atomtypes ]"
                continue

            if in_atomtypes:
                parts = line.split()
                if len(parts) >= 7:
                    atype = parts[0]
                    sigma = parts[5]
                    epsilon = parts[6]
                    custom_lj_lines.append(
                        f"    [ '{atype}',    {sigma},  {epsilon}   ],\n"
                    )

    return custom_lj_lines


def _has_moleculetype(itp_path: str) -> bool:
    """Cheap pre-scan: True iff the .itp defines at least one molecule.

    Used to split a topology bundle's includes into (FF-only, molecule)
    groups so that ``parse_itp`` is never called on a pure-FF file
    (``[ defaults ] / [ atomtypes ] / [ nonbond_params ]`` only), which
    would otherwise raise ``Missing [ moleculetype ]``.
    """
    try:
        with open(itp_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("[") and "moleculetype" in stripped.lower():
                    return True
    except OSError:
        return False
    return False


def _apply_fftable_masses(topol: dict, fftable) -> None:
    """Patch defaulted atom masses using the FF atomtype table.

    ``parse_itp`` gets only molecule-bearing itps, so pure force-field files
    no longer flow through its legacy ``[ atomtypes ]`` mass harvest. Keep the
    safer itp split, but restore the old mass behavior here for atom rows that
    fell back to ``parse_itp``'s 72.0 default.
    """
    for params in topol.values():
        for atom in params.atoms:
            atomtype = fftable.atomtypes.get(atom.atomtype)
            if atomtype is None:
                continue
            if atom.mass is None or abs(float(atom.mass) - 72.0) < 1e-12:
                atom.mass = atomtype.mass


def _resolve_thermo_groups(spec: str, system) -> list[list[str]] | None:
    """Convert ``--thermo-groups`` CLI spec into a list of groups or ``None``.

    ``None`` means "use the legacy solvent/non-solvent split inside
    :func:`write_simulation_parameters`" (backwards compatible default).
    """
    if spec is None or spec == "legacy":
        return None
    if spec == "none":
        return []
    if spec == "auto":
        return derive_thermo_groups(system)
    if spec.startswith("manual:"):
        body = spec[len("manual:"):]
        groups: list[list[str]] = []
        for chunk in body.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            names = [x.strip() for x in chunk.split(",") if x.strip()]
            if names:
                groups.append(names)
        return groups
    raise ValueError(
        f"unrecognized --thermo-groups spec {spec!r}; expected "
        "'legacy', 'auto', 'none', or 'manual:G1a,G1b;G2a'."
    )


def _format_thermo_groups_block(groups: list[list[str]]) -> str:
    """Format ``thermostat_coupling_groups`` as a TOML array-of-arrays."""
    lines = ["thermostat_coupling_groups = ["]
    for grp in groups:
        rendered = ", ".join(f"'{name}'" for name in grp)
        lines.append(f"    [{rendered}],")
    lines.append("]\n")
    return "\n".join(lines)


def write_simulation_parameters(
    names: np.ndarray,
    resnames: np.ndarray,
    ligand_lj_lines: list = None,
    cmap_block: str | None = None,
    *,
    lj_param: list[tuple[str, str, float, float]] | None = None,
    ff_family: str | None = None,
    thermo_groups: list[list[str]] | None = None,
):
    """Render ``options.toml`` from ``template.toml``.

    Backwards-compatible: when all new kwargs are ``None``, behavior matches
    the pre-CG version (solvent/non-solvent thermostat heuristic, no
    ``LJ_param`` injection, ``ff_family`` left at template default).
    """
    template = os.path.abspath(os.path.dirname(__file__)) + "/template.toml"
    out_lines = []

    all_names = np.array(names, dtype=str)
    all_resnames = np.array(resnames, dtype=str)
    _, idx = np.unique(all_names, return_index=True)
    unique_names = all_names[np.sort(idx)]
    solvent_or_ion_resnames = {
        "W", "SOL", "HOH", "NA", "NA+", "CL", "CL-", "CA", "CA+",
        "MG", "MG2+", "K", "K+", "LI", "LI+", "RB", "RB+", "CS", "CS+",
    }
    filter = np.array([
        set(all_resnames[all_names == name]).issubset(solvent_or_ion_resnames)
        for name in unique_names
    ])

    with open(template, "r", encoding="utf-8") as infile:
        in_lj_array = False
        in_field_lj_param = False
        for line in infile:
            stripped = line.strip()

            if ff_family is not None and stripped.startswith("ff_family"):
                out_lines.append(f'ff_family = "{ff_family}"\n')
                continue

            if ff_family == "martini" and stripped.startswith("combining_rule"):
                out_lines.append('combining_rule = "pairtable"\n')
                continue

            if ff_family == "martini" and stripped.startswith("lj_input_source"):
                out_lines.append('lj_input_source = "input"\n')
                continue

            if stripped.startswith("thermostat_coupling_groups"):
                if thermo_groups is not None:
                    out_lines.append(_format_thermo_groups_block(thermo_groups))
                else:
                    new_line = (
                        "thermostat_coupling_groups = [\n"
                        f"    {np.array2string(unique_names[~filter], separator=', ')},\n"
                        f"    {np.array2string(unique_names[filter], separator=', ')},\n]\n"
                    )
                    out_lines.append(new_line)
                continue

            if stripped.startswith("LJ_param"):
                # Inject the freshly-built LJ_param block in place of the
                # template's empty placeholder.
                if lj_param is not None:
                    out_lines.append(format_lj_param_toml(lj_param) + "\n")
                    in_field_lj_param = True
                    continue
                out_lines.append(line)
                in_field_lj_param = True
                continue

            if in_field_lj_param:
                if stripped == "]":
                    if lj_param is None:
                        out_lines.append(line)
                    in_field_lj_param = False
                continue

            if stripped.startswith("LJ_type_param"):
                if ff_family == "martini":
                    in_lj_array = True
                    continue
                out_lines.append(line)
                in_lj_array = True
                continue

            if in_lj_array:
                if stripped == "]":
                    if ff_family == "martini":
                        in_lj_array = False
                        continue
                    if ligand_lj_lines:
                        out_lines.append("    # --- CUSTOM LIGAND TYPES --- \n")
                        out_lines.extend(ligand_lj_lines)
                    out_lines.append(line)
                    in_lj_array = False
                    continue
                if ff_family == "martini":
                    continue

            out_lines.append(line)

    with open("options.toml", "w", encoding="utf-8") as outfile:
        outfile.write("".join(out_lines))
        if cmap_block is not None:
            # Append the CMAP grid table at the end of options.toml.  The
            # block has its own [cmap] header so it does not clash with
            # any existing template section.
            outfile.write("\n")
            outfile.write(cmap_block)


def user_input() -> argparse.Namespace:
    description = "Convert GROMACS coordinates and topology to Diff-MD-compatible H5 and TOML inputs."
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
              gmx2diffmd -f start.gro -p topol.top

              gmx2diffmd -f start.pdb -p topol.top --box 6.0 6.0 8.0 \\
                  -oc output.h5 -op topol.toml

              gmx2diffmd -f start.gro -p topol.top --amber19sb \\
                  --ff-dir amber19sb.ff

            Notes
            -----
              --amber19sb expects cmap.itp at <ff-dir>/cmap.itp. If --ff-dir is
              omitted, the converter looks for amber19sb.ff next to the input .top.
            """
        ),
    )
    ap.add_argument(
        "-f",
        "--file",
        required=True,
        dest="input",
        type=str,
        help="Input .gro/.pdb file",
    )
    ap.add_argument(
        "-p", "--top", required=True, dest="top", type=str, help="Input .top file"
    )
    ap.add_argument(
        "-b",
        "--box",
        nargs=3,
        type=float,
        default=None,
        help="Box size, takes 3 inputs (x, y, z). Required if the input pdb file does not provide it.",
    )
    ap.add_argument(
        "-oc",
        dest="out_h5",
        default=None,
        type=str,
        help="Output H5MD file (defaults to 'output.h5')",
    )
    ap.add_argument(
        "-op",
        dest="out_toml",
        default=None,
        type=str,
        help="Output toml file (defaults to 'topol.toml')",
    )
    ap.add_argument(
        "--nopar",
        action="store_true",
        dest="no_params",
        help="Disable 'options.toml' file output.",
    )
    ap.add_argument(
        "--notop",
        action="store_true",
        dest="no_topol",
        help="Disable 'topol.toml' file output.",
    )

    ap.add_argument(
        "-l",
        "--ligand",
        dest="ligand_itp",
        type=str,
        default=None,
        help="Path to the ligand .itp file to extract custom atomtypes.",
    )
    ap.add_argument(
        "--amber19sb",
        action="store_true",
        dest="amber19sb",
        help=(
            "Enable AMBER ff19SB-style backbone CMAP correction. Reads cmap.itp "
            "from --ff-dir, or from amber19sb.ff next to the input .top if --ff-dir "
            "is omitted. Fails if the CMAP file cannot be parsed."
        ),
    )
    ap.add_argument(
        "--ff-dir",
        dest="ff_dir",
        type=str,
        default=None,
        help=(
            "Path to the forcefield directory containing 'cmap.itp' "
            "(only used with --amber19sb)."
        ),
    )
    ap.add_argument(
        "--cg",
        action="store_true",
        dest="cg",
        help=(
            "Treat the input as a coarse-grained bundle (Martini-like). "
            "Forces ff_family = 'martini' and default --lj-mode to 'coupled'."
        ),
    )
    ap.add_argument(
        "--ff-family",
        dest="ff_family",
        choices=("martini", "amber_like", "amber19sb"),
        default=None,
        help="Override auto-detected Diff-MD ff_family in options.toml.",
    )
    ap.add_argument(
        "--lj-mode",
        dest="lj_mode",
        choices=("auto", "standalone", "coupled", "none"),
        default="auto",
        help=(
            "How to populate [field].LJ_param: 'standalone' = self-rows only; "
            "'coupled' = self + cross rows from [ nonbond_params ]; "
            "'none' = leave empty; 'auto' = 'coupled' if Martini, else 'standalone'."
        ),
    )
    ap.add_argument(
        "--coupled",
        dest="coupled_spec",
        type=str,
        default="auto",
        help=(
            "Coupled-mode pair selector (used only when --lj-mode=coupled). "
            "Grammar: 'auto' (all pairs in [ nonbond_params ]), "
            "'list:A-B,C-D' (explicit allowlist), 'file:PATH' (allowlist from file), "
            "'file:PATH|auto' (file allowlist, falling back to auto for the rest)."
        ),
    )
    ap.add_argument(
        "--thermo-groups",
        dest="thermo_groups",
        type=str,
        default="legacy",
        help=(
            "Thermostat coupling group strategy: 'legacy' (current solvent/ion split), "
            "'auto' (polymer/solvent/ion heuristic), 'none' (empty), "
            "or 'manual:G1a,G1b;G2a' (explicit groups separated by ';')."
        ),
    )
    return ap.parse_args()


def main():
    """Read atomistic/coarse-grained .gro/.pdb and .top files and write diff-aMD inputs."""

    start_time = time.time()
    args = user_input()
    elec_label = False

    molecule_list, itp_paths = load_top(args.top)
    # Split off pure-FF itps (no [ moleculetype ]) so parse_itp does not
    # choke on them. They still participate in the FF table parsing below.
    molecule_itps = [p for p in itp_paths if _has_moleculetype(p)]
    topol, topol_atoms, elec_label = load_top_params(
        molecule_list, molecule_itps, elec_label, bonded_itp_paths=itp_paths
    )
    fftable = parse_ff_tree(itp_paths)
    _apply_fftable_masses(topol, fftable)
    molecule_idx = {mol: idx for idx, mol in enumerate(molecule_list)}

    base, ext = os.path.splitext(args.input)
    if ext == ".gro":
        atoms, box = load_gro(args.input)
        protein_in_top = any([atom.resname in three_to_one for atom in atoms])
        if protein_in_top:
            print(
                "WARNING: converting system containing protein molecules. Atoms will have the same ordering as in the input coordinate file. "
                "Depending on the system, this might lead to inconsistencies with the output topology."
            )
        else:
            atoms.sort(key=lambda x: molecule_idx[x.resname])
    elif ext == ".pdb":
        atoms, box = load_pdb(args.input, args.box)
        protein_in_top = any([atom.residue in three_to_one for atom in atoms])
        if protein_in_top:
            print(
                "WARNING: converting system containing protein molecules. Atoms will have the same ordering as in the input coordinate file. "
                "Depending on the system, this might lead to inconsistencies. Verify that molecules in topol.toml match output.h5."
            )
        else:
            atoms.sort(key=lambda x: molecule_idx[x.residue])
    else:
        raise ValueError(
            f"Input coordinate file extension should either be .gro or .pdb. Got {ext}."
        )

    # Check that the number of atoms is the same in both files
    if topol_atoms != len(atoms):
        raise ValueError(
            f"Atom number mismatch in {args.input} (# {len(atoms)}) and {args.top} (# {topol_atoms})."
        )

    if args.out_h5 is None:
        args.out_h5 = "./output.h5"
    if args.out_toml is None:
        args.out_toml = "./topol.toml"

    system = get_system_properties(molecule_list, topol)

    ligand_lj_lines = []
    if args.ligand_itp:
        print(f"Parsing custom ligand parameters from {args.ligand_itp}...")
        ligand_lj_lines = parse_ligand_itp(args.ligand_itp)

    cmap_block = None
    if args.amber19sb:
        ff_dir = args.ff_dir
        if ff_dir is None:
            # Default: assume cmap.itp lives next to the input topology
            # in a sibling 'amber19sb.ff' directory.
            ff_dir = os.path.join(os.path.dirname(args.top), "amber19sb.ff")
        cmap_path = os.path.join(ff_dir, "cmap.itp")
        print(f"--amber19sb: reading CMAP grids from {cmap_path}")
        cmap_grids = parse_cmap_itp(cmap_path)
        print(f"--amber19sb: parsed {len(cmap_grids)} CMAP grids "
              f"({sorted(cmap_grids.keys())})")
        cmap_block = cmap_grids_to_toml_str(cmap_grids)
        # Walk every protein chain and inject a `cmap` field on its
        # Topol object so toml_utils.write_topology serializes it.
        n_total = 0
        for mol_name, params in topol.items():
            cmap_entries = build_cmap_entries_for_chain(params.atoms, cmap_grids)
            params.cmap = cmap_entries
            n_total += len(cmap_entries)
            if cmap_entries:
                print(f"  {mol_name}: {len(cmap_entries)} CMAP entries")
        print(f"--amber19sb: total {n_total} CMAP entries across all chains")

    if not args.no_topol:
        write_topology(molecule_list, topol, args.out_toml)

    if not args.no_params:
        # FF tree parsing + LJ_param / ff_family / thermostat-group resolution
        if args.amber19sb:
            ff_family = "amber19sb"
        else:
            ff_family = detect_ff_family(
                fftable, cg_flag=args.cg, explicit=args.ff_family
            )

        lj_mode = args.lj_mode
        if lj_mode == "auto":
            lj_mode = "coupled" if ff_family == "martini" else "standalone"

        lj_param_rows = None
        if lj_mode != "none":
            try:
                lj_param_rows = build_lj_param(
                    system,
                    fftable,
                    mode=lj_mode,
                    coupled_spec=args.coupled_spec if lj_mode == "coupled" else None,
                )
            except KeyError as exc:
                print(
                    f"WARNING: skipping LJ_param emission ({exc}). "
                    "Falling back to empty LJ_param block."
                )
                lj_param_rows = None

        thermo_spec = args.thermo_groups
        if thermo_spec == "legacy" and ff_family == "martini":
            # Martini bundles always want the polymer/solvent/ion split.
            thermo_spec = "auto"
        thermo_groups = _resolve_thermo_groups(thermo_spec, system)

        write_simulation_parameters(
            system.names,
            system.resnames,
            ligand_lj_lines,
            cmap_block=cmap_block,
            lj_param=lj_param_rows,
            ff_family=ff_family,
            thermo_groups=thermo_groups,
        )
        if args.amber19sb:
            print('--amber19sb: set ff_family = "amber19sb" in options.toml')

    write_coordinates(atoms, system, box, elec_label, args.out_h5)
    print(f"Conversion took {time.time() - start_time} s.")


if __name__ == "__main__":
    main()
