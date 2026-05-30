"""GROMACS-style CMAP parsing for the ``--amber19sb`` flag.

Reads the per-residue 24x24 phi/psi grids from a forcefield's
``cmap.itp`` file and emits both:

* a list of CMAP entries per protein chain
  ``[c_prev, n_curr, ca_curr, c_curr, n_next, residue_name]`` (1-indexed
  atom IDs to match the existing topology TOML convention), and
* a global ``[cmap]`` table for ``options.toml`` carrying the raw grids
  (``cmap.grids[<RES>] = [[...], ...]``) plus the grid size.

The runtime side (``src/diff_md/cmap.py``) precomputes the bicubic
spline coefficients from the raw grids; we only need to ship the raw
24x24 energies here.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np


CMAP_GRID_SIZE = 24


def parse_cmap_itp(path: str, grid_size: int = CMAP_GRID_SIZE) -> dict[str, np.ndarray]:
    """Parse a GROMACS ``cmap.itp`` file.

    Returns
    -------
    dict[residue_name, ndarray of shape (grid_size, grid_size)]
        Energies in kJ/mol on the phi (rows) x psi (cols) mesh.

    Format reminder (GROMACS)::

        [ cmaptypes ]
        C-* N-GLY XC-GLY C-GLY N-* 1 24 24\\
        e_00 e_01 ... e_0(G-1)\\
        e_10 ...                \\
        ... (G*G entries total)

        C-* N-ALA ...

    The 5 atom-type tokens encode the (i-1, i, i, i, i+1) backbone with
    the central residue replicated; the residue name in token 2 (after
    splitting on ``-``) identifies the grid.  Function type and grid
    dimensions follow.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"cmap.itp not found at '{path}'.")

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Strip GROMACS-style comments and join continuation lines.
    raw = re.sub(r";[^\n]*", "", raw)         # remove ';' comments
    raw = raw.replace("\\\n", " ")             # join backslash-continued lines

    # Locate the [ cmaptypes ] section.
    sections = re.split(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]", raw)
    grids: dict[str, np.ndarray] = {}
    expected = grid_size * grid_size
    for header, body in zip(sections[1::2], sections[2::2]):
        if header.strip().lower() != "cmaptypes":
            continue
        # Each entry begins with 5 atom-type tokens followed by the
        # function index (1) and the two grid dimensions; then ``expected``
        # floats.  Multiple entries are concatenated under one section.
        tokens = body.split()
        idx = 0
        while idx < len(tokens):
            if idx + 8 > len(tokens):
                raise ValueError(
                    f"Malformed CMAP entry in '{path}': expected 5 atom types, "
                    "function type, and two grid dimensions before grid values."
                )
            atom_types = tokens[idx : idx + 5]
            try:
                func = int(tokens[idx + 5])
                gx = int(tokens[idx + 6])
                gy = int(tokens[idx + 7])
            except ValueError as exc:
                raise ValueError(
                    f"Malformed CMAP header in '{path}' for {atom_types}: "
                    "function type and grid dimensions must be integers."
                ) from exc
            if (gx, gy) != (grid_size, grid_size):
                raise ValueError(
                    f"CMAP grid size mismatch in '{path}': "
                    f"expected ({grid_size},{grid_size}) but got ({gx},{gy})."
                )
            if func != 1:
                raise ValueError(
                    f"CMAP function type must be 1; got {func} in '{path}'."
                )
            data_start = idx + 8
            data_end = data_start + expected
            if data_end > len(tokens):
                raise ValueError(
                    f"Truncated CMAP grid in '{path}' for {atom_types}."
                )
            values = np.asarray(tokens[data_start:data_end], dtype=np.float64)
            # Token 2 is "N-RES"; the residue name is the second part.
            residue = atom_types[1].split("-", 1)[-1].strip()
            grids[residue] = values.reshape(grid_size, grid_size)
            idx = data_end

    if not grids:
        raise ValueError(f"No [ cmaptypes ] entries parsed from '{path}'.")
    return grids


def build_cmap_entries_for_chain(
    atoms: Iterable,
    grids: dict[str, np.ndarray],
) -> list[list]:
    """Build CMAP entries for a single protein chain.

    Parameters
    ----------
    atoms : iterable of objects with ``index`` (1-based), ``atomname``,
        ``resnr``, ``resname`` attributes (matches gmx2HyMD ``ItpAtom``).
    grids : residue->grid dict from :func:`parse_cmap_itp`.

    Returns
    -------
    list of [c_prev, n_curr, ca_curr, c_curr, n_next, residue_name]
        Atom IDs are 1-indexed to match the existing dihedrals/impropers
        convention.
    """
    by_resnr: dict[int, list] = {}
    for atom in atoms:
        by_resnr.setdefault(int(atom.resnr), []).append(atom)

    sorted_resnrs = sorted(by_resnr.keys())
    if len(sorted_resnrs) < 3:
        return []

    def _find(reslist, name):
        for a in reslist:
            if str(a.atomname).strip() == name:
                return a
        return None

    entries: list[list] = []
    skipped: dict[str, int] = {}
    for i in range(1, len(sorted_resnrs) - 1):
        prev_res = by_resnr[sorted_resnrs[i - 1]]
        curr_res = by_resnr[sorted_resnrs[i]]
        next_res = by_resnr[sorted_resnrs[i + 1]]

        c_prev = _find(prev_res, "C")
        n_curr = _find(curr_res, "N")
        ca_curr = _find(curr_res, "CA")
        c_curr = _find(curr_res, "C")
        n_next = _find(next_res, "N")
        if not all([c_prev, n_curr, ca_curr, c_curr, n_next]):
            continue

        residue_name = str(curr_res[0].resname).strip()
        if residue_name not in grids:
            skipped[residue_name] = skipped.get(residue_name, 0) + 1
            continue

        entries.append(
            [
                int(c_prev.index),
                int(n_curr.index),
                int(ca_curr.index),
                int(c_curr.index),
                int(n_next.index),
                residue_name,
            ]
        )

    if skipped:
        report = ", ".join(f"{name}: {n}" for name, n in sorted(skipped.items()))
        print(
            f"INFO: gmx2HyMD --amber19sb: skipped CMAP entries for residues "
            f"with no grid in cmap.itp ({report})."
        )

    return entries


def cmap_grids_to_toml_str(grids: dict[str, np.ndarray], grid_size: int = CMAP_GRID_SIZE) -> str:
    """Serialize the parsed CMAP grids as a ``[cmap]`` TOML block.

    Returned as plain TOML text so we can embed it directly into the
    options.toml output (alongside the templated other sections).  We
    don't use ``tomlkit`` here because tomlkit's nested-array handling is
    sluggish for 29 * 24 * 24 = 16k floats.
    """
    lines: list[str] = []
    lines.append("[cmap]")
    lines.append(f"grid_size = {int(grid_size)}")
    lines.append("")
    lines.append("[cmap.grids]")
    for residue in sorted(grids):
        grid = grids[residue]
        if grid.shape != (grid_size, grid_size):
            raise ValueError(
                f"Grid '{residue}' has shape {grid.shape}; expected ({grid_size},{grid_size})."
            )
        lines.append(f"{residue} = [")
        for row in grid:
            row_str = ", ".join(f"{v:.8f}" for v in row)
            lines.append(f"    [{row_str}],")
        lines.append("]")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CMAP_GRID_SIZE",
    "parse_cmap_itp",
    "build_cmap_entries_for_chain",
    "cmap_grids_to_toml_str",
]
