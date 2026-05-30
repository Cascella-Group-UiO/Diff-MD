"""GROMACS force-field tree parsing utilities.

Parses the bits of a GROMACS topology bundle that ``topol_utils.parse_itp``
ignores: ``[ defaults ]``, ``[ atomtypes ]``, ``[ nonbond_params ]``.

The parser walks the same ``itp_paths`` list that ``topol_utils.load_top``
already returns, plus any ``#include`` directives encountered while reading
those files (depth-limited, cycle-guarded). It is intentionally *additive*:
it does not modify or rely on ``parse_itp`` state.

Units convention (same as Diff-MD's ``[field].LJ_param``):
    sigma  in nm
    epsilon in kJ/mol
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import FrozenSet


# -- public dataclasses ------------------------------------------------------

@dataclass
class AtomType:
    name: str
    mass: float = 0.0
    charge: float = 0.0
    ptype: str = "A"
    sigma: float = 0.0      # nm
    epsilon: float = 0.0    # kJ/mol
    bondtype: str | None = None


@dataclass
class FFTable:
    nbfunc: int = 1                 # 1 = LJ
    combination_rule: int = 2       # 1=geom, 2=LB, 3=geom (gen_pairs)
    gen_pairs: bool = False
    fudgeLJ: float = 1.0
    fudgeQQ: float = 1.0
    atomtypes: dict[str, AtomType] = field(default_factory=dict)
    nonbond_params: dict[FrozenSet[str], tuple[float, float, int]] = field(
        default_factory=dict
    )  # key = frozenset({A,B}), value = (sigma_nm, eps_kJmol, func)
    source_files: list[str] = field(default_factory=list)

    def cross(self, a: str, b: str) -> tuple[float, float, str]:
        """Return ``(sigma_nm, eps_kJmol, origin)`` for the A-B pair.

        ``origin`` is ``"explicit"`` if the pair was present in
        ``[ nonbond_params ]``, otherwise ``"mixed"`` (combination-rule
        mix from the two self atomtypes).
        """
        key = frozenset((a, b))
        if key in self.nonbond_params:
            sig, eps, _func = self.nonbond_params[key]
            return sig, eps, "explicit"
        if a not in self.atomtypes or b not in self.atomtypes:
            raise KeyError(
                f"cannot mix LJ for ({a!r}, {b!r}): missing atomtype self-row"
            )
        at_a = self.atomtypes[a]
        at_b = self.atomtypes[b]
        sig, eps = lorentz_berthelot(
            at_a.sigma, at_a.epsilon, at_b.sigma, at_b.epsilon, self.combination_rule
        )
        return sig, eps, "mixed"


# -- helpers -----------------------------------------------------------------

_SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z_]+)\s*\]")
_INCLUDE_RE = re.compile(r'^\s*#include\s+["\']([^"\']+)["\']')
_MARTINI_NAME_RE = re.compile(r"^[STQPCNWUX]\d*[a-z]?$")


def lorentz_berthelot(
    sig_a: float, eps_a: float, sig_b: float, eps_b: float, rule: int
) -> tuple[float, float]:
    """Apply GROMACS combination rule 1 / 2 / 3."""
    if rule == 2:
        sig = 0.5 * (sig_a + sig_b)
        eps = math.sqrt(max(eps_a * eps_b, 0.0))
    else:
        # rules 1 and 3: geometric mean on both sigma and epsilon
        sig = math.sqrt(max(sig_a * sig_b, 0.0))
        eps = math.sqrt(max(eps_a * eps_b, 0.0))
    return sig, eps


def _strip_comment(line: str) -> str:
    idx = line.find(";")
    if idx >= 0:
        line = line[:idx]
    return line


def _iter_lines(path: str, _seen: set[str], _depth: int = 0):
    """Yield (line, source_path) for ``path`` and any ``#include``d files.

    Cycle-guarded by absolute-path membership in ``_seen``. Depth-limited to
    16 to catch pathological recursion.
    """
    if _depth > 16:
        raise RecursionError(
            f"FF include depth > 16 while reading {path!r}"
        )
    abs_path = os.path.abspath(path)
    if abs_path in _seen:
        return
    _seen.add(abs_path)
    base_dir = os.path.dirname(abs_path)
    try:
        with open(abs_path, "r") as fh:
            raw = fh.readlines()
    except FileNotFoundError:
        # missing #include is non-fatal in GROMACS workflows; skip silently.
        return
    for line in raw:
        m = _INCLUDE_RE.match(line)
        if m:
            inc = m.group(1)
            inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            yield from _iter_lines(inc_path, _seen, _depth + 1)
            continue
        yield line, abs_path


def _parse_defaults(tokens: list[str], table: FFTable) -> None:
    # nbfunc combrule [gen_pairs [fudgeLJ [fudgeQQ]]]
    if len(tokens) >= 1:
        table.nbfunc = int(tokens[0])
    if len(tokens) >= 2:
        table.combination_rule = int(tokens[1])
    if len(tokens) >= 3:
        table.gen_pairs = tokens[2].strip().lower() in ("yes", "y", "true", "1")
    if len(tokens) >= 4:
        table.fudgeLJ = float(tokens[3])
    if len(tokens) >= 5:
        table.fudgeQQ = float(tokens[4])


def _parse_atomtype_row(tokens: list[str]) -> AtomType | None:
    """Parse one ``[ atomtypes ]`` row.

    GROMACS allows several column layouts. The two we care about:

    Short (Martini-style, no bondtype, no atomic number):
        name  mass  charge  ptype  sigma  epsilon

    Long (GROMOS/AMBER-style):
        name  bondtype  atomic_number  mass  charge  ptype  sigma  epsilon
        (or)  name  atomic_number  mass  charge  ptype  sigma  epsilon

    Heuristic: the last two tokens are always sigma/epsilon (floats); the
    token before them is ptype (single letter A/S/V/D); we walk left from
    ptype to peel charge, mass, optional atomic number, optional bondtype.
    """
    if len(tokens) < 6:
        return None
    try:
        eps = float(tokens[-1])
        sig = float(tokens[-2])
    except ValueError:
        return None
    ptype = tokens[-3]
    if ptype not in ("A", "S", "V", "D", "M"):
        return None
    try:
        charge = float(tokens[-4])
        mass = float(tokens[-5])
    except ValueError:
        return None
    name = tokens[0]
    bondtype: str | None = None
    head = tokens[1:-5]
    # head can be [], [atomic_number], [bondtype], or [bondtype, atomic_number]
    if len(head) == 1:
        if not _looks_like_int(head[0]):
            bondtype = head[0]
    elif len(head) >= 2:
        bondtype = head[0]
    return AtomType(
        name=name,
        mass=mass,
        charge=charge,
        ptype=ptype,
        sigma=sig,
        epsilon=eps,
        bondtype=bondtype,
    )


def _looks_like_int(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def _parse_nonbond_row(tokens: list[str]) -> tuple[str, str, int, float, float] | None:
    """``A B func sigma eps`` (combrule-2/3, sigma-epsilon form)."""
    if len(tokens) < 5:
        return None
    try:
        func = int(tokens[2])
        sig = float(tokens[3])
        eps = float(tokens[4])
    except ValueError:
        return None
    return tokens[0], tokens[1], func, sig, eps


# -- public API --------------------------------------------------------------

def parse_ff_tree(itp_paths: list[str]) -> FFTable:
    """Scan ``itp_paths`` (and any ``#include``s within) for FF tables.

    The returned :class:`FFTable` consolidates all ``[ defaults ]``,
    ``[ atomtypes ]``, ``[ nonbond_params ]`` rows found across the bundle.
    Later definitions override earlier ones (GROMACS semantics).
    """
    table = FFTable()
    seen: set[str] = set()
    defaults_set = False
    in_section: str | None = None
    for path in itp_paths:
        for line, src in _iter_lines(path, seen):
            stripped = _strip_comment(line).strip()
            if not stripped:
                continue
            m = _SECTION_RE.match(stripped)
            if m:
                in_section = m.group(1).lower()
                continue
            if in_section == "defaults":
                if not defaults_set:
                    _parse_defaults(stripped.split(), table)
                    defaults_set = True
                if src not in table.source_files:
                    table.source_files.append(src)
                continue
            if in_section == "atomtypes":
                at = _parse_atomtype_row(stripped.split())
                if at is not None:
                    table.atomtypes[at.name] = at
                    if src not in table.source_files:
                        table.source_files.append(src)
                continue
            if in_section == "nonbond_params":
                row = _parse_nonbond_row(stripped.split())
                if row is not None:
                    a, b, func, sig, eps = row
                    table.nonbond_params[frozenset((a, b))] = (sig, eps, func)
                    if src not in table.source_files:
                        table.source_files.append(src)
                continue
            # other sections (moleculetype, atoms, ...) are handled by parse_itp
    return table


def detect_ff_family(
    fftable: FFTable,
    *,
    cg_flag: bool = False,
    explicit: str | None = None,
) -> str:
    """Return one of ``"martini"``, ``"amber_like"``, ``"amber19sb"``.

    Precedence:
      1. ``explicit`` argument wins if provided.
      2. ``cg_flag=True`` forces ``"martini"``.
      3. Heuristic on source filenames + atomtype-name shape.
    """
    if explicit is not None:
        if explicit not in ("martini", "amber_like", "amber19sb"):
            raise ValueError(
                f"unknown ff_family {explicit!r}; "
                "expected 'martini', 'amber_like', or 'amber19sb'"
            )
        return explicit
    if cg_flag:
        return "martini"
    # filename heuristic
    for src in fftable.source_files:
        base = os.path.basename(src).lower()
        if "martini" in base:
            return "martini"
    # atomtype-name heuristic: ≥50% of names match Martini regex AND
    # at least one nonbond_params entry exists (Martini ships a cross table)
    if fftable.atomtypes and fftable.nonbond_params:
        n_match = sum(
            1 for name in fftable.atomtypes if _MARTINI_NAME_RE.match(name)
        )
        if n_match >= 0.5 * len(fftable.atomtypes):
            return "martini"
    return "amber_like"
