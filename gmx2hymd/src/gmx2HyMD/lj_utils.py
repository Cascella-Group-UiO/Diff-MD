"""Lennard-Jones parameter emitter for Diff-MD ``options.toml``.

Produces the ``[field].LJ_param`` block as a list of
``(type1, type2, sigma_nm, eps_kJmol)`` tuples.

Two modes:

* ``standalone`` — emit only self-rows. Diff-MD's runtime falls back to
  Lorentz–Berthelot mixing for cross interactions.
* ``coupled`` — emit self-rows plus explicit cross rows. Cross-row source
  data comes from the FF table's ``[ nonbond_params ]`` block; if a
  requested pair is missing, Lorentz–Berthelot mixing is used and a
  warning is emitted.

For self-row sigma/epsilon, ``[ nonbond_params ]`` overrides
``[ atomtypes ]`` if both define the same self-pair (GROMACS semantics).
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

from .ff_utils import FFTable


# -- coupled-spec parsing ----------------------------------------------------

def parse_coupled_spec(spec: str | None) -> tuple[str, list[frozenset[str]] | None]:
    """Parse a ``--coupled`` spec string.

    Returns ``(kind, pairs)`` where ``kind`` is one of:

    * ``"none"``   — no coupled cross rows (standalone-only).
    * ``"auto"``   — every pair in the FF's ``[ nonbond_params ]``.
    * ``"list"``   — explicit allowlist (``pairs`` is non-None).
    * ``"file"``   — load allowlist from file (``pairs`` is non-None).
    * ``"file_else_auto"`` — load allowlist from file; fall back to ``auto``
      for any pair the file does not mention (``pairs`` non-None).

    Accepted spec grammars:
        ``auto``
        ``list:A-B,C-D``
        ``file:/path/to/file``
        ``file:/path/to/file|auto``
    """
    if spec is None or spec == "" or spec == "none":
        return "none", None
    if spec == "auto":
        return "auto", None
    if spec.startswith("list:"):
        return "list", _parse_pair_csv(spec[len("list:"):])
    fallback_to_auto = False
    if spec.endswith("|auto"):
        fallback_to_auto = True
        spec = spec[: -len("|auto")]
    if spec.startswith("file:"):
        path = spec[len("file:"):]
        pairs = _load_pair_file(path)
        return ("file_else_auto" if fallback_to_auto else "file", pairs)
    raise ValueError(
        f"unrecognized --coupled spec {spec!r}; "
        "expected 'auto', 'list:A-B,...', 'file:PATH', or 'file:PATH|auto'"
    )


def _parse_pair_csv(text: str) -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"pair {chunk!r} missing '-' separator")
        a, b = chunk.split("-", 1)
        a = a.strip()
        b = b.strip()
        if not a or not b:
            raise ValueError(f"empty atomtype in pair {chunk!r}")
        out.append(frozenset((a, b)))
    return out


def _load_pair_file(path: str) -> list[frozenset[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"coupled-pair file not found: {path}")
    text = open(path, "r").read()
    # Strip comments
    lines: list[str] = []
    for raw in text.splitlines():
        idx = raw.find("#")
        if idx >= 0:
            raw = raw[:idx]
        raw = raw.strip()
        if raw:
            lines.append(raw)
    return _parse_pair_csv(",".join(lines))


# -- emitter -----------------------------------------------------------------

def _decode(name) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _unique_present_types(system) -> list[str]:
    """Return atomtype names present in ``system.names`` in first-seen order."""
    seen: dict[str, None] = {}
    for raw in system.names:
        n = _decode(raw)
        if n not in seen:
            seen[n] = None
    return list(seen.keys())


def _self_lj(fftable: FFTable, name: str) -> tuple[float, float]:
    """Self sigma/epsilon, honoring ``[ nonbond_params ]`` override."""
    key = frozenset((name,))  # frozenset of one element (A,A)
    # GROMACS lists self pairs as "A A" -> frozenset({A}). Check both
    # the single-element form and the two-element form (some files use
    # explicit duplicate tokens).
    for k in (key, frozenset((name, name))):
        if k in fftable.nonbond_params:
            sig, eps, _func = fftable.nonbond_params[k]
            return sig, eps
    if name not in fftable.atomtypes:
        raise KeyError(f"atomtype {name!r} not in FF table")
    at = fftable.atomtypes[name]
    return at.sigma, at.epsilon


def build_lj_param(
    system,
    fftable: FFTable,
    *,
    mode: str = "standalone",
    coupled_spec: str | None = None,
    warn_missing: bool = True,
) -> list[tuple[str, str, float, float]]:
    """Build the ``[field].LJ_param`` block.

    Parameters
    ----------
    system
        Object with a ``.names`` array of bead atomtype names (bytes or str).
    fftable
        Output of :func:`gmx2HyMD.ff_utils.parse_ff_tree`.
    mode
        ``"standalone"`` or ``"coupled"``.
    coupled_spec
        Spec for which pairs to emit in coupled mode (see
        :func:`parse_coupled_spec`).
    warn_missing
        If True, emit a warning for any cross pair that had to fall back
        to combination-rule mixing.

    Returns
    -------
    list of ``(type1, type2, sigma_nm, eps_kJmol)`` tuples.
    """
    if mode not in ("standalone", "coupled"):
        raise ValueError(f"mode must be 'standalone' or 'coupled', got {mode!r}")

    present = _unique_present_types(system)
    rows: list[tuple[str, str, float, float]] = []

    # 1. Self-rows
    for name in present:
        sig, eps = _self_lj(fftable, name)
        rows.append((name, name, sig, eps))

    if mode == "standalone":
        return rows

    # 2. Coupled cross rows
    kind, requested = parse_coupled_spec(coupled_spec or "auto")
    present_set = set(present)

    if kind == "none":
        return rows

    if kind == "auto":
        candidate_pairs = [
            pair for pair in fftable.nonbond_params
            if len(pair) == 2 and pair.issubset(present_set)
        ]
    elif kind == "list":
        candidate_pairs = [
            pair for pair in (requested or []) if pair.issubset(present_set)
        ]
    elif kind == "file":
        candidate_pairs = [
            pair for pair in (requested or []) if pair.issubset(present_set)
        ]
    elif kind == "file_else_auto":
        file_pairs = {
            pair for pair in (requested or []) if pair.issubset(present_set)
        }
        auto_pairs = {
            pair for pair in fftable.nonbond_params
            if len(pair) == 2 and pair.issubset(present_set)
        }
        candidate_pairs = sorted(file_pairs | auto_pairs, key=lambda p: sorted(p))
    else:
        raise AssertionError(f"unreachable kind={kind!r}")

    emitted: set[frozenset[str]] = set()
    for pair in candidate_pairs:
        a, b = sorted(pair)
        if a == b:
            continue  # self-rows already added
        key = frozenset((a, b))
        if key in emitted:
            continue
        emitted.add(key)
        sig, eps, origin = fftable.cross(a, b)
        if origin == "mixed" and warn_missing:
            warnings.warn(
                f"LJ pair {a!r}-{b!r} not in [ nonbond_params ]; "
                f"using Lorentz-Berthelot mix (sigma={sig:.4f}, eps={eps:.4f})",
                stacklevel=2,
            )
        rows.append((a, b, sig, eps))

    return rows


# -- TOML formatting ---------------------------------------------------------

def format_lj_param_toml(rows: Iterable[tuple[str, str, float, float]]) -> str:
    """Render ``rows`` as a TOML array-of-arrays literal.

    Output matches the hand-written reference TOMLs in
    ``DIRxClaude_example/multi_mpi/DIFF_SYS/*/options.toml``.
    """
    lines = ["LJ_param = ["]
    for a, b, sig, eps in rows:
        lines.append(
            f'[ "{a}", "{b}", {sig:.6e}, {eps:.6e}],'
        )
    lines.append("]")
    return "\n".join(lines)


def format_lj_type_param_toml(rows: Iterable[tuple[str, str, float, float]]) -> str:
    """Render the per-type self-rows of ``rows`` as an ``LJ_type_param`` header.

    Atomistic Diff-MD runs (``lj_input_source = "mixing"``) take per-type
    sigma/epsilon as ``[type_name, sigma, epsilon]`` and build cross
    interactions from ``combining_rule``. Only self-rows (``type1 == type2``)
    are emitted; explicit cross/couple rows, if any, belong in the optional
    ``LJ_param`` block.

    The closing ``]`` is intentionally omitted so the caller can append custom
    ligand rows before closing the array.
    """
    lines = ["LJ_type_param = ["]
    for a, b, sig, eps in rows:
        if a != b:
            continue
        lines.append(f'    [ "{a}", {sig:.6e}, {eps:.6e} ],')
    return "\n".join(lines)
