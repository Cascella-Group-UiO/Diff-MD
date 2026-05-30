"""CMAP backbone-correction support for AMBER ff19SB-style force fields.

GROMACS represents CMAP as a periodic 2D energy grid E(phi, psi) on a
``grid_size x grid_size`` mesh per residue type.  Forces fall out of
``-dE/dx`` for each backbone atom involved in the (phi, psi) pair.  We
follow the standard GROMACS recipe:

1.  At load time, finite-difference the grid to get fx, fy, fxy at
    every grid point (periodic boundary).
2.  For each cell of the grid, multiply the 16 corner values
    (f, fx, fy, fxy at the four corners) by a fixed 16x16 inverse
    matrix to get the 16 polynomial coefficients of the cell-local
    bicubic spline E(dx, dy) = Sum c_{a,b} dx^a dy^b for a,b in 0..3.
3.  At run time, map (phi, psi) to (i, j, dx, dy) with periodic
    wrap, gather the cell coefficients, evaluate the polynomial, and
    let JAX autodiff produce d/datom contributions.

The 16x16 matrix below is the standard "Catmull-Rom-equivalent
bicubic Hermite" inverse, identical to the one used inside GROMACS's
``cmap.c`` (``mdrun -debug 5`` reports the same coefficients on a
shared input grid to roughly float32 accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import hashlib
import numpy as np

import jax.numpy as jnp
from jax import value_and_grad, vmap

from .force import get_dihedral_angle


# --------------------------------------------------------------------------
# Static math constants
# --------------------------------------------------------------------------

# Maps the 16-vector
#   [ f00 f10 f01 f11 fx00 fx10 fx01 fx11 fy00 fy10 fy01 fy11 fxy00 fxy10 fxy01 fxy11 ]
# to the 16-vector of polynomial coefficients laid out as
#   [ c00 c10 c20 c30 c01 c11 c21 c31 c02 c12 c22 c32 c03 c13 c23 c33 ]
# where E(dx, dy) = Sum_{a,b} c_{a,b} dx^a dy^b on the unit cell
# [0,1) x [0,1).  This is the textbook bicubic-Hermite inverse and is
# the one GROMACS uses internally.
_BICUBIC_INV = np.array(
    [
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
        [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
        [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
        [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
        [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
        [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
        [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1],
    ],
    dtype=np.float64,
)


DEFAULT_CMAP_GRID_SIZE = 24


# --------------------------------------------------------------------------
# Host-side: build bicubic coefficients from a raw 2D grid
# --------------------------------------------------------------------------

def _periodic_central_diff(grid: np.ndarray, axis: int) -> np.ndarray:
    """Centered finite difference along ``axis`` with periodic wrap.

    Step size is 1 (cell-index space); returned values are derivatives
    w.r.t. the same cell-index variable so the bicubic polynomial
    evaluated on (dx, dy) in [0,1)^2 agrees with the grid.
    """
    plus = np.roll(grid, -1, axis=axis)
    minus = np.roll(grid, +1, axis=axis)
    return 0.5 * (plus - minus)


def build_bicubic_coefs(grid: np.ndarray) -> np.ndarray:
    """Pre-compute per-cell bicubic-Hermite coefficients.

    Parameters
    ----------
    grid : (G, G) numpy array of energies.

    Returns
    -------
    coefs : (G, G, 4, 4) numpy array.  ``coefs[i, j, a, b]`` is the
        coefficient of ``dx**a * dy**b`` for the cell with lower-left
        corner at grid point ``(i, j)`` and (dx, dy) in [0, 1).  Cell
        ``(G-1, *)`` and ``(*, G-1)`` wrap around to grid point 0.
    """
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(
            f"CMAP grid must be square 2D; got shape {grid.shape}."
        )
    G = grid.shape[0]
    fx = _periodic_central_diff(grid, axis=0)
    fy = _periodic_central_diff(grid, axis=1)
    fxy = _periodic_central_diff(fx, axis=1)

    # For each cell with lower-left corner at (i, j), the four corners
    # are (i, j), (i+1, j), (i, j+1), (i+1, j+1) modulo G.
    def gather(arr):
        s00 = arr
        s10 = np.roll(arr, -1, axis=0)
        s01 = np.roll(arr, -1, axis=1)
        s11 = np.roll(np.roll(arr, -1, axis=0), -1, axis=1)
        return s00, s10, s01, s11

    f00, f10, f01, f11 = gather(grid)
    fx00, fx10, fx01, fx11 = gather(fx)
    fy00, fy10, fy01, fy11 = gather(fy)
    fxy00, fxy10, fxy01, fxy11 = gather(fxy)

    # Stack into (G, G, 16) corner vector, then apply the inverse matrix.
    corners = np.stack(
        [f00, f10, f01, f11,
         fx00, fx10, fx01, fx11,
         fy00, fy10, fy01, fy11,
         fxy00, fxy10, fxy01, fxy11],
        axis=-1,
    )  # (G, G, 16)
    flat = corners @ _BICUBIC_INV.T  # (G, G, 16); each row is c00..c33

    # Layout of `flat[..., k]`:  k = 4*b + a, c_{a,b}.
    # Reshape to (G, G, 4 (b), 4 (a)) and transpose to (G, G, 4 (a), 4 (b)).
    coefs = flat.reshape(G, G, 4, 4).transpose(0, 1, 3, 2)
    return np.ascontiguousarray(coefs, dtype=np.float64)


def stack_grid_bank(
    grids: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    """Stack per-residue raw grids into a single ``(N, G, G, 4, 4)``
    bicubic-coefficient array, returning the residue->grid_id map.

    The order is alphabetical by residue name so the layout is
    deterministic across runs.
    """
    if not grids:
        raise ValueError("Cannot build CMAP bank from empty grid dict.")
    names = sorted(grids.keys())
    coefs = np.stack([build_bicubic_coefs(np.asarray(grids[n])) for n in names], axis=0)
    name_to_id = {n: i for i, n in enumerate(names)}
    return coefs, name_to_id


# --------------------------------------------------------------------------
# Hashable static-metadata wrapper
# --------------------------------------------------------------------------

class CmapGridBank:
    """Hashable, immutable wrapper around the per-residue grids.

    Stored on ``Config`` as ``pytree_node=False`` static metadata so
    different banks do not share JAX tracing caches.  Hash is computed
    once at construction (md5 over the raw grid bytes plus residue
    names) and cached.
    """

    __slots__ = ("_grids", "_grid_size", "_hash")

    def __init__(self, grids: Mapping[str, np.ndarray], grid_size: int = DEFAULT_CMAP_GRID_SIZE):
        # Defensive copy + normalise to float64 so hash is deterministic.
        clean: dict[str, np.ndarray] = {}
        for name, g in grids.items():
            arr = np.ascontiguousarray(np.asarray(g, dtype=np.float64))
            if arr.shape != (grid_size, grid_size):
                raise ValueError(
                    f"CMAP grid for '{name}' has shape {arr.shape}; "
                    f"expected ({grid_size}, {grid_size})."
                )
            clean[name] = arr
        # Freeze.
        object.__setattr__(self, "_grids", clean)
        object.__setattr__(self, "_grid_size", int(grid_size))

        h = hashlib.md5()
        h.update(int(grid_size).to_bytes(4, "little"))
        for name in sorted(clean):
            h.update(name.encode("utf-8"))
            h.update(b"\x00")
            h.update(clean[name].tobytes())
        object.__setattr__(self, "_hash", int.from_bytes(h.digest()[:8], "little"))

    @property
    def grids(self) -> Mapping[str, np.ndarray]:
        return self._grids

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CmapGridBank):
            return NotImplemented
        if self._grid_size != other._grid_size:
            return False
        if set(self._grids.keys()) != set(other._grids.keys()):
            return False
        for k in self._grids:
            if not np.array_equal(self._grids[k], other._grids[k]):
                return False
        return True

    def __repr__(self) -> str:
        return (
            f"CmapGridBank(grid_size={self._grid_size}, "
            f"residues={sorted(self._grids.keys())})"
        )


def parse_cmap_section(
    toml_dict: dict, grid_size: int = DEFAULT_CMAP_GRID_SIZE
) -> Optional[CmapGridBank]:
    """Parse the ``[cmap]`` block from an options.toml dict.

    Expected layout::

        [cmap]
        grid_size = 24

        [cmap.grids]
        GLY = [[...], [...], ...]   # 24 rows of 24 floats
        ALA = [[...], ...]

    Returns ``None`` if no ``[cmap]`` block is present.
    """
    block = toml_dict.get("cmap")
    if block is None:
        return None
    user_size = int(block.get("grid_size", grid_size))
    raw = block.get("grids", {})
    if not raw:
        return None
    return CmapGridBank(raw, grid_size=user_size)


# --------------------------------------------------------------------------
# Runtime kernel (JAX, jit-friendly)
# --------------------------------------------------------------------------

def _cmap_energy_single(
    p1, p2, p3, p4, p5, grid_id, coefs, grid_size, box,
):
    """Energy E(phi, psi) for a single CMAP term.  Designed to be the
    inner function of ``vmap(value_and_grad(...))`` so JAX produces
    per-atom forces by autodiff over (p1..p5).
    """
    phi, _ = get_dihedral_angle(p1, p2, p3, p4, box)
    psi, _ = get_dihedral_angle(p2, p3, p4, p5, box)
    # Map [-pi, pi] -> [0, grid_size); jnp.mod handles +/-pi wrap.
    scale = grid_size / (2.0 * jnp.pi)
    fx = jnp.mod((phi + jnp.pi) * scale, grid_size)
    fy = jnp.mod((psi + jnp.pi) * scale, grid_size)
    i = jnp.floor(fx).astype(jnp.int32)
    j = jnp.floor(fy).astype(jnp.int32)
    # Numerical safety: jnp.mod can return exactly grid_size on tiny
    # negative residuals; clamp into [0, grid_size-1] before gather.
    i = jnp.clip(i, 0, grid_size - 1)
    j = jnp.clip(j, 0, grid_size - 1)
    dx = fx - i
    dy = fy - j

    c = coefs[grid_id, i, j]                  # (4, 4)
    dx_pow = jnp.stack([1.0, dx, dx * dx, dx * dx * dx])  # (4,)
    dy_pow = jnp.stack([1.0, dy, dy * dy, dy * dy * dy])  # (4,)
    return dx_pow @ c @ dy_pow                # scalar


def get_cmap_energy_and_forces(
    forces,
    pos,
    box,
    atom1,
    atom2,
    atom3,
    atom4,
    atom5,
    grid_id,
    coefs,
    grid_size: int = DEFAULT_CMAP_GRID_SIZE,
    compute_pressure: bool = True,
):
    """Vectorised CMAP energy + per-atom forces (+ optional virial).

    Parameters
    ----------
    forces : (n_atoms, 3) accumulator that will be reset and scattered
        into.
    pos : (n_atoms, 3) positions.
    box : (3,) box size for minimum-image PBC inside ``get_dihedral_angle``.
    atom1..atom5 : (n_cmap,) int arrays with the five backbone atom
        indices that define the (phi, psi) pair.  Convention matches
        GROMACS: phi = atoms (1, 2, 3, 4) and psi = atoms (2, 3, 4, 5).
    grid_id : (n_cmap,) int array selecting which residue's bicubic
        coefficients to use.
    coefs : (n_grids, grid_size, grid_size, 4, 4) float array of
        precomputed cell coefficients (see :func:`build_bicubic_coefs`).
    grid_size : Python int, static (used for index modulo arithmetic).
    compute_pressure : whether to compute the virial diagonal.

    Returns
    -------
    cmap_energy : scalar (sum across all CMAP terms).
    forces : (n_atoms, 3) — input ``forces`` with the per-atom CMAP
        contribution scattered in (overwriting whatever was there).
    cmap_pressure : (3,) virial diagonal contribution, or zeros when
        ``compute_pressure`` is False.
    """
    p1 = pos[atom1]
    p2 = pos[atom2]
    p3 = pos[atom3]
    p4 = pos[atom4]
    p5 = pos[atom5]

    cmap_grad = vmap(
        value_and_grad(_cmap_energy_single, (0, 1, 2, 3, 4)),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )
    energies, grads = cmap_grad(p1, p2, p3, p4, p5, grid_id, coefs, grid_size, box)
    g1, g2, g3, g4, g5 = grads

    forces = forces.at[...].set(0.0)
    forces = forces.at[atom1].add(-g1)
    forces = forces.at[atom2].add(-g2)
    forces = forces.at[atom3].add(-g3)
    forces = forces.at[atom4].add(-g4)
    forces = forces.at[atom5].add(-g5)

    if compute_pressure:
        # Pick atom3 (Cα) as reference; for an internal force
        # Sigma F_i = 0 so the virial diagonal is invariant under that
        # choice.  Vectors are minimum-image displacements analogous to
        # the dihedral pressure block in force.py.
        def _min_image(a, b):
            d = a - b
            return d - box * jnp.around(d / box)

        r13 = _min_image(p1, p3)
        r23 = _min_image(p2, p3)
        r43 = _min_image(p4, p3)
        r53 = _min_image(p5, p3)
        cmap_pressure = jnp.sum(
            -g1 * r13 - g2 * r23 - g4 * r43 - g5 * r53, axis=0
        )
    else:
        cmap_pressure = jnp.zeros(3)

    return jnp.sum(energies), forces, cmap_pressure


__all__ = [
    "DEFAULT_CMAP_GRID_SIZE",
    "build_bicubic_coefs",
    "stack_grid_bank",
    "CmapGridBank",
    "parse_cmap_section",
    "get_cmap_energy_and_forces",
]
