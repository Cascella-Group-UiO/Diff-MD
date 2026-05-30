import warnings

import jax
from jax import jit, Array, lax
from jax import numpy as jnp
import numpy as np
from typing import Tuple


INDEX_DTYPE = jnp.int32


def resolve_verlet_radii(rc: float, rlj: float, rv: float, skin: float):
    """Resolve force cutoff, Verlet radius, and jax-md buffer.

    Diff-MD's ``rv`` is the neighbor-list search radius used by the cell
    backend.  jax-md expresses the same search radius as
    ``r_cutoff + dr_threshold``.  Return the common force cutoff, the effective
    Verlet radius, and the jax-md ``dr_threshold`` that reproduces it.
    """
    force_cutoff = max(float(rc), float(rlj))
    verlet_radius = float(rv)
    skin = float(skin)
    min_verlet_radius = force_cutoff + skin

    if verlet_radius < min_verlet_radius:
        warnings.warn(
            f"rv ({verlet_radius}) < max(rc, rlj) + skin ({min_verlet_radius}). "
            f"Increasing rv to {min_verlet_radius} for Verlet-list correctness.",
            RuntimeWarning,
            stacklevel=2,
        )
        verlet_radius = min_verlet_radius

    jaxmd_skin = max(0.0, verlet_radius - force_cutoff)
    return force_cutoff, verlet_radius, jaxmd_skin


def _ensure_index_dtype(idx: Array) -> Array:
    """Ensure JAX indices use an integer dtype accepted by gather/take."""
    if jnp.issubdtype(idx.dtype, jnp.integer):
        return idx
    return idx.astype(INDEX_DTYPE)


@jit
def nlist(positions, box_size, r_cut, i, j):
    """Brute-force O(N²) neighbor search.  DO NOT USE for N > ~2000.

    This function allocates an (N, N, 3) distance matrix, which for
    N = 15 000 atoms in f64 is ~50 GB.  It exists only as a fallback
    for tiny test systems and is called inside ``maybe_rebuild_verlet_list_jax``
    which wraps it in ``lax.cond`` — meaning XLA reserves the full N² buffer
    even when the condition is false.  For production use, set
    ``nlist_method = "jaxmd"`` in rhe TOML to use the O(N) cell-list path.
    """
    positions_i = jnp.expand_dims(positions, axis=1)

    r_vec = positions_i - positions
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=2)

    # mask = jnp.less(r, r_cut)
    mask = jnp.triu(r < r_cut, k=1)

    i_idx, j_idx = jnp.where(mask, size=i.shape[0], fill_value=-1)

    i = i.at[...].set(i_idx.astype(INDEX_DTYPE))
    j = j.at[...].set(j_idx.astype(INDEX_DTYPE))

    n_pairs = jnp.sum(mask)
    return i, j, n_pairs


# ---------------------------------------------------------------------------
# Verlet list with skin: displacement-based conditional rebuild
# ---------------------------------------------------------------------------

def _max_displacement_sq_pbc(positions: Array, ref_positions: Array, box_size: Array) -> Array:
    """Maximum squared single-particle displacement (PBC-aware) since last rebuild.

    Returns the *squared* displacement to avoid N square-root operations;
    callers should compare against a squared threshold.
    """
    dr = positions - ref_positions
    dr = dr - box_size * jnp.around(dr / box_size)
    return jnp.max(jnp.sum(dr * dr, axis=1))


@jit
def maybe_rebuild_verlet_list_jax(
    step: int,
    ns_nlist: int,
    positions: Array,
    ref_positions: Array,
    box_size: Array,
    r_cut: float,
    skin: float,
    neigh_i: Array,
    neigh_j: Array,
    excluded_i: Array,
    excluded_j: Array,
    apply_exclusions: bool,
):
    """Verlet-list rebuild with skin-based displacement criterion.

    Every ``ns_nlist`` steps the maximum single-particle displacement since the
    last rebuild is computed.  The list is rebuilt only when the worst-case
    drift exceeds ``skin / 2`` (two approaching particles could each drift by
    that amount).  When ``skin == 0`` the list is rebuilt unconditionally.

    Returns ``(neigh_i, neigh_j, overflow, new_ref_positions)``.
    ``new_ref_positions`` equals ``positions`` after a rebuild, or the
    unchanged ``ref_positions`` otherwise.
    """

    should_check = jnp.equal(jnp.mod(step, ns_nlist), 0)

    def _check_and_maybe_rebuild(_):
        max_disp_sq = _max_displacement_sq_pbc(positions, ref_positions, box_size)
        # Compare squared displecement against (skin/2)² to avoid sqrt.
        # When skin == 0 ,must rebuild unconditionally (backward compat),
        # so use >= to catch the max_disp_sq == 0 case AND add an explicit
        # zero-skin guard.
        threshold_sq = (skin * 0.5) ** 2
        need_rebuild = (max_disp_sq >= threshold_sq) | jnp.equal(skin, 0.0)

        def _rebuild(_):
            new_i, new_j, n_pairs = nlist(
                positions, box_size, r_cut, neigh_i, neigh_j
            )
            overflow = n_pairs > neigh_i.shape[0]
            new_i, new_j = lax.cond(
                apply_exclusions,
                lambda pair: exclude_bonded_neighbors(
                    pair[0], pair[1], excluded_i, excluded_j
                ),
                lambda pair: pair,
                operand=(new_i, new_j),
            )
            return new_i, new_j, overflow, positions  # update ref

        def _keep(_):
            return neigh_i, neigh_j, jnp.array(False), ref_positions

        return lax.cond(need_rebuild, _rebuild, _keep, operand=None)

    def _skip(_):
        return neigh_i, neigh_j, jnp.array(False), ref_positions

    return lax.cond(should_check, _check_and_maybe_rebuild, _skip, operand=None)


# ---------------------------------------------------------------------------
# Cell-list accelerated neighbor search (NumPy — used for initial build)
# ---------------------------------------------------------------------------

def estimate_initial_capacity(
    n_particles: int,
    box_size,
    rv: float,
    multiplier: float = 1.25,
    min_buffer: int = 256,
) -> int:
    """Estimate a safe initial capacity for the cell neighbor list.

    Size is derived from the expected pair count of a uniformly distributed
    system (half-shell volume * density * N / 2), scaled by ``multiplier`` and
    padded by ``min_buffer`` to cover low-density / small-N edge cases.  Used
    by both ``mdrun.py`` and ``simulate.py`` so the two drivers agree on the
    initial allocation and both react to ``config.nlist_capacity_multiplier``.
    """
    box = np.asarray(box_size, dtype=np.float64)
    volume = float(np.prod(box))
    dens = float(n_particles) / volume
    raw = 0.5 * float(n_particles) * (4.0 * np.pi * float(rv) ** 3 / 3.0) * dens
    return int(np.ceil(raw * float(multiplier))) + int(min_buffer)


def build_neighbor_list_cell(
    positions_np,
    box_size_np,
    r_cut: float,
    initial_capacity: int,
):
    """Build a neighbor list using a cell-list algorithm (NumPy).

    This avoids the O(N²) memory of the brute-force approach and is used for
    the initial (non-JIT) neighbor list construction.  The result is returned
    as padded JAX arrays identical in format to :func:`build_neighbor_list_cell`.

    Returns ``(neigh_i, neigh_j, capacity)``.
    """
    pos = np.asarray(positions_np, dtype=np.float64)
    box = np.asarray(box_size_np, dtype=np.float64)
    n_atoms = pos.shape[0]

    # Guard against NaN/Inf positions (simulation blow-up)
    if np.any(~np.isfinite(pos)):
        n_bad = int(np.sum(np.any(~np.isfinite(pos), axis=1)))
        raise ValueError(
            f"build_neighbor_list_cell: {n_bad}/{n_atoms} atoms have "
            f"NaN/Inf coordinates — the simulation has likely diverged. "
            f"Check timestep, thermostat, or initial configuration."
        )

    # --- cell grid ---
    n_cells_xyz = np.maximum(np.floor(box / r_cut).astype(np.int64), 3)
    cell_size = box / n_cells_xyz
    total_cells = int(np.prod(n_cells_xyz))

    # Assign atoms to cells
    frac = pos / cell_size
    cell_coords = np.floor(frac).astype(np.int64) % n_cells_xyz
    cell_id = (
        cell_coords[:, 0] * n_cells_xyz[1] * n_cells_xyz[2]
        + cell_coords[:, 1] * n_cells_xyz[2]
        + cell_coords[:, 2]
    )

    # Build cell -> atom mapping using sorted indices for cache-friendly access
    sort_idx = np.argsort(cell_id)
    sorted_cell_id = cell_id[sort_idx]
    # cell_start[c] = first index in sort_idx belonging to cell c
    # cell_count[c] = number of atoms in cell c
    cell_start = np.zeros(total_cells + 1, dtype=np.int64)
    cell_count = np.zeros(total_cells, dtype=np.int64)
    for i, cid in enumerate(sorted_cell_id):
        cell_count[cid] += 1
    np.cumsum(cell_count, out=cell_start[1:])

    # 13 unique neighbor offsets (half-shell to avoid double counting)
    half_shell = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                idx = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                if idx > 13:  # only offsets with flat index > center (13)
                    half_shell.append((dx, dy, dz))

    r_cut_sq = r_cut * r_cut
    pairs_i_list: list[np.ndarray] = []
    pairs_j_list: list[np.ndarray] = []

    ncx, ncy, ncz = int(n_cells_xyz[0]), int(n_cells_xyz[1]), int(n_cells_xyz[2])

    def _cell_atoms(cid):
        """Return atom indices belonging to cell *cid*."""
        s = cell_start[cid]
        return sort_idx[s : s + cell_count[cid]]

    def _add_pairs_self(atoms_a):
        """Vectorized self-cell pair search (upper-triangle, i < j)."""
        na = len(atoms_a)
        if na < 2:
            return
        pa = pos[atoms_a]  # (na, 3)
        # Upper-triangle indices
        ii, jj = np.triu_indices(na, k=1)
        dr = pa[ii] - pa[jj]
        dr -= box * np.round(dr / box)
        dist_sq = np.einsum('ij,ij->i', dr, dr)
        mask = dist_sq < r_cut_sq
        if mask.any():
            pairs_i_list.append(atoms_a[ii[mask]])
            pairs_j_list.append(atoms_a[jj[mask]])

    def _add_pairs_cross(atoms_a, atoms_b):
        """Vectorized cross-cell pair search with canonical (lo, hi) ordering."""
        na, nb = len(atoms_a), len(atoms_b)
        if na == 0 or nb == 0:
            return
        pa = pos[atoms_a]  # (na, 3)
        pb = pos[atoms_b]  # (nb, 3)
        # All-pairs via broadcasting: (na, nb, 3)
        dr = pa[:, None, :] - pb[None, :, :]
        dr -= box * np.round(dr / box)
        dist_sq = np.einsum('ijk,ijk->ij', dr, dr)  # (na, nb)
        mi, mj = np.where(dist_sq < r_cut_sq)
        if len(mi) == 0:
            return
        ai = atoms_a[mi]
        aj = atoms_b[mj]
        # Canonical ordering: lo < hi
        lo = np.minimum(ai, aj)
        hi = np.maximum(ai, aj)
        pairs_i_list.append(lo)
        pairs_j_list.append(hi)

    for cx in range(ncx):
        for cy in range(ncy):
            for cz in range(ncz):
                cid = cx * ncy * ncz + cy * ncz + cz
                atoms_a = _cell_atoms(cid)

                # Self-cell pairs (i < j)
                _add_pairs_self(atoms_a)

                # Half-shell neighbor cells
                for dx, dy, dz in half_shell:
                    nx = (cx + dx) % ncx
                    ny = (cy + dy) % ncy
                    nz = (cz + dz) % ncz
                    nid = nx * ncy * ncz + ny * ncz + nz
                    atoms_b = _cell_atoms(nid)
                    _add_pairs_cross(atoms_a, atoms_b)

    if pairs_i_list:
        all_i = np.concatenate(pairs_i_list).astype(np.int32)
        all_j = np.concatenate(pairs_j_list).astype(np.int32)
        n_pairs = len(all_i)
    else:
        all_i = np.empty(0, dtype=np.int32)
        all_j = np.empty(0, dtype=np.int32)
        n_pairs = 0
    capacity = max(n_pairs, initial_capacity)

    neigh_i = np.full(capacity, -1, dtype=np.int32)
    neigh_j = np.full(capacity, -1, dtype=np.int32)
    neigh_i[:n_pairs] = all_i
    neigh_j[:n_pairs] = all_j

    return jnp.asarray(neigh_i), jnp.asarray(neigh_j), capacity


@jit
def apply_cutoff(
    r: Array,
    r_norm: Array,
    sigma: Array,
    epsilon: Array, 
    neigh_i: Array,
    neigh_j: Array,
    rc: float
) -> Tuple[float, Array]:

    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)

    # Use jnp.less to create a boolean mask
    mask = jnp.less(r_norm, rc)

    # Account for padding in nlists
    mask = jnp.where(neigh_i==-1, False, mask)

    # Use jnp.where with the mask to get indices
    cut_indx = jnp.where(mask, size=r_norm.shape[0], fill_value=-1)[0]

    # Apply the mask using jnp.take
    r_values = jnp.take(r, cut_indx, axis=0)
    r_values = jnp.where(jnp.reshape(cut_indx, (-1, 1))==-1, 1, r_values)

    sigma_values = jnp.take(sigma, cut_indx)
    epsilon_values = jnp.take(epsilon, cut_indx)
    # Set epsilon to 0 for pair interactions outside the cutoff
    epsilon_values = jnp.where(cut_indx == -1, 0, epsilon_values)

    # Update neighbor list
    neigh_i_mod = jnp.take(neigh_i, cut_indx)
    neigh_j_mod = jnp.take(neigh_j, cut_indx)

    return r_values, sigma_values, epsilon_values, neigh_i_mod, neigh_j_mod


@jit
def apply_cutoff_elec(
    r: Array,
    r_norm: Array,
    q_i: Array,
    q_j: Array,
    neigh_i: Array,
    neigh_j: Array,
    rc: float
) -> Tuple[float, Array]:

    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)

    # Use jnp.less to create a boolean mask
    mask = jnp.less(r_norm, rc)
    # Account for padding in nlists
    mask = jnp.where(neigh_i==-1, False, mask)

    # Use jnp.where with the mask to get indices
    cut_indx = jnp.where(mask, size=r_norm.shape[0], fill_value=-1)[0]

    # Apply the mask using jnp.take
    r_values = jnp.take(r, cut_indx, axis=0)
    r_values = jnp.where(jnp.reshape(cut_indx, (-1, 1))==-1, 1, r_values)
    
    r_norm = jnp.take(r_norm, cut_indx)
    q_i = jnp.take(q_i, cut_indx)
    q_j = jnp.take(q_j, cut_indx)

    # Set q to 0 for pair interactions outside the cutoff
    q_i = jnp.where(cut_indx == -1, 0, q_i)
    q_j = jnp.where(cut_indx == -1, 0, q_j)

    # Update neighbor list
    neigh_i_mod = jnp.take(neigh_i, cut_indx)
    neigh_j_mod = jnp.take(neigh_j, cut_indx)

    return r_values, r_norm, q_i, q_j, neigh_i_mod, neigh_j_mod

@jit
def apply_nlist(
    neigh_i: Array,
    neigh_j: Array,
    positions: Array,
    box_size: Array,
    sigma: Array,
    epsilon: Array,
    types: Array    
):
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)

    # For calculation of pair potential energy
    i = jnp.take(positions, neigh_i, axis=0)
    j = jnp.take(positions, neigh_j, axis=0)
    r_vec = i - j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)
    q_i = None
    q_j = None

    s_ij = sigma[types[neigh_i[:]], types[neigh_j[:]]]
    e_ij = epsilon[types[neigh_i[:]], types[neigh_j[:]]]

    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij

@jit
def apply_nlist_elec(
    neigh_i: Array,
    neigh_j: Array,
    positions: Array,
    charges: Array,
    box_size: Array,
    sigma: Array,
    epsilon: Array,
    types: Array    
):
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)

    # For calculation of pair potential energy
    i = jnp.take(positions, neigh_i, axis=0)
    j = jnp.take(positions, neigh_j, axis=0)
    r_vec = i - j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)
    # Flatten to 1D pair charges to avoid unintended broadcasting in
    # pairwise electrostatic energy expressions.
    q_i = jnp.ravel(charges[neigh_i])
    q_j = jnp.ravel(charges[neigh_j])

    s_ij = sigma[types[neigh_i[:]], types[neigh_j[:]]]
    e_ij = epsilon[types[neigh_i[:]], types[neigh_j[:]]]

    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij


# ---------------------------------------------------------------------------
# Precomputed-distance variants  (optimisation)
# ---------------------------------------------------------------------------

@jit
def compute_pair_distances(
    idx_i: Array,
    idx_j: Array,
    positions: Array,
    box_size: Array,
):
    """Compute PBC minimum-image distance vectors and norms for atom pairs.

    Returns ``(r_vec, r)`` with shapes ``(n_pairs, 3)`` and ``(n_pairs,)``.
    """
    idx_i = _ensure_index_dtype(idx_i)
    idx_j = _ensure_index_dtype(idx_j)
    pos_i = jnp.take(positions, idx_i, axis=0)
    pos_j = jnp.take(positions, idx_j, axis=0)
    r_vec = pos_i - pos_j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)
    return r_vec, r


@jit
def apply_nlist_precomputed(
    neigh_i: Array,
    neigh_j: Array,
    r_vec: Array,
    r: Array,
    sigma: Array,
    epsilon: Array,
    types: Array,
):
    """Build pair-parameter tuple from *pre-computed* distances (no charges)."""
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)
    s_ij = sigma[types[neigh_i], types[neigh_j]]
    e_ij = epsilon[types[neigh_i], types[neigh_j]]
    return r_vec, r, neigh_i, neigh_j, None, None, s_ij, e_ij


@jit
def apply_nlist_elec_precomputed(
    neigh_i: Array,
    neigh_j: Array,
    r_vec: Array,
    r: Array,
    charges: Array,
    sigma: Array,
    epsilon: Array,
    types: Array,
):
    """Build pair-parameter tuple from *pre-computed* distances (with charges)."""
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)
    q_i = jnp.ravel(charges[neigh_i])
    q_j = jnp.ravel(charges[neigh_j])
    s_ij = sigma[types[neigh_i], types[neigh_j]]
    e_ij = epsilon[types[neigh_i], types[neigh_j]]
    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij


@jit
def apply_nlist_general(
    neigh_i: Array,
    neigh_j: Array,
    positions: Array,
    charges: Array,
    box_size: Array,
    sigma: Array,
    epsilon: Array,
    types: Array,
):
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)

    # For calculation of pair potential energy
    i = jnp.take(positions, neigh_i, axis=0)
    j = jnp.take(positions, neigh_j, axis=0)
    r_vec = i - j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)

    s_ij = sigma[types[neigh_i[:]], types[neigh_j[:]]]
    e_ij = epsilon[types[neigh_i[:]], types[neigh_j[:]]]

    q_i, q_j = lax.cond(
        charges is not None,
        lambda: (
            jnp.ravel(charges[neigh_i]),        # true branch
            jnp.ravel(charges[neigh_j])),
        lambda: (
            jnp.zeros((neigh_i.shape[0],)),     # false branch
            jnp.zeros((neigh_i.shape[0],))),
    )

    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij


@jit
def exclude_bonded_neighbors(neigh_i, neigh_j, bonded_i, bonded_j):
    """Exclude bonded (excluded) pairs from the neighbor list.

    Memory cost is O(n_neighbors + n_excluded) rather than the
    O(n_neighbors * n_excluded) of a full broadcast comparison, avoiding OOM
    on large atomistic systems.

    Membership is resolved by ``apply_exclusions`` with a runtime dispatch on
    the largest atom index: a fast int32 Szudzik-key path for indices within
    the int32-safe bound (every normal system), and an overflow-free
    ``(lo, hi)`` lexsort path for larger systems where the squared key would
    overflow int32 and silently collide distinct pairs.
    """
    neigh_i = _ensure_index_dtype(neigh_i)
    neigh_j = _ensure_index_dtype(neigh_j)
    bonded_i = _ensure_index_dtype(bonded_i)
    bonded_j = _ensure_index_dtype(bonded_j)

    def no_exclusions(_):
        return neigh_i, neigh_j

    def apply_exclusions(_):
        # Decide "is this neighbor pair bonded/excluded?" for every entry.
        #
        # Two implementations, dispatched at runtime by the largest atom index:
        #
        #   _fast_mask  — the historical int32 Szudzik key + searchsorted.  A
        #     Szudzik key squares the atom index, so it overflows int32 once an
        #     index exceeds ~46339 and then silently collides distinct pairs
        #     (wrong exclusions -> wrong nonbonded forces).  Below that bound it
        #     is exact and is the cheapest option, so every normal-sized system
        #     keeps its original speed.
        #
        #   _safe_mask  — overflow-free pairwise membership: canonicalize each
        #     pair to (lo, hi), lexsort the union of excluded + neighbor pairs,
        #     and propagate an "excluded" flag within each run of identical
        #     pairs via a segmented sum.  No scalar key is ever formed, so it is
        #     exact for any atom count under any precision.  It sorts the full
        #     neighbor list (a few x costlier), so it is used ONLY when an index
        #     actually exceeds the safe bound.
        #
        # lax.cond executes just the taken branch, so the safe path adds no
        # runtime cost to systems that stay within int32.
        #
        # SAFE_MAX = largest A with A*A + 2*A <= 2**31 - 1 (worst-case key is
        # a*a + a + b <= A*A + 2*A for a, b <= A), i.e. no int32 overflow.
        SAFE_MAX = jnp.int32(46339)
        P = neigh_i.shape[0]

        def _fast_mask(_):
            def _pair_key(a, b):
                # Szudzik pairing for non-negative ints (int32; safe here
                # because every index is <= SAFE_MAX on this branch).
                a, b = jnp.int32(a), jnp.int32(b)
                return jnp.where(a >= b, a * a + a + b, b * b + a)

            # Excluded-pair keys, both orderings, plus a -1 sentinel so the
            # array is never empty (gather ops would crash on an empty array).
            excl_keys = jnp.concatenate([
                _pair_key(bonded_i, bonded_j),
                _pair_key(bonded_j, bonded_i),
                jnp.array([-1], dtype=jnp.int32),
            ])
            excl_keys = jnp.sort(excl_keys)
            neigh_keys = _pair_key(neigh_i, neigh_j)
            insert_idx = jnp.searchsorted(excl_keys, neigh_keys)
            in_range = insert_idx < excl_keys.shape[0]
            safe_idx = jnp.clip(insert_idx, 0, excl_keys.shape[0] - 1)
            return in_range & (excl_keys[safe_idx] == neigh_keys)

        def _safe_mask(_):
            E = bonded_i.shape[0]
            # Canonicalize so that (i, j) and (j, i) map to the same pair.
            e_lo = jnp.minimum(bonded_i, bonded_j)
            e_hi = jnp.maximum(bonded_i, bonded_j)
            n_lo = jnp.minimum(neigh_i, neigh_j)
            n_hi = jnp.maximum(neigh_i, neigh_j)

            all_lo = jnp.concatenate([e_lo, n_lo])
            all_hi = jnp.concatenate([e_hi, n_hi])
            # 1 marks an excluded-set member, 0 marks a neighbor-list query.
            is_excl = jnp.concatenate(
                [jnp.ones(E, dtype=jnp.int32), jnp.zeros(P, dtype=jnp.int32)]
            )

            # Lexicographic sort: primary key = lo, secondary key = hi (the
            # last key passed to lexsort is the primary one).  Identical pairs
            # become adjacent regardless of original orientation.
            order = jnp.lexsort((all_hi, all_lo))
            s_lo = all_lo[order]
            s_hi = all_hi[order]
            s_is_excl = is_excl[order]

            # Dense group id: a new group starts whenever (lo, hi) changes.
            same_as_prev = jnp.concatenate([
                jnp.array([False]),
                (s_lo[1:] == s_lo[:-1]) & (s_hi[1:] == s_hi[:-1]),
            ])
            group_id = jnp.cumsum((~same_as_prev).astype(jnp.int32)) - 1  # 0..G-1

            M = E + P
            group_has_excl = (
                jax.ops.segment_sum(s_is_excl, group_id, num_segments=M) > 0
            )
            s_excluded = group_has_excl[group_id]            # per sorted entry

            # Scatter the per-sorted flag back to the original combined order,
            # then slice out the neighbor half (first E entries are excluded).
            excluded_combined = jnp.zeros(M, dtype=bool).at[order].set(s_excluded)
            return excluded_combined[E:]                     # (P,)

        # Largest index across all pair endpoints (initial=-1 keeps the empty
        # bonded-array trace valid; -1 simply selects the fast path).
        max_idx = jnp.maximum(
            jnp.maximum(jnp.max(neigh_i, initial=jnp.int32(-1)),
                        jnp.max(neigh_j, initial=jnp.int32(-1))),
            jnp.maximum(jnp.max(bonded_i, initial=jnp.int32(-1)),
                        jnp.max(bonded_j, initial=jnp.int32(-1))),
        )
        is_excluded = lax.cond(
            max_idx <= SAFE_MAX, _fast_mask, _safe_mask, operand=None
        )

        # Also mask out padding entries (neigh_i == -1)
        mask = (~is_excluded) & (neigh_i != -1)

        indx = jnp.where(mask, size=mask.shape[0], fill_value=-1)[0].astype(INDEX_DTYPE)
        filtered_neigh_i = jnp.take(neigh_i, indx)
        filtered_neigh_j = jnp.take(neigh_j, indx)
        # `jnp.take(..., -1)` selects the last element, so convert fill slots
        # back to sentinel -1 explicitly to avoid duplicating a valid pair.
        filtered_neigh_i = jnp.where(indx == -1, INDEX_DTYPE(-1), filtered_neigh_i)
        filtered_neigh_j = jnp.where(indx == -1, INDEX_DTYPE(-1), filtered_neigh_j)
        return filtered_neigh_i, filtered_neigh_j

    return lax.cond(bonded_i.shape[0] > 0, apply_exclusions, no_exclusions, operand=None)


# ---------------------------------------------------------------------------
# jax-md neighbor list wrapper (O(N) cell-list hashing)
# ---------------------------------------------------------------------------

def _jaxmd_idx_to_pairs(idx, n_atoms):
    """Convert jax-md OrderedSparse idx ``(2, max_pairs)`` to Diff-MD
    ``(neigh_i, neigh_j)`` arrays with ``-1`` padding."""
    neigh_i = jnp.where(idx[0] < n_atoms, idx[0], INDEX_DTYPE(-1))
    neigh_j = jnp.where(idx[1] < n_atoms, idx[1], INDEX_DTYPE(-1))
    return neigh_i.astype(INDEX_DTYPE), neigh_j.astype(INDEX_DTYPE)


def init_jaxmd_neighbor_list(positions, box_size, r_cutoff_phys, skin,
                              capacity_multiplier=1.25):
    """Create a jax-md ``OrderedSparse`` neighbor list.

    Uses ``space.periodic_general`` with ``fractional_coordinates=True``
    so that the box size can be updated dynamically during NPT
    simulations via ``nbrs.update(pos_frac, box=new_box)``.

    Parameters
    ----------
    positions : array (N, 3)
        Initial particle positions (Cartesian, nm).
    box_size : array (3,)
        Orthorhombic box dimensions.
    r_cutoff_phys : float
        Physical force cutoff ``max(rc, rlj)``.  jax-md will search
        ``r_cutoff_phys + skin`` internally.
    skin : float
        Verlet skin (``dr_threshold`` in jax-md).  Rebuild fires when
        max single-particle displacement exceeds ``skin / 2``.
    capacity_multiplier : float
        Buffer factor for the pair-list capacity (default 1.25).

    Returns
    -------
    neighbor_fn : callable
        Factory needed for re-allocation on overflow.
    nbrs : jax_md.partition.NeighborList
        Initial neighbor list (JAX pytree).
    neigh_i, neigh_j : Array
        Pair index arrays with ``-1`` padding.
    """
    from jax_md import space, partition
    import logging
    _log = logging.getLogger(__name__)
    _log.debug(
        f"init_jaxmd_neighbor_list: N={positions.shape[0]}, "
        f"box={box_size}, rc={r_cutoff_phys}, skin={skin}, "
        f"box.shape={getattr(box_size, 'shape', 'scalar')}, "
        f"box.dtype={getattr(box_size, 'dtype', type(box_size).__name__)}"
    )

    displacement_fn, _ = space.periodic_general(
        box_size, fractional_coordinates=True,
    )
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=float(r_cutoff_phys),
        dr_threshold=float(skin),
        capacity_multiplier=capacity_multiplier,
        format=partition.NeighborListFormat.OrderedSparse,
        mask_self=True,
        fractional_coordinates=True,
    )

    pos_frac = positions / box_size

    with warnings.catch_warnings():
        # jax-md internally scatters int64 into int32 under x64 mode
        warnings.filterwarnings("ignore", message="scatter inputs have incompatible types",
                                category=FutureWarning)
        try:
            nbrs = neighbor_fn.allocate(pos_frac, box=box_size)
        except OverflowError:
            # jax-md sparse neighbor list index exceeds int32 (>2^31 pairs).
            # Retry with x64 temporarily enabled so that Python ints > 2^31
            # can be cast to int64 inside jnp.where.
            _prev_x64 = jax.config.jax_enable_x64
            try:
                jax.config.update("jax_enable_x64", True)
                nbrs = neighbor_fn.allocate(pos_frac, box=box_size)
            except Exception:
                raise OverflowError(
                    f"jax-md OrderedSparse neighbor list allocation exceeds "
                    f"int32 range ({positions.shape[0]} atoms, "
                    f"box={box_size}, rc={r_cutoff_phys}, skin={skin}, "
                    f"capacity_multiplier={capacity_multiplier}).  "
                    f"Fix: set  nlist_method = \"cell\"  in options.toml, or "
                    f"export JAX_ENABLE_X64=True before running."
                )
            finally:
                jax.config.update("jax_enable_x64", _prev_x64)
    n_atoms = positions.shape[0]
    neigh_i, neigh_j = _jaxmd_idx_to_pairs(nbrs.idx, n_atoms)

    return neighbor_fn, nbrs, neigh_i, neigh_j


def maybe_update_jaxmd_nlist(step, ns_nlist, positions, nbrs, n_atoms,
                              neigh_i, neigh_j,
                              excluded_i, excluded_j, apply_exclusions,
                              box_size):
    """Gated jax-md neighbor-list update (compatible with ``lax.scan``).

    Every *ns_nlist* steps, calls ``nbrs.update(pos_frac, box=box_size)``
    which internally checks particle displacement against the skin
    threshold and rebuilds only when necessary.  Positions are converted
    to fractional coordinates before the update since the jax-md list
    uses ``fractional_coordinates=True`` for NPT compatibility.

    Returns the same 4-tuple as :func:`maybe_rebuild_verlet_list_jax`:
    ``(neigh_i, neigh_j, overflow, nlist_state)`` where *nlist_state* is
    the updated ``NeighborList`` object.
    """

    should_check = jnp.equal(jnp.mod(step, ns_nlist), 0)

    def _update(_):
        pos_frac = positions / box_size
        new_nbrs = nbrs.update(pos_frac, box=box_size)
        ni, nj = _jaxmd_idx_to_pairs(new_nbrs.idx, n_atoms)
        ni, nj = lax.cond(
            apply_exclusions,
            lambda pair: exclude_bonded_neighbors(
                pair[0], pair[1], excluded_i, excluded_j
            ),
            lambda pair: pair,
            operand=(ni, nj),
        )
        overflow = jnp.bool_(new_nbrs.did_buffer_overflow)
        return ni, nj, overflow, new_nbrs

    def _skip(_):
        return neigh_i, neigh_j, jnp.bool_(False), nbrs

    return lax.cond(should_check, _update, _skip, operand=None)
