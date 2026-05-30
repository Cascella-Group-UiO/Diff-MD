from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
from jax import Array, jit
from mpi4py import MPI

from .config import Config, get_type_to_LJ
from .models import GeneralModel
from .simulate import simulator


if not hasattr(MPI, "SUM"):
    MPI.SUM = None


def _single_rank_allreduce_fallback(value, op=None, comm=None):
    del op
    size = 1
    if comm is not None and hasattr(comm, "Get_size"):
        size = comm.Get_size()
    if size != 1:
        raise RuntimeError(
            "mpi4jax.allreduce is unavailable, but a multi-rank reduction was requested."
        )
    return value


if not hasattr(mpi4jax, "allreduce"):
    mpi4jax.allreduce = _single_rank_allreduce_fallback


def _local_diag_snapshot(**kwargs):
    """Snapshot per-rank diagnostic values BEFORE mpi4jax.allreduce.

    Stored under the reserved key '_local' inside the diag dict; the
    optimize loop's step() peels it off in multidir mode for per-rank
    logging and never feeds it back into the loss.  Values must be
    jnp arrays/scalars so this works inside value_and_grad's aux
    tuple.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _resolve_kde_comm(comm, replica_comm):
    """Pick which communicator KDE / per-system observables aggregate over.

    Returns ``(kde_comm, kde_size)``.  In heterogeneous multidir mode the
    caller passes ``replica_comm`` = per-replica-group sub-communicator,
    so KDEs from distinct logical systems are NOT mixed across the world.
    When ``replica_comm`` is None (single-dir, homogeneous multidir,
    legacy callers) the WORLD ``comm`` is used and behavior is bit-
    identical to pre-2026-05-26.
    """
    chosen = replica_comm if replica_comm is not None else comm
    return chosen, chosen.Get_size()


def get_LJ_param(
    model: GeneralModel, config: Config, types: Array
) -> Tuple[Array, Array, dict[int, Array], Array]:
    assert model.LJ_param is not None, "GeneralModel.chi should not be 'None' here."

    epsl_constraint = {}
    sgm = config.sgm_table
    epsl = jnp.zeros((config.n_types, config.n_types))

    if getattr(model, "lj_mode", "pair") == "type":
        assert model.lj_sigma_ref is not None
        assert model.lj_epsilon_ref is not None
        assert model.lj_sigma_idx is not None
        assert model.lj_epsilon_idx is not None

        sigma_type = jnp.asarray(model.lj_sigma_ref)
        epsilon_type = jnp.asarray(model.lj_epsilon_ref)
        sigma_idx = jnp.asarray(model.lj_sigma_idx, dtype=jnp.int32)
        epsilon_idx = jnp.asarray(model.lj_epsilon_idx, dtype=jnp.int32)

        n_sigma = int(model.n_sigma_train)
        n_epsilon = int(model.n_epsilon_train)

        if n_sigma > 0:
            sigma_type = sigma_type.at[sigma_idx].set(model.LJ_param[:n_sigma])
        if n_epsilon > 0:
            eps_values = model.LJ_param[n_sigma : n_sigma + n_epsilon]
            epsilon_type = epsilon_type.at[epsilon_idx].set(eps_values)

        local_types = jnp.asarray(config.unique_types)
        sigma_local = sigma_type[local_types]
        epsilon_local = epsilon_type[local_types]

        if config.combining_rule == "geometric":
            sgm = jnp.sqrt(jnp.outer(sigma_local, sigma_local))
        else:
            sgm = 0.5 * (jnp.expand_dims(sigma_local, 1) + jnp.expand_dims(sigma_local, 0))

        # Safe sqrt: clamp outer product away from zero before differentiating.
        # grad of sqrt(x) = 1/(2*sqrt(x)) is NaN at x=0 (e.g. HW/HO epsilon=0).
        epsl_outer = jnp.outer(epsilon_local, epsilon_local)
        epsl = jnp.sqrt(jnp.maximum(epsl_outer, 1e-30))

        # Gate constraint dict to rows whose type lies in THIS rank's local
        # config.unique_types.  Without the gate the constraint flows back to
        # rows that this rank's simulation does not touch; after
        # `mpi4jax.allreduce(SUM, WORLD)` those non-local rows pick up a
        # world_size factor.  Gating + the 1/world_size scaling applied
        # to `loss_constraint` inside each loss function below give an
        # effective k_constraint that matches the TOML value across all
        # world sizes (see optimize.py grad_normalizer comment).
        local_set = set(int(t) for t in config.unique_types)
        sigma_idx_np = np.asarray(model.lj_sigma_idx) if model.lj_sigma_idx is not None else np.array([], dtype=int)
        epsilon_idx_np = np.asarray(model.lj_epsilon_idx)
        for ttc, val in model.epsl_constraints.items():
            if ttc < n_sigma:
                t = int(sigma_idx_np[ttc])
            else:
                t = int(epsilon_idx_np[ttc - n_sigma])
            if t in local_set:
                epsl_constraint[ttc] = val

        types = jnp.where(
            jnp.expand_dims(jnp.asarray(config.unique_types), 0)
            == jnp.expand_dims(types, 1)
        )[1]

        return sgm, epsl, epsl_constraint, types

    # ── pair mode ──
    ttlj_full = get_type_to_LJ(model.n_types)
    local = jnp.asarray(config.unique_types)

    # Precompute flat-pair-index → (type_i, type_j) mapping
    flat_to_pair = {}
    for i in range(model.n_types):
        for j in range(i, model.n_types):
            flat_to_pair[int(ttlj_full[i, j])] = (i, j)

    # Map training-space type id → local index in config tables
    local_set = set(int(t) for t in config.unique_types)
    type_to_local = {int(ti): li for li, ti in enumerate(config.unique_types)}

    # Override sigma for trainable pairs
    n_sigma = int(model.n_sigma_train)
    if n_sigma > 0 and model.lj_sigma_idx is not None:
        for k in range(n_sigma):
            flat_idx = int(model.lj_sigma_idx[k])
            ti, tj = flat_to_pair[flat_idx]
            if ti in local_set and tj in local_set:
                li, lj = type_to_local[ti], type_to_local[tj]
                val = model.LJ_param[k]
                sgm = sgm.at[li, lj].set(val)
                sgm = sgm.at[lj, li].set(val)

    # Override epsilon for trainable pairs
    n_epsilon = int(model.n_epsilon_train)
    epsl = config.epsl_table
    if n_epsilon > 0 and model.lj_epsilon_idx is not None:
        for k in range(n_epsilon):
            flat_idx = int(model.lj_epsilon_idx[k])
            ti, tj = flat_to_pair[flat_idx]
            if ti in local_set and tj in local_set:
                li, lj = type_to_local[ti], type_to_local[tj]
                val = model.LJ_param[n_sigma + k]
                epsl = epsl.at[li, lj].set(val)
                epsl = epsl.at[lj, li].set(val)

    # Remap particle types to local indices [0, 1, 6] -> [0, 1, 2]
    types = jnp.where(
        jnp.expand_dims(local, 0) == jnp.expand_dims(types, 1)
    )[1]

    # Parse constraints — gate to rows whose pair (ti, tj) BOTH lie in
    # this rank's local config.unique_types.  Mirrors the sigma/epsilon
    # override branches at lines ~96-117; see the type-mode block above
    # for the rationale.  Combined with the 1/world_size scaling on
    # `loss_constraint` inside each loss, the effective k_constraint
    # matches the TOML value across all world sizes and all ownership
    # patterns.
    n_sigma_train = int(model.n_sigma_train)
    for ttc, val in model.epsl_constraints.items():
        if ttc < n_sigma_train:
            flat_idx = int(model.lj_sigma_idx[ttc])
        else:
            flat_idx = int(model.lj_epsilon_idx[ttc - n_sigma_train])
        ti, tj = flat_to_pair[flat_idx]
        if ti in local_set and tj in local_set:
            epsl_constraint[ttc] = val

    return sgm, epsl, epsl_constraint, types


def _local_boundary_epsilon_params(model: GeneralModel, config: Config) -> Array:
    """Return trainable epsilon rows owned by this rank's system.

    Boundary penalties are regularizers on trainable LJ epsilon parameters,
    not on the full symmetric epsilon table.  That avoids penalizing fixed
    entries and avoids charging off-diagonal pair parameters twice.
    """
    if model.LJ_param is None or model.lj_epsilon_idx is None:
        dtype = jnp.asarray(0.0).dtype if model.LJ_param is None else model.LJ_param.dtype
        return jnp.zeros((0,), dtype=dtype)

    local_set = set(int(t) for t in config.unique_types)
    n_sigma = int(model.n_sigma_train)
    epsilon_idx_np = np.asarray(model.lj_epsilon_idx)
    owned_rows: list[int] = []

    if getattr(model, "lj_mode", "pair") == "type":
        for k, type_id in enumerate(epsilon_idx_np):
            if int(type_id) in local_set:
                owned_rows.append(n_sigma + k)
    else:
        ttlj_full = get_type_to_LJ(model.n_types)
        flat_to_pair = {}
        for i in range(model.n_types):
            for j in range(i, model.n_types):
                flat_to_pair[int(ttlj_full[i, j])] = (i, j)
        for k, flat_idx in enumerate(epsilon_idx_np):
            ti, tj = flat_to_pair[int(flat_idx)]
            if ti in local_set and tj in local_set:
                owned_rows.append(n_sigma + k)

    if not owned_rows:
        return jnp.zeros((0,), dtype=model.LJ_param.dtype)
    return jnp.take(model.LJ_param, jnp.asarray(owned_rows, dtype=jnp.int32))


def _globalize_regularizer(local_loss: Array, comm):
    local_scaled = local_loss / float(comm.Get_size())
    global_loss = mpi4jax.allreduce(local_scaled, op=MPI.SUM, comm=comm)
    return global_loss, local_scaled


def _regularizer_losses(
    model: GeneralModel,
    config: Config,
    comm,
    constraint,
    k_constraint,
    param_constraints,
    boundary_C,
    boundary_S,
    upper_boundary,
    lower_boundary,
):
    zero = jnp.asarray(0.0, dtype=model.LJ_param.dtype)
    loss_constraint = zero
    local_loss_constraint = zero
    if constraint:
        local_raw_constraint = constraint(model.LJ_param, k_constraint, param_constraints)
        loss_constraint, local_loss_constraint = _globalize_regularizer(
            local_raw_constraint, comm
        )

    loss_boundary = zero
    local_loss_boundary = zero
    if upper_boundary is not None or lower_boundary is not None:
        boundary_params = _local_boundary_epsilon_params(model, config)
        local_raw_boundary = boundary_constraint(
            boundary_params, boundary_C, boundary_S, upper_boundary, lower_boundary
        )
        loss_boundary, local_loss_boundary = _globalize_regularizer(
            local_raw_boundary, comm
        )

    return loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary


@jit
def _center(pos: jnp.ndarray, com: float, box: float):
    """Centers positions in the box with respect to the center of mass"""
    pos += 0.5 * box - com
    pos = jnp.where(pos > box, pos - box, pos)
    pos = jnp.where(pos < 0.0, pos + box, pos)
    return pos - 0.5 * box


@jit
def _compute_com(pos: jnp.ndarray, box: float):
    """Compute center of mass along the z-axis"""
    pos_map = 2 * jnp.pi * pos / box
    cos_map = jnp.cos(pos_map)
    sin_map = jnp.sin(pos_map)

    # Using jnp.sum instead of jnp.mean because arctan2 calculates the ratio between the arguments
    theta = jnp.arctan2(-jnp.sum(sin_map), -jnp.sum(cos_map)) + jnp.pi
    com = box * theta / (2 * jnp.pi)
    return com


@jit
def _unwrap_chains_pbc(chains_pos, box):
    """Unwrap chains using sequential minimum-image displacements.

    Each chain is unwrapped by computing the displacement between consecutive
    atoms along the chain backbone, applying minimum-image convention, and
    reconstructing contiguous positions via cumulative summation.

    Parameters
    ----------
    chains_pos : (n_chains, n_atoms_per_chain, 3)
    box : (3,)

    Returns
    -------
    (n_chains, n_atoms_per_chain, 3) unwrapped positions
    """
    dr = jnp.diff(chains_pos, axis=1)                     # (n_chains, n_atoms-1, 3)
    dr = dr - box * jnp.around(dr / box)                   # minimum-image
    cumul = jnp.concatenate(
        [jnp.zeros((chains_pos.shape[0], 1, 3)),
         jnp.cumsum(dr, axis=1)], axis=1,
    )                                                      # (n_chains, n_atoms, 3)
    return chains_pos[:, :1, :] + cumul


def _chain_rg(chains_pos, chain_masses):
    """Mass-weighted radius of gyration per chain, about the center of MASS.

    Parameters
    ----------
    chains_pos : (n_chains, n_atoms_per_chain, 3)
        Already PBC-unwrapped chain coordinates for one frame.
    chain_masses : (n_chains, n_atoms_per_chain)
        Per-atom masses for each chain.

    Returns
    -------
    (n_chains,) radius of gyration, mass-weighted about the center of mass
    (matches ``gmx gyrate``).  Uses ``sum(x**2)`` and a ``1e-30`` floor so the
    gradient stays finite as Rg -> 0 (see
    ``test_rg_pbc.TestNormVsSumGradientSafety``).
    """
    total_mass = jnp.sum(chain_masses, axis=1)                       # (n_chains,)
    com = (jnp.sum(chain_masses[:, :, None] * chains_pos, axis=1)
           / total_mass[:, None])                                    # (n_chains, 3)
    com = jnp.expand_dims(com, axis=1)                               # (n_chains, 1, 3)
    Rg2 = jnp.sum(chain_masses * jnp.sum((chains_pos - com) ** 2, axis=2),
                  axis=1) / total_mass
    return jnp.sqrt(jnp.maximum(Rg2, 1e-30))


@jit
def _pairwise_distances_pbc(pos, indices_a, indices_b, box):
    """Minimum-image pairwise distances between two atom groups.

    Returns a 1-D array of length Na * Nb with all pair distances.
    """
    pos_a = pos[indices_a]                              # (Na, 3)
    pos_b = pos[indices_b]                              # (Nb, 3)
    dr = pos_a[:, None, :] - pos_b[None, :, :]         # (Na, Nb, 3)
    dr = dr - box * jnp.around(dr / box)
    return jnp.linalg.norm(dr, axis=-1).ravel()         # (Na*Nb,)


# Available metrics: mse, rmse, smape, l2e, Kullback_Leibler, wasserstein_1d
@partial(jit, static_argnums=(2,))
def smape(predictions, targets, axis=None):
    denominator = jnp.abs(targets) + jnp.abs(predictions)
    condition = denominator > jnp.nextafter(1, 2) - 1
    safe_denom = jnp.where(condition, denominator, 1.0)
    result = jnp.where(
        condition,
        jnp.abs(targets - predictions) / safe_denom,
        0.0,
    )
    return jnp.mean(result, axis=axis)


@partial(jit, static_argnums=(2,))
def mse(predictions, targets, axis=None):
    # Mean Squared Error
    return jnp.mean((predictions - targets) ** 2, axis=axis)


@partial(jit, static_argnums=(2,))
def rmse(predictions, targets, axis=None):
    # Root Mean Squared Error
    return jnp.sqrt(jnp.mean((predictions - targets) ** 2, axis=axis))


@partial(jit, static_argnums=(2,))
def l2e(predictions, targets, axis=None):
    # L2 Error
    return jnp.linalg.norm(predictions - targets, axis=axis)


@partial(jit, static_argnums=(2,))
def Kullback_Leibler(predictions, targets, axis=None):
    """Forward KL divergence: KL(targets || predictions).

    Both inputs are normalised internally so that they sum to 1 along
    *axis* before the divergence is computed.  Bins where the target is
    (near-)zero are excluded to avoid log(0).

    Uses ``_eps = 1e-8`` in the log domain to bound per-bin gradients
    (``|dKL/dq| ≤ 1/eps ≈ 10⁸``) and prevent NaN during backprop
    when the simulated distribution has near-zero bins.
    """
    _eps = 1e-8   # log-domain floor: bounds |grad| and |log_ratio|
    # Broadcast so axis= works even when predictions is 1-D and targets is 2-D
    predictions, targets = jnp.broadcast_arrays(predictions, targets)
    # Normalise to proper probability distributions
    p = targets / jnp.sum(targets, axis=axis, keepdims=True).clip(1e-30)
    q = predictions / jnp.sum(predictions, axis=axis, keepdims=True).clip(1e-30)
    # KL(p || q) = sum p * log(p / q), ignoring bins where p ≈ 0
    log_ratio = jnp.log(jnp.maximum(p, _eps) / jnp.maximum(q, _eps))
    kl = jnp.where(p > _eps, p * log_ratio, 0.0)
    return jnp.sum(kl, axis=axis)


@partial(jit, static_argnums=(2,))
def wasserstein_1d(predictions, targets, axis=None):
    """1-D Wasserstein distance (Earth Mover's Distance) via the CDF trick.

    W1 = integral |CDF_P(x) - CDF_Q(x)| dx.  For discrete bins of equal
    width the integral reduces to sum |CDF_P - CDF_Q| (times bin width,
    which is a constant scale factor that does not affect optimisation).

    Both inputs are normalised internally to sum to 1 along *axis*.
    """
    _eps = 1e-30
    # Broadcast so axis= works even when predictions is 1-D and targets is 2-D
    predictions, targets = jnp.broadcast_arrays(predictions, targets)
    p = targets / jnp.sum(targets, axis=axis, keepdims=True).clip(_eps)
    q = predictions / jnp.sum(predictions, axis=axis, keepdims=True).clip(_eps)
    cdf_p = jnp.cumsum(p, axis=axis)
    cdf_q = jnp.cumsum(q, axis=axis)
    return jnp.sum(jnp.abs(cdf_p - cdf_q), axis=axis)


@jit
def harmonic_constraint(y, k, constraints):
    r"""
    Restrain :math:`\Chi` parameters with a harmonic potential:
    ..math::
        k \sum_{i \in \text{restraints}} (\Chi_i - \Chi_0)^2.
    Here ..math::
        k \eq \frac{1}{\Delta^2}
    where :math:`\Delta` defines the range :math:`[\Chi_0 - \Delta, \Chi_0 + \Delta]`
    outside which the restraint becomes greater than 1.
    """
    # Here chi is a 1D array of the upper triangular portion of the full matrix
    return k * jnp.sum(
        jnp.array([(y[ttc] - val) ** 2 for ttc, val in constraints.items()])
    )


@jit
def cubic_constraint(y, k, constraints):
    # Here chi is a 1D array of the upper triangular portion of the full matrix
    return k * jnp.sum(
        jnp.array([jnp.abs(y[ttc] - val) ** 3 for ttc, val in constraints.items()]),
    )


@jit
def boundary_constraint(params, C, S, upper_boundary=None, lower_boundary=None):
    '''Soft boundary penalty for parameters using sigmoid functions.
    
    Applies a smooth penalty when parameters exceed upper or fall below lower boundaries.
    Both upper and lower boundaries can be specified independently.
    
    Args:
        params: Parameter array (e.g., epsilon or sigma values)
        C: Amplitude of the penalty (larger = steeper wall)
        S: Steepness of the sigmoid (larger = sharper transition)
        upper_boundary: Maximum allowed value (None = no upper limit)
        lower_boundary: Minimum allowed value (None = no lower limit)
    
    Returns:
        Penalty term to add to loss function
    '''
    penalty = 0.0
    if upper_boundary is not None:
        # Penalty increases as params exceed upper_boundary
        penalty += 0.5 * jnp.sum(C * jax.nn.sigmoid((params - upper_boundary) * S))
    if lower_boundary is not None:
        # Penalty increases as params fall below lower_boundary
        penalty += 0.5 * jnp.sum(C * jax.nn.sigmoid((lower_boundary - params) * S))
    return penalty


@jit
def _sanitize_kde_bandwidth(bandwidth, data_range):
    min_bw = jnp.maximum((data_range[1] - data_range[0]) * 1e-3, 1e-6)
    return jnp.maximum(jnp.asarray(bandwidth, dtype=data_range.dtype), min_bw)


@jit
def _fixed_bandwidth_kde(samples, data_range, bandwidth):
    samples = jnp.ravel(samples)
    if samples.shape[0] == 0:
        return jnp.zeros_like(data_range)

    bandwidth = _sanitize_kde_bandwidth(bandwidth, data_range)
    norm = bandwidth * jnp.sqrt(2.0 * jnp.pi)

    def _eval_point(x):
        diffs = (x - samples) / bandwidth
        return jnp.mean(jnp.exp(-0.5 * diffs ** 2)) / norm

    return jax.lax.map(_eval_point, data_range)


@jit
def kde(samples, kde_out, data_range, bandwidth):
    kde_value = _fixed_bandwidth_kde(samples, data_range, bandwidth)
    kde_out = kde_out.at[...].set(kde_value)
    return kde_out


_COORDINATION_DIAGNOSTIC_BINS = 256
_COORDINATION_DIAGNOSTIC_PAD = 0.2
_COORDINATION_DIAGNOSTIC_MIN_SPAN = 0.4
_Q_DIAGNOSTIC_MIN = -3.0
_Q_DIAGNOSTIC_MAX = 1.0


@jit
def _coordination_diagnostic_range(target_values):
    target_values = jnp.ravel(jnp.asarray(target_values))
    lower = jnp.min(target_values)
    upper = jnp.max(target_values)
    span = jnp.maximum(upper - lower, _COORDINATION_DIAGNOSTIC_MIN_SPAN)
    pad = jnp.maximum(0.25 * span, _COORDINATION_DIAGNOSTIC_PAD)
    lower = jnp.maximum(0.0, lower - pad)
    upper = jnp.maximum(upper + pad, lower + _COORDINATION_DIAGNOSTIC_MIN_SPAN)
    return jnp.linspace(lower, upper, _COORDINATION_DIAGNOSTIC_BINS).astype(target_values.dtype)


@jit
def _q_diagnostic_range():
    return jnp.linspace(_Q_DIAGNOSTIC_MIN, _Q_DIAGNOSTIC_MAX, _COORDINATION_DIAGNOSTIC_BINS)


@jit
def lateral_density_kde(
    # fmt: off
    kde_density, centered_pos, types,
    z_range, bandwidth, bin_size, scaling_factor, config, box_z
):

    for i, t in enumerate(config.unique_types):
        sel = jnp.where(types == t, size=config.particle_per_type[t])
        type_t_pos = centered_pos[sel]

        # Reflect to account for PBC
        z_lower = - box_z + type_t_pos
        z_upper = box_z + type_t_pos
        type_t_pos_reflected = jnp.concat((z_lower, type_t_pos, z_upper))

        kde_value = (
            _fixed_bandwidth_kde(type_t_pos_reflected, z_range, bandwidth)
            * bin_size * config.particle_per_type[t] / scaling_factor * 3
        )
        kde_density = kde_density.at[i].add(kde_value)
    return kde_density


# ── Historical note ─────────────────────────────────────────────────────
# The original implementation passed `width_ratio * bin_size` into
# gaussian_kde(bw_method=...), which gaussian_kde interprets as a factor on
# the sample covariance, not as an absolute kernel width.  The helpers above
# now implement a true fixed-width Gaussian KDE so `width_ratio` has the
# documented semantics: absolute bandwidth = width_ratio * bin_size.
#
# @jit
# def _lateral_density_kde_fixed(
#     # fmt: off
#     kde_density, centered_pos, types,
#     z_range, bandwidth, bin_size, scaling_factor, config, box_z
# ):
#     """Fixed-width Gaussian kernel variant — vmap-compatible."""
#     for i, t in enumerate(config.unique_types):
#         sel = jnp.where(types == i, size=config.particle_per_type[t])
#         type_t_pos = centered_pos[sel]
#
#         z_lower = - box_z + type_t_pos
#         z_upper = box_z + type_t_pos
#         type_t_pos_reflected = jnp.concat((z_lower, type_t_pos, z_upper))
#
#         diffs = z_range[None, :] - type_t_pos_reflected[:, None]
#         kernels = jnp.exp(-0.5 * (diffs / bandwidth) ** 2) / (bandwidth * jnp.sqrt(2.0 * jnp.pi))
#         pdf = jnp.mean(kernels, axis=0)
#         kde_value = (
#             pdf * bin_size * config.particle_per_type[t] / scaling_factor * 3
#         )
#         kde_density = kde_density.at[i].add(kde_value)
#     return kde_density
#
#
# def density_and_apl_vectorized(
#     # fmt: off
#     model, system, key, start_temperature, comm,
#     z_range, com_type, n_lipids, target_density, target_apl,
#     metric, density_weight=1.0, k_constraint=0.01, apl_weight=1.0, width_ratio=1.0,
#     upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
#     boundary=None,
# ):
#     """Vectorized (vmap) variant using fixed-width Gaussian kernel.
#
#     Faster than the for-loop version (no Python loop over frames) but uses
#     an absolute bandwidth (width_ratio * bin_size) instead of the adaptive
#     gaussian_kde(bw_method=factor * std(data)).  This can produce spikier
#     profiles and hurt convergence for CG systems — use with caution.
#     """
#     sgm_table, epsl_table, param_constraints, types = get_LJ_param(model, system.config, jnp.array(system.types))
#
#     trj, key, config = simulator(
#         model, system.positions, system.velocities, types, system.masses, system.charges,
#         sgm_table, epsl_table, key, system.topol, system.config, start_temperature
#     )
#
#     comm_size = comm.Get_size()
#     n_bins = z_range.size
#     bin_size = z_range[1] - z_range[0]
#     n_frames = len(trj["positions"])
#
#     n_skip = 0
#     n_frames_adj = n_frames - n_skip
#     bandwidth = width_ratio * bin_size
#     z_length = z_range[-1] - z_range[0] + bin_size
#
#     pos_stack = jnp.stack(trj["positions"][n_skip:])
#     box_stack = jnp.stack(trj["box"][n_skip:])
#
#     fixed_sel = jnp.where(
#         system.types == com_type, size=config.particle_per_type[com_type]
#     )
#
#     def _frame_density(pos, box):
#         box_x, box_y, box_z = box
#         xy_area = box_x * box_y
#         scaling_factor = xy_area * z_length / n_bins
#
#         tails = pos[fixed_sel, 2]
#         com = _compute_com(tails, box_z)
#         centered_pos = _center(pos[:, 2], com, box_z)
#
#         frame_kde = _lateral_density_kde_fixed(
#             jnp.zeros((config.n_types, n_bins)), centered_pos, types,
#             z_range, bandwidth, bin_size, scaling_factor, config, box_z
#         )
#         return frame_kde, xy_area
#
#     all_kde, all_xy = jax.vmap(_frame_density)(pos_stack, box_stack)
#     kde_density = jnp.sum(all_kde, axis=0)
#     xy_apl = jnp.sum(all_xy)
#
#     kde_density = mpi4jax.allreduce(kde_density, op=MPI.SUM, comm=comm)
#     kde_density /= comm_size * n_frames_adj
#
#     error = (
#         jnp.sum(density_weight * metric(kde_density, target_density, axis=1))
#         / config.n_types
#     )
#
#     mean_apl = 2 * xy_apl / n_lipids
#     mean_apl = mpi4jax.allreduce(mean_apl, op=MPI.SUM, comm=comm)
#     mean_apl /= comm_size * n_frames_adj
#     error += apl_weight * metric(mean_apl, target_apl)
#
#     if constraint:
#         error += constraint(model.LJ_param, k_constraint, param_constraints)
#
#     _upper = upper_boundary if upper_boundary is not None else boundary
#     if _upper is not None or lower_boundary is not None:
#         error += boundary_constraint(epsl_table, boundary_C, boundary_S, _upper, lower_boundary)
#
#     return error, (
#         {"density": kde_density, "area per lipid": mean_apl},
#         trj, key, config, types,
#     )


def density_and_apl(
    # fmt: off
    model, system, key, start_temperature, comm,
    z_range, com_type, n_lipids, target_density, target_apl,    # System specific arguments
    metric, density_weight=1.0, k_constraint=0.01, apl_weight=1.0, width_ratio=1.0,   # General arguments for all systems (nn_options.loss_args)
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    """Loss function for lipid membranes based on lateral density profile and area per lipid"""

    sgm_table, epsl_table, param_constraints, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    # KDE / observable aggregation uses the per-replica-group sub-communicator
    # in heterogeneous multidir mode; falls back to WORLD otherwise.
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_bins = z_range.size
    bin_size = z_range[1] - z_range[0]
    n_frames = len(trj["positions"])

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip
    xy_apl = 0.0
    kde_density = jnp.zeros((config.n_types, n_bins))
    bandwidth = width_ratio * bin_size
    z_length = z_range[-1] - z_range[0] + bin_size

    for pos, box in zip(trj["positions"][n_skip:], trj["box"][n_skip:]):
        box_x, box_y, box_z = box

        # Add frame area per lipid
        xy_area = box_x * box_y
        scaling_factor = xy_area * z_length / n_bins
        xy_apl += xy_area

        fixed_sel = jnp.where(
            system.types == com_type, size=config.particle_per_type[com_type]
        )
        tails = pos[fixed_sel, 2]

        com = _compute_com(tails, box_z)
        centered_pos = _center(pos[:, 2], com, box_z)

        kde_density = lateral_density_kde(
            # fmt: off
            kde_density, centered_pos, system.types,
            z_range, bandwidth, bin_size, scaling_factor, config, box_z
        )

    # Pre-allreduce snapshot (this rank's per-frame averaged values)
    local_kde_density = kde_density / n_frames_adj
    local_mean_apl = (2 * xy_apl / n_lipids) / n_frames_adj

    kde_density = mpi4jax.allreduce(kde_density, op=MPI.SUM, comm=_kde_comm)
    kde_density /= _kde_size * n_frames_adj

    # Calculate error due to density
    err_density = jnp.sum(metric(kde_density, target_density, axis=1)) / config.n_types
    loss_density = (
        jnp.sum(density_weight * metric(kde_density, target_density, axis=1))
        / config.n_types
    )
    error = loss_density

    # Calculate error due to area per lipid
    mean_apl = 2 * xy_apl / n_lipids
    mean_apl = mpi4jax.allreduce(mean_apl, op=MPI.SUM, comm=_kde_comm)
    mean_apl /= _kde_size * n_frames_adj
    err_apl = metric(mean_apl, target_apl)
    loss_apl = apl_weight * err_apl
    error += loss_apl

    # Backward compatibility: old 'boundary' key maps to upper_boundary.
    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, param_constraints,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce component breakdown (for multidir _local)
    local_err_density = (
        jnp.sum(metric(local_kde_density, target_density, axis=1)) / config.n_types
    )
    local_loss_density = (
        jnp.sum(density_weight * metric(local_kde_density, target_density, axis=1))
        / config.n_types
    )
    local_err_apl = metric(local_mean_apl, target_apl)
    local_loss_apl = apl_weight * local_err_apl
    local_loss_total = (
        local_loss_density + local_loss_apl + local_loss_constraint + local_loss_boundary
    )

    return error, (
        {
            "density": kde_density,
            "area per lipid": mean_apl,
            "value_apl": mean_apl,
            "err_density": err_density,
            "err_apl": err_apl,
            "loss_density": loss_density,
            "loss_apl": loss_apl,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_density": jnp.asarray(density_weight),
            "weight_apl": jnp.asarray(apl_weight),
            "_local": _local_diag_snapshot(**{
                "density": local_kde_density,
                "area per lipid": local_mean_apl,
                "value_apl": local_mean_apl,
                "err_density": local_err_density,
                "err_apl": local_err_apl,
                "loss_density": local_loss_density,
                "loss_apl": local_loss_apl,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types,
    )


def radius_of_gyration_dist(
    # fmt: off
    model, system, key, start_temperature, comm,
    n_chains, n_atoms_per_chain, chain_indices, chain_masses,
    metric, data_range, target_dist, rg_weight=1.0, width_ratio=1.0, k_constraint=0.01,
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    # KDE aggregation runs over the per-replica-group sub-communicator in
    # heterogeneous multidir mode (so different logical systems' KDEs are
    # not mixed); falls back to WORLD when caller does not provide one.
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])

    n_bins = data_range.size
    bin_size = data_range[1] - data_range[0]

    kde_rg = jnp.zeros(n_bins)
    bandwidth = width_ratio * bin_size

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])   # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])          # (n_frames, 3)

    def _frame_rg(pos, box):
        chains_pos = jnp.take(pos, chain_indices, axis=0)
        chains_pos = _unwrap_chains_pbc(chains_pos, box)

        Rg = _chain_rg(chains_pos, chain_masses)
        return Rg                               # (n_chains,)

    Rg_all = jax.vmap(_frame_rg)(pos_stack, box_stack)  # (n_frames, n_chains)
    Rg_timeseries = Rg_all.reshape(-1)                   # (n_frames * n_chains,)

    kde_rg = kde(Rg_timeseries, kde_rg, data_range, bandwidth)

    # Pre-allreduce snapshot (this rank's local values)
    local_mean_Rg = jnp.mean(Rg_timeseries)
    local_kde_rg = kde_rg

    kde_rg = mpi4jax.allreduce(kde_rg, op=MPI.SUM, comm=_kde_comm)
    kde_rg /= _kde_size

    # Calculate error due to probablity density function
    err_rg_kde = jnp.sum(metric(kde_rg, target_dist, axis=1))
    loss_rg_kde = jnp.sum(rg_weight * metric(kde_rg, target_dist, axis=1))
    error = loss_rg_kde

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce component breakdown (for multidir _local)
    local_err_rg_kde = jnp.sum(metric(local_kde_rg, target_dist, axis=1))
    local_loss_rg_kde = jnp.sum(rg_weight * metric(local_kde_rg, target_dist, axis=1))
    local_loss_total = local_loss_rg_kde + local_loss_constraint + local_loss_boundary

    return error, (
        {
            "mean radius of gyration": local_mean_Rg,
            "Rg PDF": kde_rg,
            "value_mean_rg": local_mean_Rg,
            "err_rg_kde": err_rg_kde,
            "loss_rg_kde": loss_rg_kde,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_rg_kde": jnp.asarray(rg_weight),
            "_local": _local_diag_snapshot(**{
                "mean radius of gyration": local_mean_Rg,
                "Rg PDF": local_kde_rg,
                "value_mean_rg": local_mean_Rg,
                "err_rg_kde": local_err_rg_kde,
                "loss_rg_kde": local_loss_rg_kde,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types
    )


def radius_of_gyration(
    # fmt: off
    model, system, key, start_temperature, comm,
    n_chains, n_atoms_per_chain, chain_indices, chain_masses,   # arguments from unpacked reference dict
    metric, target_rg, rg_weight=1.0, k_constraint=0.01,   # arguments from unpacked reference dict
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])

    n_skip = 0
    n_frames_adj = n_frames - n_skip

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])   # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])          # (n_frames, 3)

    def _frame_rg(pos, box):
        chains_pos = jnp.take(pos, chain_indices, axis=0)
        chains_pos = _unwrap_chains_pbc(chains_pos, box)

        Rg = _chain_rg(chains_pos, chain_masses)
        return jnp.sum(Rg) / n_chains

    all_rg = jax.vmap(_frame_rg)(pos_stack, box_stack)  # (n_frames,)
    mean_Rg = jnp.sum(all_rg)

    # Pre-allreduce snapshot (this rank's per-frame mean)
    local_mean_Rg = mean_Rg / n_frames_adj

    # Calculate error
    mean_Rg = mpi4jax.allreduce(mean_Rg, op=MPI.SUM, comm=_kde_comm)
    mean_Rg /= _kde_size * n_frames_adj
    err_rg = metric(mean_Rg, target_rg)
    loss_rg = rg_weight * err_rg
    error = loss_rg

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_rg = metric(local_mean_Rg, target_rg)
    local_loss_rg = rg_weight * local_err_rg
    local_loss_total = local_loss_rg + local_loss_constraint + local_loss_boundary

    return error, (
        {
            "radius of gyration": mean_Rg,
            "value_rg": mean_Rg,
            "err_rg": err_rg,
            "loss_rg": loss_rg,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_rg": jnp.asarray(rg_weight),
            "_local": _local_diag_snapshot(**{
                "radius of gyration": local_mean_Rg,
                "value_rg": local_mean_Rg,
                "err_rg": local_err_rg,
                "loss_rg": local_loss_rg,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types
    )


def radius_of_gyration_median(
    # fmt: off
    model, system, key, start_temperature, comm,
    n_chains, n_atoms_per_chain, chain_indices, chain_masses,   # arguments from unpacked reference dict
    metric, target_rg, rg_weight=1.0, k_constraint=0.01,   # arguments from unpacked reference dict
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip

    all_Rg = jnp.zeros(_kde_size)

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])   # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])          # (n_frames, 3)

    def _frame_rg(pos, box):
        chains_pos = jnp.take(pos, chain_indices, axis=0)
        chains_pos = _unwrap_chains_pbc(chains_pos, box)

        Rg = _chain_rg(chains_pos, chain_masses)
        return jnp.sum(Rg) / n_chains

    all_frame_rg = jax.vmap(_frame_rg)(pos_stack, box_stack)  # (n_frames,)
    mean_Rg = jnp.sum(all_frame_rg) / n_frames_adj

    # Pre-allreduce snapshot (this rank's mean Rg)
    local_mean_Rg = mean_Rg

    all_Rg = all_Rg.at[_kde_comm.Get_rank()].set(mean_Rg)

    # Calculate error
    all_Rg = mpi4jax.allreduce(all_Rg, op=MPI.SUM, comm=_kde_comm)
    median_rg = jnp.median(all_Rg)
    err_rg_median = metric(median_rg, target_rg)
    loss_rg_median = rg_weight * err_rg_median
    error = loss_rg_median

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_rg_median = metric(local_mean_Rg, target_rg)
    local_loss_rg_median = rg_weight * local_err_rg_median
    local_loss_total = local_loss_rg_median + local_loss_constraint + local_loss_boundary

    return error, (
        {
            "radius of gyration": median_rg,
            "value_rg_median": median_rg,
            "err_rg_median": err_rg_median,
            "loss_rg_median": loss_rg_median,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_rg_median": jnp.asarray(rg_weight),
            "_local": _local_diag_snapshot(**{
                "radius of gyration": local_mean_Rg,
                "value_rg_median": local_mean_Rg,
                "err_rg_median": local_err_rg_median,
                "loss_rg_median": local_loss_rg_median,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types
    )


def filter_repls(all_rg, tol):
    median = jnp.median(all_rg)

    mask = (all_rg >= median*(1-tol)) & (all_rg <= median*(1+tol))

    ind = jnp.where(mask, size=all_rg.shape[0], fill_value=-1)[0]
    rg_filtered = jnp.take(all_rg, ind)
    rg_filtered = jnp.where(ind == -1, 0, rg_filtered)
    mean_filtered = jnp.sum(rg_filtered)/jnp.count_nonzero(rg_filtered)
    
    return mean_filtered


def radius_of_gyration_filter_repls(
    # fmt: off
    model, system, key, start_temperature, comm,
    n_chains, n_atoms_per_chain, chain_indices, chain_masses,   # arguments from unpacked reference dict
    metric, target_rg, rg_weight=1.0, k_constraint=0.01,   # arguments from unpacked reference dict
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip

    all_Rg = jnp.zeros(_kde_size)

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])   # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])          # (n_frames, 3)

    def _frame_rg(pos, box):
        chains_pos = jnp.take(pos, chain_indices, axis=0)
        chains_pos = _unwrap_chains_pbc(chains_pos, box)

        Rg = _chain_rg(chains_pos, chain_masses)
        return jnp.sum(Rg) / n_chains

    all_frame_rg = jax.vmap(_frame_rg)(pos_stack, box_stack)  # (n_frames,)
    mean_Rg = jnp.sum(all_frame_rg) / n_frames_adj

    # Pre-allreduce snapshot (this rank's mean Rg)
    local_mean_Rg = mean_Rg

    all_Rg = all_Rg.at[_kde_comm.Get_rank()].set(mean_Rg)

    # Calculate error
    all_Rg = mpi4jax.allreduce(all_Rg, op=MPI.SUM, comm=_kde_comm)

    all_Rg_filtered = filter_repls(all_Rg, 0.1)

    all_rg_mean = jnp.mean(all_Rg_filtered)
    err_rg_filtered = metric(all_rg_mean, target_rg)
    loss_rg_filtered = rg_weight * err_rg_filtered
    error = loss_rg_filtered

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_rg_filtered = metric(local_mean_Rg, target_rg)
    local_loss_rg_filtered = rg_weight * local_err_rg_filtered
    local_loss_total = local_loss_rg_filtered + local_loss_constraint + local_loss_boundary

    return error, (
        {
            "radius of gyration": all_rg_mean,
            "value_rg_filtered": all_rg_mean,
            "err_rg_filtered": err_rg_filtered,
            "loss_rg_filtered": loss_rg_filtered,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_rg_filtered": jnp.asarray(rg_weight),
            "_local": _local_diag_snapshot(**{
                "radius of gyration": local_mean_Rg,
                "value_rg_filtered": local_mean_Rg,
                "err_rg_filtered": local_err_rg_filtered,
                "loss_rg_filtered": local_loss_rg_filtered,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types
    )


def radius_of_gyration_and_end_to_end(
    # fmt: off
    model, system, key, start_temperature, comm,
    n_chains, n_atoms_per_chain, chain_indices, chain_masses,   # arguments from unpacked reference dict
    metric, target_rg, target_end_to_end, rg_weight=10.0, end_to_end_weight=1.0, k_constraint=0.01,   # arguments from unpacked reference dict
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
):
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(model, system.config, jnp.array(system.types))

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])   # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])          # (n_frames, 3)

    def _frame_rg_e2e(pos, box):
        # Calculate Rg
        chains_pos = jnp.take(pos, chain_indices, axis=0)
        chains_pos = _unwrap_chains_pbc(chains_pos, box)

        Rg = _chain_rg(chains_pos, chain_masses)
        rg_val = jnp.sum(Rg) / n_chains

        # Calculate end-to-end distance (unwrapped positions are contiguous)
        e2e_vec = chains_pos[0][-1] - chains_pos[0][0]
        e2e_val = jnp.sqrt(jnp.maximum(jnp.sum(e2e_vec ** 2), 1e-30))
        return rg_val, e2e_val

    all_rg, all_e2e = jax.vmap(_frame_rg_e2e)(pos_stack, box_stack)
    # all_rg: (n_frames,), all_e2e: (n_frames,)
    mean_Rg = jnp.sum(all_rg)
    mean_end_to_end = jnp.sum(all_e2e)

    # Pre-allreduce snapshot (this rank's per-frame means)
    local_mean_Rg = mean_Rg / n_frames_adj
    local_mean_end_to_end = mean_end_to_end / n_frames_adj

    # Calculate error due to Rg
    mean_Rg = mpi4jax.allreduce(mean_Rg, op=MPI.SUM, comm=_kde_comm)
    mean_Rg /= _kde_size * n_frames_adj
    err_rg = metric(mean_Rg, target_rg)
    loss_rg = rg_weight * err_rg
    error = loss_rg

    # Calculate error due to end-to-end distance
    mean_end_to_end = mpi4jax.allreduce(mean_end_to_end, op=MPI.SUM, comm=_kde_comm)
    mean_end_to_end /= _kde_size * n_frames_adj
    err_e2e = metric(mean_end_to_end, target_end_to_end)
    loss_e2e = end_to_end_weight * err_e2e
    error += loss_e2e

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_rg = metric(local_mean_Rg, target_rg)
    local_loss_rg = rg_weight * local_err_rg
    local_err_e2e = metric(local_mean_end_to_end, target_end_to_end)
    local_loss_e2e = end_to_end_weight * local_err_e2e
    local_loss_total = (
        local_loss_rg + local_loss_e2e + local_loss_constraint + local_loss_boundary
    )

    return error, (
        {
            "radius of gyration": mean_Rg,
            "end-to-end distance": mean_end_to_end,
            "value_rg": mean_Rg,
            "value_e2e": mean_end_to_end,
            "err_rg": err_rg,
            "err_e2e": err_e2e,
            "loss_rg": loss_rg,
            "loss_e2e": loss_e2e,
            "loss_constraint": loss_constraint,
            "loss_boundary": loss_boundary,
            "loss_total": error,
            "weight_rg": jnp.asarray(rg_weight),
            "weight_e2e": jnp.asarray(end_to_end_weight),
            "_local": _local_diag_snapshot(**{
                "radius of gyration": local_mean_Rg,
                "end-to-end distance": local_mean_end_to_end,
                "value_rg": local_mean_Rg,
                "value_e2e": local_mean_end_to_end,
                "err_rg": local_err_rg,
                "err_e2e": local_err_e2e,
                "loss_rg": local_loss_rg,
                "loss_e2e": local_loss_e2e,
                "loss_constraint": local_loss_constraint,
                "loss_boundary": local_loss_boundary,
                "loss_total": local_loss_total,
            }),
        },
        trj,
        key,
        config,
        types
    )


def thickness(
    # fmt: off
    model, system, key, start_temperature, comm,
    chain_indices, chain_masses,   # arguments from unpacked reference dict
    metric, target_rg, rg_weight=1.0, k_constraint=0.01,   # arguments from unpacked reference dict
    boundary=None, constraint=None,
):
    raise NotImplementedError(
        "The 'thickness' loss function is currently incomplete and disabled. "
        "Use one of the implemented losses (e.g. density_and_apl or radius_of_gyration*)."
    )


# ---------------------------------------------------------------------------
# Coordination distance loss functions
# ---------------------------------------------------------------------------

def coordination_distance_dist(
    # fmt: off
    model, system, key, start_temperature, comm,
    coord_pair_indices, coord_pairs,                                        # System specific
    data_range, target_dist,                                                # Reference distribution
    metric, dist_weight=1.0, width_ratio=1.0, k_constraint=0.01,           # General
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
    # fmt: on
):
    """Loss based on KDE distributions of coordination distances.

    Compares simulated pairwise distance distributions for each coordination
    pair (e.g. SZ-Zn, NZ-Zn, SD-Zn) against reference distributions loaded
    from a .xvg file.  New pairs can be added by extending ``coord_pairs``
    and ``coord_pair_indices`` in the training TOML.

    Each MPI rank collects all pairwise distances over the full trajectory,
    builds a KDE per pair, then averages KDEs across ranks.
    """
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])
    n_pairs = len(coord_pair_indices)

    n_bins = data_range.size
    bin_size = data_range[1] - data_range[0]
    bandwidth = width_ratio * bin_size

    kde_dists = jnp.zeros((n_pairs, n_bins))
    mean_dists = jnp.zeros(n_pairs)

    n_skip = 0

    # Vectorised frame loop: stack trajectory, vmap over frames per pair.
    pos_stack = jnp.stack(trj["positions"][n_skip:])  # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])         # (n_frames, 3)

    for p, (idx_a, idx_b) in enumerate(coord_pair_indices):
        # vmap computes all pairwise distances in one vectorised call
        frame_dists = jax.vmap(
            lambda pos, box: _pairwise_distances_pbc(pos, idx_a, idx_b, box)
        )(pos_stack, box_stack)                         # (n_frames, Na*Nb)
        all_dists_p = frame_dists.reshape(-1)            # (n_frames * Na*Nb,)
        mean_dists = mean_dists.at[p].set(jnp.mean(all_dists_p))
        pair_kde = kde(all_dists_p, jnp.zeros(n_bins), data_range, bandwidth)
        kde_dists = kde_dists.at[p].set(pair_kde)

    # Pre-allreduce snapshot (this rank's per-pair distributions / means)
    local_kde_dists = kde_dists
    local_mean_dists = mean_dists

    # Average KDE and mean distances across ranks in this replica group
    # (each rank contributes one independent trajectory-averaged KDE)
    kde_dists = mpi4jax.allreduce(kde_dists, op=MPI.SUM, comm=_kde_comm)
    kde_dists /= _kde_size

    mean_dists = mpi4jax.allreduce(mean_dists, op=MPI.SUM, comm=_kde_comm)
    mean_dists /= _kde_size

    # Error from distribution matching (normalized over pairs)
    err_kde_per_pair = metric(kde_dists, target_dist, axis=1)               # (n_pairs,)
    loss_kde_per_pair = dist_weight * err_kde_per_pair / n_pairs            # (n_pairs,) — sum == error
    err_kde_total = jnp.sum(err_kde_per_pair) / n_pairs
    loss_kde_total = jnp.sum(loss_kde_per_pair)
    error = loss_kde_total

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_kde_per_pair = metric(local_kde_dists, target_dist, axis=1)
    local_loss_kde_per_pair = dist_weight * local_err_kde_per_pair / n_pairs
    local_err_kde_total = jnp.sum(local_err_kde_per_pair) / n_pairs
    local_loss_kde_total = jnp.sum(local_loss_kde_per_pair)
    local_loss_total = local_loss_kde_total + local_loss_constraint + local_loss_boundary

    # Build per-pair diagnostics
    diag = {
        "coordination distance KDE": kde_dists,
        "coordination distance distribution grid": data_range,
        "loss_kde_total": loss_kde_total,
        "err_kde_total": err_kde_total,
        "weight_kde": jnp.asarray(dist_weight),
        "loss_constraint": loss_constraint,
        "loss_boundary": loss_boundary,
        "loss_total": error,
    }
    local_diag_kwargs = {
        "coordination distance KDE": local_kde_dists,
        "loss_kde_total": local_loss_kde_total,
        "err_kde_total": local_err_kde_total,
        "loss_constraint": local_loss_constraint,
        "loss_boundary": local_loss_boundary,
        "loss_total": local_loss_total,
    }
    for p, pair_name in enumerate(coord_pairs):
        label = f"mean dist {pair_name[0]}-{pair_name[1]}"
        sanitized = f"{pair_name[0]}_{pair_name[1]}"
        diag[label] = mean_dists[p]
        diag[f"value_dist_{sanitized}"] = mean_dists[p]
        diag[f"err_kde_{sanitized}"] = err_kde_per_pair[p]
        diag[f"loss_kde_{sanitized}"] = loss_kde_per_pair[p]
        local_diag_kwargs[label] = local_mean_dists[p]
        local_diag_kwargs[f"value_dist_{sanitized}"] = local_mean_dists[p]
        local_diag_kwargs[f"err_kde_{sanitized}"] = local_err_kde_per_pair[p]
        local_diag_kwargs[f"loss_kde_{sanitized}"] = local_loss_kde_per_pair[p]
    diag["_local"] = _local_diag_snapshot(**local_diag_kwargs)

    return error, (diag, trj, key, config, types)


def coordination_distance(
    # fmt: off
    model, system, key, start_temperature, comm,
    coord_pair_indices, coord_pairs,                                        # System specific
    target_distances,                                                       # Target mean distances
    metric, dist_weight=1.0, k_constraint=0.01,                            # General
    data_range=None, target_dist=None, width_ratio=1.0,                    # Optional diagnostic KDE output
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
    # fmt: on
):
    """Loss based on mean coordination distances.

    Compares the time-averaged mean pairwise distance for each coordination
    pair against target values.  Simpler alternative to
    ``coordination_distance_dist`` when only the mean distance matters.
    """
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])
    n_pairs = len(coord_pair_indices)

    mean_dists = jnp.zeros(n_pairs)

    n_skip = 0
    n_frames_adj = n_frames - n_skip

    # Vectorised frame loop: stack trajectory, vmap over frames.
    # The inner pair loop is a small static Python loop (typically 1-5 pairs)
    # that gets unrolled at trace time.
    pos_stack = jnp.stack(trj["positions"][n_skip:])      # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])             # (n_frames, 3)

    def _frame_mean_dists(pos, box):
        pair_means = []
        for idx_a, idx_b in coord_pair_indices:
            dists = _pairwise_distances_pbc(pos, idx_a, idx_b, box)
            pair_means.append(jnp.mean(dists))
        return jnp.array(pair_means)                       # (n_pairs,)

    all_means = jax.vmap(_frame_mean_dists)(pos_stack, box_stack)  # (n_frames, n_pairs)
    mean_dists = jnp.mean(all_means, axis=0)                       # (n_pairs,)

    # Pre-allreduce snapshot (this rank's per-pair means)
    local_mean_dists = mean_dists

    # Average across ranks in this replica group
    mean_dists = mpi4jax.allreduce(mean_dists, op=MPI.SUM, comm=_kde_comm)
    mean_dists /= _kde_size

    diag_data_range = data_range
    if diag_data_range is None:
        diag_data_range = _coordination_diagnostic_range(target_distances)

    n_bins = diag_data_range.size
    bin_size = diag_data_range[1] - diag_data_range[0]
    bandwidth = width_ratio * bin_size
    kde_dists = jnp.zeros((n_pairs, n_bins))

    for p, (idx_a, idx_b) in enumerate(coord_pair_indices):
        frame_dists = jax.vmap(
            lambda pos, box: _pairwise_distances_pbc(pos, idx_a, idx_b, box)
        )(pos_stack, box_stack)
        all_dists_p = frame_dists.reshape(-1)
        pair_kde = kde(all_dists_p, jnp.zeros(n_bins), diag_data_range, bandwidth)
        kde_dists = kde_dists.at[p].set(pair_kde)

    # Pre-allreduce snapshot of the diagnostic KDE
    local_kde_dists = kde_dists

    kde_dists = mpi4jax.allreduce(kde_dists, op=MPI.SUM, comm=_kde_comm)
    kde_dists /= _kde_size

    # Error from mean distance matching (metric normalizes over pairs)
    err_dist_total = metric(mean_dists, target_distances)
    loss_dist_total = dist_weight * err_dist_total
    error = loss_dist_total

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_dist_total = metric(local_mean_dists, target_distances)
    local_loss_dist_total = dist_weight * local_err_dist_total
    local_loss_total = local_loss_dist_total + local_loss_constraint + local_loss_boundary

    # Build per-pair diagnostics
    diag = {
        "coordination distance KDE": kde_dists,
        "coordination distance distribution grid": diag_data_range,
        "loss_dist_total": loss_dist_total,
        "err_dist_total": err_dist_total,
        "weight_dist": jnp.asarray(dist_weight),
        "loss_constraint": loss_constraint,
        "loss_boundary": loss_boundary,
        "loss_total": error,
    }
    local_diag_kwargs = {
        "coordination distance KDE": local_kde_dists,
        "loss_dist_total": local_loss_dist_total,
        "err_dist_total": local_err_dist_total,
        "loss_constraint": local_loss_constraint,
        "loss_boundary": local_loss_boundary,
        "loss_total": local_loss_total,
    }
    for p, pair_name in enumerate(coord_pairs):
        label = f"mean dist {pair_name[0]}-{pair_name[1]}"
        sanitized = f"{pair_name[0]}_{pair_name[1]}"
        diag[label] = mean_dists[p]
        diag[f"value_dist_{sanitized}"] = mean_dists[p]
        local_diag_kwargs[label] = local_mean_dists[p]
        local_diag_kwargs[f"value_dist_{sanitized}"] = local_mean_dists[p]
    diag["_local"] = _local_diag_snapshot(**local_diag_kwargs)

    return error, (diag, trj, key, config, types)


# ---------------------------------------------------------------------------
# Tetrahedral order parameter helpers
# ---------------------------------------------------------------------------

@jit
def _tetrahedral_order_q(pos, metal_idx, ligand_indices, box):
    """Orientational tetrahedral order parameter (Errington-Debenedetti).

    Computes q for a single metal centre and its four ligand atoms::

        q = 1 - (3/8) * sum_{j<k} (cos(psi_jk) + 1/3)^2

    where psi_jk is the angle subtended at the metal centre by the
    displacement vectors to ligands j and k.  A perfect tetrahedron
    gives q = 1 and a random arrangement gives q ~ 0.

    Parameters
    ----------
    pos : (N, 3)   All atomic positions for this frame (nm).
    metal_idx : int   Index of the central metal atom.
    ligand_indices : (4,)  Indices of the four coordinating atoms.
    box : (3,)   Periodic box lengths.

    Returns
    -------
    q : scalar   Tetrahedral order parameter.
    """
    r_metal = pos[metal_idx]                           # (3,)
    r_lig = pos[ligand_indices]                        # (4, 3)

    # Minimum-image displacement vectors metal → ligand
    dr = r_lig - r_metal                               # (4, 3)
    dr = dr - box * jnp.around(dr / box)

    # Normalise to unit vectors
    norms = jnp.linalg.norm(dr, axis=1, keepdims=True)  # (4, 1)
    u = dr / jnp.maximum(norms, 1e-12)                   # (4, 3)

    # All C(4,2) = 6 cosines between pairs of unit vectors
    # Pair indices: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    idx_j = jnp.array([0, 0, 0, 1, 1, 2])
    idx_k = jnp.array([1, 2, 3, 2, 3, 3])

    cos_psi = jnp.sum(u[idx_j] * u[idx_k], axis=1)    # (6,)

    q = 1.0 - (3.0 / 8.0) * jnp.sum((cos_psi + 1.0 / 3.0) ** 2)
    return q


@jit
def _site_distances_pbc(pos, metal_idx, ligand_indices, box):
    """Metal-ligand distances for one coordination site under PBC.

    Returns the 4 individual distances (not all-vs-all).
    """
    r_metal = pos[metal_idx]
    r_lig = pos[ligand_indices]                        # (4, 3)
    dr = r_lig - r_metal
    dr = dr - box * jnp.around(dr / box)
    return jnp.linalg.norm(dr, axis=1)                 # (4,)


# ---------------------------------------------------------------------------
# Tetrahedral coordination loss functions
# ---------------------------------------------------------------------------

def coordination_tetrahedral(
    # fmt: off
    model, system, key, start_temperature, comm,
    site_metal_indices, site_ligand_indices,                                # Per-site topology
    site_group_labels, site_group_slices,                                   # Per-site ligand grouping
    target_q, target_site_distances,                                        # Per-site targets
    metric, dist_weight=1.0, q_weight=1.0, k_constraint=0.01,              # General
    data_range=None, target_dist=None, q_data_range=None, target_q_dist=None, width_ratio=1.0,
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
    # fmt: on
):
    """Loss combining mean coordination distances with the tetrahedral
    order parameter *q* for an arbitrary number of metal-centre sites.

    Each site is defined by one metal atom and four coordinating ligand
    atoms (given as absolute H5 indices).  Bridging atoms can appear in
    multiple sites.  Within each site, ligand atoms are grouped by atom
    type so that per-type mean distances are matched independently.

    target_q : (n_sites,)
        Target *q* for each site.  Usually 1.0 for tetrahedral Zn2+.
    target_site_distances : (n_groups,)
        Flat target distance array.  One entry per (site, ligand_type)
        group, ordered as reported in the startup log.
    """
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])
    n_sites = len(site_metal_indices)
    n_groups = len(site_group_labels)

    # Precompute per-site group list with site-local slice indices.
    # site_group_slices stores global offsets (site s occupies global slots
    # [4*s, 4*(s+1))).  Converting to local once here avoids repeating the
    # arithmetic inside the hot frame loop.
    site_groups = [[] for _ in range(n_sites)]
    for g, (gstart, gend) in enumerate(site_group_slices):
        s_owner = gstart // 4          # which site owns this group
        ls = gstart - 4 * s_owner      # site-local start index (0-3)
        le = gend   - 4 * s_owner      # site-local end   index (1-4)
        site_groups[s_owner].append((g, ls, le))

    mean_q = jnp.zeros(n_sites)
    mean_group_dists = jnp.zeros(n_groups)

    n_skip = 0
    n_frames_adj = n_frames - n_skip

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])  # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])         # (n_frames, 3)

    def _frame_q_and_dists(pos, box):
        q_vals = jnp.array([
            _tetrahedral_order_q(pos, site_metal_indices[s],
                                site_ligand_indices[s], box)
            for s in range(n_sites)
        ])                                                  # (n_sites,)
        site_dists = jnp.stack([
            _site_distances_pbc(pos, site_metal_indices[s],
                               site_ligand_indices[s], box)
            for s in range(n_sites)
        ])                                                  # (n_sites, 4)
        group_means = jnp.zeros(n_groups)
        for s in range(n_sites):
            dists_s = site_dists[s]
            for g, ls, le in site_groups[s]:
                group_means = group_means.at[g].set(jnp.mean(dists_s[ls:le]))
        return q_vals, group_means, site_dists             # (n_sites,), (n_groups,), (n_sites, 4)

    all_q, all_dists, all_site_dists = jax.vmap(_frame_q_and_dists)(pos_stack, box_stack)
    # all_q: (n_frames, n_sites), all_dists: (n_frames, n_groups)
    # all_site_dists: (n_frames, n_sites, 4)
    mean_q = jnp.mean(all_q, axis=0)
    mean_group_dists = jnp.mean(all_dists, axis=0)

    # Pre-allreduce snapshot (this rank's per-site/per-group means)
    local_mean_q = mean_q
    local_mean_group_dists = mean_group_dists

    # Average over frames and replica-group ranks
    mean_q = mpi4jax.allreduce(mean_q, op=MPI.SUM, comm=_kde_comm)
    mean_q /= _kde_size

    mean_group_dists = mpi4jax.allreduce(mean_group_dists, op=MPI.SUM, comm=_kde_comm)
    mean_group_dists /= _kde_size

    dist_diag_range = data_range
    if dist_diag_range is None:
        dist_diag_range = _coordination_diagnostic_range(target_site_distances)

    n_bins = dist_diag_range.size
    bin_size = dist_diag_range[1] - dist_diag_range[0]
    bandwidth = width_ratio * bin_size
    kde_dists = jnp.zeros((n_groups, n_bins))

    for g, (gstart, gend) in enumerate(site_group_slices):
        s_owner = gstart // 4
        ls = gstart - 4 * s_owner
        le = gend   - 4 * s_owner
        group_d = all_site_dists[:, s_owner, ls:le].reshape(-1)
        pair_kde = kde(group_d, jnp.zeros(n_bins), dist_diag_range, bandwidth)
        kde_dists = kde_dists.at[g].set(pair_kde)

    # Pre-allreduce snapshot of per-group distance KDE
    local_kde_dists = kde_dists

    kde_dists = mpi4jax.allreduce(kde_dists, op=MPI.SUM, comm=_kde_comm)
    kde_dists /= _kde_size

    q_diag_range = q_data_range if q_data_range is not None else _q_diagnostic_range()
    q_n_bins = q_diag_range.size
    q_bin_size = q_diag_range[1] - q_diag_range[0]
    q_bw = width_ratio * q_bin_size
    kde_q = jnp.zeros((n_sites, q_n_bins))

    for s in range(n_sites):
        q_series = all_q[:, s]
        site_kde = kde(q_series, jnp.zeros(q_n_bins), q_diag_range, q_bw)
        kde_q = kde_q.at[s].set(site_kde)

    # Pre-allreduce snapshot of per-site q KDE
    local_kde_q = kde_q

    kde_q = mpi4jax.allreduce(kde_q, op=MPI.SUM, comm=_kde_comm)
    kde_q /= _kde_size

    # Error terms
    err_q_total = metric(mean_q, target_q)
    loss_q_total = q_weight * err_q_total
    error = loss_q_total

    err_dist_total = metric(mean_group_dists, target_site_distances)
    loss_dist_total = dist_weight * err_dist_total
    error += loss_dist_total

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Per-rank pre-allreduce breakdown (for multidir _local)
    local_err_q_total = metric(local_mean_q, target_q)
    local_loss_q_total = q_weight * local_err_q_total
    local_err_dist_total = metric(local_mean_group_dists, target_site_distances)
    local_loss_dist_total = dist_weight * local_err_dist_total
    local_loss_total = (
        local_loss_q_total + local_loss_dist_total + local_loss_constraint + local_loss_boundary
    )

    # Diagnostics
    diag = {
        "coordination distance KDE": kde_dists,
        "coordination distance distribution grid": dist_diag_range,
        "q distribution KDE": kde_q,
        "q distribution grid": q_diag_range,
        "loss_q_total": loss_q_total,
        "loss_dist_total": loss_dist_total,
        "err_q_total": err_q_total,
        "err_dist_total": err_dist_total,
        "weight_q": jnp.asarray(q_weight),
        "weight_dist": jnp.asarray(dist_weight),
        "loss_constraint": local_loss_constraint,
        "loss_boundary": local_loss_boundary,
        "loss_total": error,
    }
    local_diag_kwargs = {
        "coordination distance KDE": local_kde_dists,
        "q distribution KDE": local_kde_q,
        "loss_q_total": local_loss_q_total,
        "loss_dist_total": local_loss_dist_total,
        "err_q_total": local_err_q_total,
        "err_dist_total": local_err_dist_total,
        "loss_constraint": local_loss_constraint,
        "loss_boundary": local_loss_boundary,
        "loss_total": local_loss_total,
    }
    for s in range(n_sites):
        diag[f"site {s} q"] = mean_q[s]
        diag[f"value_q_{s}"] = mean_q[s]
        local_diag_kwargs[f"site {s} q"] = local_mean_q[s]
        local_diag_kwargs[f"value_q_{s}"] = local_mean_q[s]
    for g, label in enumerate(site_group_labels):
        diag[f"mean dist {label}"] = mean_group_dists[g]
        diag[f"value_dist_{label}"] = mean_group_dists[g]
        local_diag_kwargs[f"mean dist {label}"] = local_mean_group_dists[g]
        local_diag_kwargs[f"value_dist_{label}"] = local_mean_group_dists[g]
    diag["_local"] = _local_diag_snapshot(**local_diag_kwargs)

    return error, (diag, trj, key, config, types)


def coordination_tetrahedral_dist(
    # fmt: off
    model, system, key, start_temperature, comm,
    site_metal_indices, site_ligand_indices,                                # Per-site topology
    site_group_labels, site_group_slices,                                   # Per-site ligand grouping
    target_q,                                                               # Per-site target q
    data_range, target_dist,                                                # Distance distributions
    metric, dist_weight=1.0, q_weight=1.0, width_ratio=1.0,                # General
    k_constraint=0.01,
    q_data_range=None, target_q_dist=None, q_dist_weight=1.0,              # Optional q distribution
    upper_boundary=None, lower_boundary=None, boundary_S=2, boundary_C=500, constraint=None,
    boundary=None,  # deprecated, use upper_boundary
    replica_comm=None,
    # fmt: on
):
    """Distribution-based variant of ``coordination_tetrahedral``.

    Distance distributions are built per (site, ligand_type) group and
    compared against reference distributions from an XVG file.

    The *q* matching uses the mean value by default.  If ``target_q_dist``
    is provided (via a second XVG), the loss also includes a per-site KDE
    match on the *q* time series.
    """
    sgm_table, epsl_table, epsl_constraint, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types, system.masses, system.charges,
        sgm_table, epsl_table, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    _kde_comm, _kde_size = _resolve_kde_comm(comm, replica_comm)
    n_frames = len(trj["positions"])
    n_sites = len(site_metal_indices)
    n_groups = len(site_group_labels)

    # ── Guard: at least one frame must be available ────────────────────
    if n_frames < 1:
        raise ValueError(
            f"coordination_tetrahedral_dist: no trajectory frames were collected. "
            f"Decrease n_print (currently {config.n_print}) so that "
            f"n_steps // n_print >= 1, e.g. n_print = {max(1, config.n_steps // 10)}."
        )

    n_bins = data_range.size
    bin_size = data_range[1] - data_range[0]
    bandwidth = width_ratio * bin_size

    # Precompute per-site group list with site-local slice indices.
    site_groups = [[] for _ in range(n_sites)]
    for g, (gstart, gend) in enumerate(site_group_slices):
        s_owner = gstart // 4
        ls = gstart - 4 * s_owner
        le = gend   - 4 * s_owner
        site_groups[s_owner].append((g, ls, le))

    n_skip = 0

    # Vectorised frame loop: stack trajectory, vmap over frames.
    pos_stack = jnp.stack(trj["positions"][n_skip:])  # (n_frames, N, 3)
    box_stack = jnp.stack(trj["box"][n_skip:])         # (n_frames, 3)

    def _frame_q_and_dists(pos, box):
        q_vals = jnp.array([
            _tetrahedral_order_q(pos, site_metal_indices[s],
                                site_ligand_indices[s], box)
            for s in range(n_sites)
        ])                                                  # (n_sites,)
        site_dists = jnp.stack([
            _site_distances_pbc(pos, site_metal_indices[s],
                               site_ligand_indices[s], box)
            for s in range(n_sites)
        ])                                                  # (n_sites, 4)
        return q_vals, site_dists

    all_q, all_site_dists = jax.vmap(_frame_q_and_dists)(pos_stack, box_stack)
    # all_q: (n_frames, n_sites), all_site_dists: (n_frames, n_sites, 4)

    # --- Distance KDE per group ---
    kde_dists = jnp.zeros((n_groups, n_bins))
    mean_group_dists = jnp.zeros(n_groups)

    for g, (gstart, gend) in enumerate(site_group_slices):
        s_owner = gstart // 4
        ls = gstart - 4 * s_owner
        le = gend   - 4 * s_owner
        group_d = all_site_dists[:, s_owner, ls:le].reshape(-1)
        mean_group_dists = mean_group_dists.at[g].set(jnp.mean(group_d))
        pair_kde = kde(group_d, jnp.zeros(n_bins), data_range, bandwidth)
        kde_dists = kde_dists.at[g].set(pair_kde)

    # Pre-allreduce snapshots (this rank's per-group KDE / means)
    local_kde_dists = kde_dists
    local_mean_group_dists = mean_group_dists

    kde_dists = mpi4jax.allreduce(kde_dists, op=MPI.SUM, comm=_kde_comm)
    kde_dists /= _kde_size

    mean_group_dists = mpi4jax.allreduce(mean_group_dists, op=MPI.SUM, comm=_kde_comm)
    mean_group_dists /= _kde_size

    # --- q statistics ---
    mean_q = jnp.mean(all_q, axis=0)                       # (n_sites,)
    local_mean_q = mean_q  # snapshot before allreduce
    mean_q = mpi4jax.allreduce(mean_q, op=MPI.SUM, comm=_kde_comm)
    mean_q /= _kde_size

    # --- Error: distance distribution ---
    err_dist_kde = jnp.sum(metric(kde_dists, target_dist, axis=1)) / n_groups
    loss_dist_kde = jnp.sum(dist_weight * metric(kde_dists, target_dist, axis=1)) / n_groups
    error = loss_dist_kde

    # --- Error: q ---
    err_q_mean = metric(mean_q, target_q)
    loss_q_mean = q_weight * err_q_mean
    error += loss_q_mean

    # --- Optional: q distribution KDE ---
    q_diag_range = q_data_range if q_data_range is not None else _q_diagnostic_range()
    q_n_bins = q_diag_range.size
    q_bin_size = q_diag_range[1] - q_diag_range[0]
    q_bw = width_ratio * q_bin_size
    kde_q = jnp.zeros((n_sites, q_n_bins))

    for s in range(n_sites):
        q_series = all_q[:, s]                          # (n_frames,)
        site_kde = kde(q_series, jnp.zeros(q_n_bins), q_diag_range, q_bw)
        kde_q = kde_q.at[s].set(site_kde)

    # Pre-allreduce snapshot of per-site q KDE
    local_kde_q = kde_q

    kde_q = mpi4jax.allreduce(kde_q, op=MPI.SUM, comm=_kde_comm)
    kde_q /= _kde_size

    loss_q_dist = jnp.array(0.0)
    err_q_dist = jnp.array(0.0)
    if target_q_dist is not None and q_data_range is not None:
        err_q_dist = jnp.sum(metric(kde_q, target_q_dist, axis=1)) / n_sites
        loss_q_dist = jnp.sum(q_dist_weight * metric(kde_q, target_q_dist, axis=1)) / n_sites
        error += loss_q_dist

    _upper = upper_boundary if upper_boundary is not None else boundary
    loss_constraint, local_loss_constraint, loss_boundary, local_loss_boundary = _regularizer_losses(
        model, system.config, comm, constraint, k_constraint, epsl_constraint,
        boundary_C, boundary_S, _upper, lower_boundary,
    )
    error += loss_constraint + loss_boundary

    # Diagnostics
    diag = {
        "coordination distance KDE": kde_dists,
        "coordination distance distribution grid": data_range,
        "q distribution KDE": kde_q,
        "q distribution grid": q_diag_range,
    }
    diag["loss_dist_kde"] = loss_dist_kde
    diag["loss_q_mean"] = loss_q_mean
    diag["loss_q_dist"] = loss_q_dist
    diag["loss_constraint"] = loss_constraint
    diag["loss_boundary"] = loss_boundary
    diag["loss_total"] = error
    diag["err_dist_kde"] = err_dist_kde
    diag["err_q_mean"] = err_q_mean
    diag["err_q_dist"] = err_q_dist
    diag["weight_dist_kde"] = jnp.asarray(dist_weight)
    diag["weight_q_mean"] = jnp.asarray(q_weight)
    diag["weight_q_dist"] = jnp.asarray(q_dist_weight)

    # Per-rank pre-allreduce breakdown (computed from local_* arrays so
    # the multidir per-system table reflects each rank's contribution).
    local_err_dist_kde = jnp.sum(metric(local_kde_dists, target_dist, axis=1)) / n_groups
    local_loss_dist_kde = (
        jnp.sum(dist_weight * metric(local_kde_dists, target_dist, axis=1)) / n_groups
    )
    local_err_q_mean = metric(local_mean_q, target_q)
    local_loss_q_mean = q_weight * local_err_q_mean
    local_err_q_dist = jnp.array(0.0)
    local_loss_q_dist = jnp.array(0.0)
    if target_q_dist is not None and q_data_range is not None:
        local_err_q_dist = jnp.sum(metric(local_kde_q, target_q_dist, axis=1)) / n_sites
        local_loss_q_dist = (
            jnp.sum(q_dist_weight * metric(local_kde_q, target_q_dist, axis=1)) / n_sites
        )
    local_loss_total = (
        local_loss_dist_kde + local_loss_q_mean + local_loss_q_dist
        + local_loss_constraint + local_loss_boundary
    )

    local_diag_kwargs = {
        "coordination distance KDE": local_kde_dists,
        "q distribution KDE": local_kde_q,
        "loss_dist_kde": local_loss_dist_kde,
        "loss_q_mean": local_loss_q_mean,
        "loss_q_dist": local_loss_q_dist,
        "loss_constraint": local_loss_constraint,
        "loss_boundary": local_loss_boundary,
        "loss_total": local_loss_total,
        "err_dist_kde": local_err_dist_kde,
        "err_q_mean": local_err_q_mean,
        "err_q_dist": local_err_q_dist,
    }
    for s in range(n_sites):
        diag[f"site {s} q"] = mean_q[s]
        diag[f"value_q_{s}"] = mean_q[s]
        local_diag_kwargs[f"site {s} q"] = local_mean_q[s]
        local_diag_kwargs[f"value_q_{s}"] = local_mean_q[s]
    for g, label in enumerate(site_group_labels):
        diag[f"mean dist {label}"] = mean_group_dists[g]
        diag[f"value_dist_{label}"] = mean_group_dists[g]
        local_diag_kwargs[f"mean dist {label}"] = local_mean_group_dists[g]
        local_diag_kwargs[f"value_dist_{label}"] = local_mean_group_dists[g]
    diag["_local"] = _local_diag_snapshot(**local_diag_kwargs)

    return error, (diag, trj, key, config, types)
