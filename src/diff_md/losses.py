from functools import partial
from typing import Tuple

import jax.numpy as jnp
import mpi4jax
from jax import Array, jit
from jax.scipy.stats import gaussian_kde
from mpi4py import MPI

from .config import Config
from .models import GeneralModel
from .simulate import simulator


def _fill_diagonal(array, value):
    i, j = jnp.diag_indices(min(array.shape[-2:]))
    return array.at[..., i, j].set(value)


def get_chi(
    model: GeneralModel, types: Array, config: Config
) -> Tuple[Array, dict[int, Array], dict[int, Array]]:
    assert model.chi is not None, "GeneralModel.chi should not be 'None' here."

    type_mask = {}
    chi_constraint = {}
    chi = jnp.zeros((config.n_types, config.n_types))

    # Preprocessing when only specifying a subset of chi values to train
    if model.type_to_chi.ndim == 1:
        dummy_chi = config.chi
        for ttc, c in zip(model.type_to_chi, model.chi):
            if ttc in config.type_to_chi:
                dummy_chi = dummy_chi.at[ttc].set(c)

    for i, ti in enumerate(config.unique_types):
        type_mask[ti] = jnp.where(types == ti, 1.0, 0.0).reshape(-1, 1)

        if model.type_to_chi.ndim == 1:
            chi = chi.at[i].set(dummy_chi[config.type_to_chi[i]])
        else:
            chi = chi.at[i].set(model.chi[model.type_to_chi[ti, config.unique_types]])

    # Set diagonal elements to zero, to prevent learning self interactions
    if not model.self_interaction:
        chi = _fill_diagonal(chi, 0.0)

    # Parse constraints
    for ttc, val in model.chi_constraints.items():
        if ttc in config.type_to_chi:
            chi_constraint[ttc] = val

    chi = chi.reshape(config.n_types, config.n_types, 1, 1, 1)
    return chi, chi_constraint, type_mask


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


# Available metrics: mse, rmse, smape, rse
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
    return jnp.sqrt(jnp.mean((predictions - targets) ** 2), axis=axis)


@partial(jit, static_argnums=(2,))
def l2e(predictions, targets, axis=None):
    # L2 Error
    return jnp.linalg.norm(predictions - targets, axis=axis)


@jit
def harmonic_constraint(chi, k, constraints):
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
        jnp.array([(chi[ttc] - val) ** 2 for ttc, val in constraints.items()])
    )


@jit
def cubic_constraint(chi, k, constraints):
    # Here chi is a 1D array of the upper triangular portion of the full matrix
    return k * jnp.sum(
        jnp.array([jnp.abs(chi[ttc] - val) ** 3 for ttc, val in constraints.items()]),
    )


@jit
def boundary_constraint(chi, delta):
    boundary = 1 / delta
    # Here chi is a symmetric matrix, so we divide by 2 to avoid double counting
    return jnp.sum(0.5 * jnp.abs(chi * boundary) ** 3)


@jit
def lateral_density_kde(
    # fmt: off
    kde_density, centered_pos, types,
    z_range, bandwidth, bin_size, scaling_factor, config,
):
    for i, t in enumerate(config.unique_types):
        sel = jnp.where(types == t, size=config.particle_per_type[t])
        type_t_pos = centered_pos[sel]
        gaussians = gaussian_kde(type_t_pos, bw_method=bandwidth)
        kde_value = (
            gaussians(z_range) * bin_size * config.particle_per_type[t] / scaling_factor
        )
        kde_density = kde_density.at[i].add(kde_value)
    return kde_density


def density_and_apl(
    # fmt: off
    model, system, key, start_temperature, comm,
    z_range, com_type, n_lipids, target_density, target_apl, metric,           # arguments from unpacked reference dict
    density_weight=1.0, k_constraint=0.01, apl_weight=1.0, width_ratio=1.0,   # arguments from unpacked reference dict
    boundary=None, constraint=None,
):
    """Loss function for lipid membranes based on lateral density profile and area per lipid"""
    types = jnp.array(system.types)

    chi, chi_constraint, type_mask = get_chi(model, types, system.config)
    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, type_mask, system.charges,
        chi, key, system.topol, system.config, start_temperature
    )

    comm_size = comm.Get_size()
    n_bins = z_range.size
    bin_size = z_range[1] - z_range[0]
    n_frames = len(trj["positions"])

    # CHECK: skip the initial equilibration steps
    n_skip = 0
    n_frames_adj = n_frames - n_skip
    xy_apl = 0.0
    kde_density = jnp.zeros((config.n_types, n_bins))
    bandwidth = float(width_ratio * bin_size)
    z_length = z_range[-1] - z_range[0] + bin_size

    for pos, box in zip(trj["positions"][n_skip:], trj["box"][n_skip:]):
        box_x, box_y, box_z = box

        # Add frame area per lipid
        xy_area = box_x * box_y
        scaling_factor = xy_area * z_length / n_bins
        xy_apl += xy_area

        fixed_sel = jnp.where(
            types == com_type, size=config.particle_per_type[com_type]
        )
        tails = pos[fixed_sel, 2]

        com = _compute_com(tails, box_z)
        centered_pos = _center(pos[:, 2], com, box_z)

        kde_density = lateral_density_kde(
            # fmt: off
            kde_density, centered_pos, types,
            z_range, bandwidth, bin_size, scaling_factor, config,
        )

    kde_density, _ = mpi4jax.allreduce(kde_density, op=MPI.SUM, comm=comm)
    kde_density /= comm_size * n_frames_adj

    # Calculate error due to density
    error = (
        jnp.sum(density_weight * metric(kde_density, target_density, axis=1))
        / config.n_types
    )

    # Calculate error due to area per lipid
    mean_apl = 2 * xy_apl / n_lipids
    mean_apl, _ = mpi4jax.allreduce(mean_apl, op=MPI.SUM, comm=comm)
    mean_apl /= comm_size * n_frames_adj
    error += apl_weight * metric(mean_apl, target_apl)

    # Error from chi constraints
    if constraint:
        error += constraint(model.chi, k_constraint, chi_constraint)

    # Prevent chi from reaching unphysical values (hopefully)
    if boundary:
        error += boundary_constraint(chi, boundary)

    return error, (
        {"density": kde_density, "area per lipid": mean_apl},
        trj,
        key,
        config,
    )
