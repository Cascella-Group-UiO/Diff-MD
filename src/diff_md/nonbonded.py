from functools import partial
from math import comb, factorial
from typing import Tuple

import itertools
import numpy as np
import jax.numpy as jnp
from jax import Array, checkpoint, custom_vjp, grad, jit, lax, vmap, value_and_grad

from .config import Config

from .neighbor_list import apply_cutoff, apply_cutoff_elec


# ---------------------------------------------------------------------------
# Cardinal B-spline interpolation (generalises CIC to arbitrary order)
# ---------------------------------------------------------------------------

def _bspline_connection(order: int) -> np.ndarray:
    """Precompute the 3-D stencil offsets for a B-spline of given *order*.

    For order *p* the stencil in each dimension is
    ``[-(p//2 - 1), ..., p - 1 - (p//2 - 1)]``.
    The returned array has shape ``(1, p**3, 3)`` so it can be broadcast
    over the particle dimension.
    """
    shift = order // 2 - 1
    offsets_1d = list(range(-shift, order - shift))
    return np.array(
        list(itertools.product(offsets_1d, repeat=3)),
        dtype=np.int32,
    ).reshape(1, order**3, 3)


def _bspline_binom(order: int) -> np.ndarray:
    """Binomial coefficients C(order, k) for k = 0 … order (float64)."""
    from math import comb
    return np.array([comb(order, k) for k in range(order + 1)], dtype=np.float64)


def _cardinal_bspline_weights_1d(frac: Array, order: int, binom: Array) -> Array:
    r"""Evaluate :math:`M_p(\text{frac} + (p-1) - j)` for j = 0 … p-1.

    Uses the closed-form

    .. math::
        M_p(x) = \frac{1}{(p-1)!}\sum_{k=0}^{p}(-1)^k\binom{p}{k}(x-k)_+^{p-1}

    Parameters
    ----------
    frac : Array, shape (N,)
        Fractional mesh coordinate, ``u - floor(u)``.
    order : int
        B-spline order (2 = CIC, 3 = TSC, 4 = cubic, …).
    binom : Array, shape (order+1,)
        Pre-computed binomial coefficients ``C(order, k)``.

    Returns
    -------
    weights : Array, shape (N, order)
        One weight per stencil point.
    """
    p = order
    pm1_fac = factorial(p - 1)
    k = jnp.arange(p + 1, dtype=frac.dtype)          # (p+1,)
    signs = ((-1.0) ** k)                              # (p+1,)
    binom_f = binom.astype(frac.dtype)                 # (p+1,)

    j = jnp.arange(p, dtype=frac.dtype)                # (p,)
    # x_j = frac + (p-1) - j   — the argument into M_p for stencil slot j
    x = frac[:, None] + (p - 1) - j[None, :]          # (N, p)

    # M_p(x) = (1/(p-1)!) * sum_k (-1)^k C(p,k) (x-k)_+^{p-1}
    # Shape: (N, p, p+1)
    terms = signs[None, None, :] * binom_f[None, None, :] * jnp.maximum(
        x[:, :, None] - k[None, None, :], 0.0
    ) ** (p - 1)
    return jnp.sum(terms, axis=-1) / pm1_fac           # (N, p)


def _cardinal_bspline_weights_deriv_1d(
    frac: Array, order: int, binom_pm1: Array,
) -> Array:
    r"""Derivative of :func:`_cardinal_bspline_weights_1d` w.r.t. ``frac``.

    Uses the standard cardinal B-spline differentiation identity

    .. math::
        \frac{d}{dx} M_p(x) = M_{p-1}(x) - M_{p-1}(x - 1).

    Returned shape is ``(N, p)``, matching the value weights.

    Parameters
    ----------
    frac : Array, shape (N,)
    order : int
        Order ``p`` of the original B-spline (>= 2).
    binom_pm1 : Array, shape (order,)
        Pre-computed binomial coefficients ``C(p-1, k)`` for ``k = 0 ... p-1``.
    """
    p = order
    if p < 2:
        return jnp.zeros((frac.shape[0], p), dtype=frac.dtype)

    pm2_fac = factorial(p - 2)
    k = jnp.arange(p, dtype=frac.dtype)              # (p,)
    signs = ((-1.0) ** k)                              # (p,)
    binom_f = binom_pm1.astype(frac.dtype)             # (p,)

    j = jnp.arange(p, dtype=frac.dtype)                # (p,)
    x = frac[:, None] + (p - 1) - j[None, :]          # (N, p)

    # Truncated power (x-k)_+^{p-2}: must be 0 when x-k <= 0. Naive
    # ``max(x-k, 0)^(p-2)`` breaks for p == 2 because 0^0 = 1; switching
    # to a strict-positivity guard fixes the order-2 (CIC) backward.
    xk = x[:, :, None] - k[None, None, :]
    pow_x = jnp.where(xk > 0, xk ** (p - 2), 0.0)
    M_p1_x = jnp.sum(signs[None, None, :] * binom_f[None, None, :] * pow_x,
                     axis=-1) / pm2_fac

    x1k = (x - 1.0)[:, :, None] - k[None, None, :]
    pow_x1 = jnp.where(x1k > 0, x1k ** (p - 2), 0.0)
    M_p1_x1 = jnp.sum(signs[None, None, :] * binom_f[None, None, :] * pow_x1,
                      axis=-1) / pm2_fac

    return M_p1_x - M_p1_x1                            # (N, p)


def get_bspline_kernel(
    positions: Array, config: Config, mass: float | Array = 1.0,
) -> Tuple[Array, Array]:
    """Generalised kernel for B-spline of order ``config.pme_order``.

    Returns
    -------
    kernel : Array, shape (N, pme_order**3)
    neighbour_coords : Array, shape (N, pme_order**3, 3)
    """
    order = config.pme_order
    scale = config.mesh_size / config.box_size
    u = scale * positions                               # (N, 3)

    floor_u = jnp.floor(u)                              # (N, 3)
    frac = u - floor_u                                  # (N, 3)

    # Pre-computed binomial coefficients (static w.r.t. JIT)
    binom = jnp.array(_bspline_binom(order), dtype=frac.dtype)

    # 1-D weights for each axis: each is (N, order)
    wx = _cardinal_bspline_weights_1d(frac[:, 0], order, binom)
    wy = _cardinal_bspline_weights_1d(frac[:, 1], order, binom)
    wz = _cardinal_bspline_weights_1d(frac[:, 2], order, binom)

    # 3-D weight = product of 1-D weights along the three axes
    # connection has shape (1, order**3, 3): each row is (ox, oy, oz)
    connection = jnp.array(_bspline_connection(order), dtype=jnp.int32)
    # Map offset indices 0…order-1 to weight arrays
    shift = order // 2 - 1
    # ox, oy, oz each in range [-shift, order-1-shift]
    # index into weight array: ox + shift  (so 0-based)
    oi = connection[0, :, 0] + shift   # (order**3,)
    oj = connection[0, :, 1] + shift
    ok = connection[0, :, 2] + shift
    kernel = (mass *
              wx[:, oi] * wy[:, oj] * wz[:, ok])       # (N, order**3)

    # Grid coordinates with PBC
    neighbour_coords = floor_u[:, None, :] + connection  # (N, order**3, 3)
    neighbour_coords = jnp.mod(neighbour_coords, config.mesh_size)
    return kernel, neighbour_coords


@jit
def bspline_paint(
    positions: Array, config: Config, mass: float | Array = 1.0,
) -> Array:
    """Paint charges/masses into the mesh using B-spline of order ``config.pme_order``."""
    order = config.pme_order
    kernel, neighbour_coords = get_bspline_kernel(positions, config, mass)
    n_stencil = order ** 3

    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    mesh = lax.scatter_add(
        jnp.asarray(config.empty_mesh),
        neighbour_coords.reshape([-1, n_stencil, 3]).astype("int32"),
        kernel.reshape([-1, n_stencil]),
        dnums,
    )
    return mesh


@jit
def bspline_readout(
    positions: Array, from_mesh: Array, config: Config,
    mass: float | Array = 1.0,
) -> Array:
    """Read values from a (3, *mesh) arrey using B-spline interpolation."""
    order = config.pme_order
    kernel, neighbour_coords = get_bspline_kernel(positions, config, mass)
    n_stencil = order ** 3

    dnums = lax.GatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1, 2, 3),
        start_index_map=(1, 2, 3),
    )
    mesh_vals = lax.gather(
        from_mesh,
        neighbour_coords.reshape([-1, n_stencil, 3]).astype("int32"),
        dnums,
        (3, 1, 1, 1),
    )
    weighted_vals = mesh_vals * kernel.reshape([-1, n_stencil])
    value = jnp.sum(weighted_vals, axis=-1)
    return value.T


def pme_paint(positions: Array, config: Config, mass: float | Array = 1.0) -> Array:
    """Dispatch to CIC or B-spline paint based on ``config.pme_order``."""
    if config.pme_order == 2:
        return cic_paint(positions, config, mass)
    return bspline_paint(positions, config, mass)


def pme_readout(
    positions: Array, from_mesh: Array, config: Config,
    mass: float | Array = 1.0,
) -> Array:
    """Dispatch to CIC or B-spline readout based on ``config.pme_order``."""
    if config.pme_order == 2:
        return cic_readout(positions, from_mesh, config, mass)
    return bspline_readout(positions, from_mesh, config, mass)



@jit
def LJ_energy(r_vec, s, e):
    r = jnp.linalg.norm(r_vec)
    r6 = (s/r)**6
    return 4.0 * e * r6 * (r6 - 1.0)


@jit
def LJ_energy_r(r, s, e):
    r6 = (s/r)**6
    return 4.0 * e * r6 * (r6 - 1.0)


_dLJ_dr = jit(grad(LJ_energy_r, argnums=0))


@jit
def LJ_energy_force_shifted(r_vec, s, e, rlj):
    r = jnp.linalg.norm(r_vec)
    v_r = LJ_energy_r(r, s, e)
    v_rc = LJ_energy_r(rlj, s, e)
    dv_rc = _dLJ_dr(rlj, s, e)
    return (v_r - v_rc) - (r - rlj) * dv_rc


@jit
def get_LJ_energy_and_forces(
    forces: Array,
    pair_params: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config
) -> Tuple[float, Array]:
    
    forces = jnp.zeros_like(forces)

    rlj = config.rlj #+ 5e-7

    r_vec, r, neigh_i, neigh_j, _, _, s_ij, e_ij = pair_params
    r_vec, s_ij, e_ij, neigh_i, neigh_j = apply_cutoff(r_vec, r, s_ij, e_ij, neigh_i, neigh_j, rlj)

    if config.lj_force_shift:
        rlj_arr = jnp.full_like(s_ij, rlj)
        LJ_grad = vmap(value_and_grad(LJ_energy_force_shifted), (0, 0, 0, 0))
        energies, grads = LJ_grad(r_vec, s_ij, e_ij, rlj_arr)
    else:
        LJ_grad = vmap(value_and_grad(LJ_energy), (0, 0, 0))
        energies, grads = LJ_grad(r_vec, s_ij, e_ij)
        e_cut = LJ_energy_r(rlj, s_ij, e_ij)
        energies = energies - e_cut

    grads = jnp.nan_to_num(grads, nan=0.0)
    forces = forces.at[neigh_i].add(-grads)
    forces = forces.at[neigh_j].add(grads)

    energy = jnp.nan_to_num(energies, nan=0.0)

    return jnp.sum(energy), forces


@jit
def get_LJ_energy_and_forces_npt(
    forces,
    pair_params: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config
) -> Tuple[float, Array]:

    num_particles = config.n_particles
    forces = jnp.zeros((num_particles, 3))
    energy = 0.0

    r_vec, r, neigh_i, neigh_j, _, _, s_ij, e_ij = pair_params
    r_vec, s_ij, e_ij, neigh_i, neigh_j = apply_cutoff(r_vec, r, s_ij, e_ij, neigh_i, neigh_j, config.rlj)

    if config.lj_force_shift:
        rlj_arr = jnp.full_like(s_ij, config.rlj)
        LJ_grad = vmap(value_and_grad(LJ_energy_force_shifted), (0, 0, 0, 0))
        energies, grads = LJ_grad(r_vec, s_ij, e_ij, rlj_arr)
    else:
        LJ_grad = vmap(value_and_grad(LJ_energy), (0, 0, 0))
        energies, grads = LJ_grad(r_vec, s_ij, e_ij)
        e_cut = LJ_energy_r(config.rlj, s_ij, e_ij)
        energies = energies - e_cut

    grads = jnp.nan_to_num(grads, nan=0.0)
    forces = forces.at[neigh_i].add(-grads)
    forces = forces.at[neigh_j].add(grads)

    energy = jnp.nan_to_num(energies, nan=0.0)

    pressure = jnp.sum(-grads * r_vec, axis=0)

    return jnp.sum(energy), forces, pressure


@jit
def get_kernel(
    positions: Array, config: Config, mass: float | Array = 1.0
) -> Tuple[Array, Array]:
    scale = config.mesh_size / config.box_size
    positions = scale * positions

    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)

    # fmt: off
    connection = jnp.array(
        [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]]
    )
    # fmt: on

    neighboor_coords = floor + connection
    kernel = 1.0 - jnp.abs(positions - neighboor_coords)
    kernel = mass * kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    # Add PBC
    neighboor_coords = jnp.mod(neighboor_coords, config.mesh_size)
    return kernel, neighboor_coords


@jit
def cic_paint(positions: Array, config: Config, mass: float | Array = 1.0) -> Array:
    kernel, neighboor_coords = get_kernel(positions, config, mass)

    # The code below does:
    # for n in range(len(positions)):
    #     for grid_point, value in zip(neighboor_coords[n], kernel[n]):
    #         idx = tuple(grid_point)
    #         mesh = mesh.at[idx].add(value)

    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )

    mesh = lax.scatter_add(
        jnp.asarray(config.empty_mesh),  # jnp.zeros(mesh_size)
        neighboor_coords.reshape([-1, 8, 3]).astype("int32"),
        kernel.reshape([-1, 8]),
        dnums,
    )
    return mesh


@jit
def cic_readout(
    positions: Array, from_mesh: Array, config: Config, mass: float | Array = 1.0
) -> Array:
    # 3D readout function
    # input from_mesh.shape == (3, *mesh_size)
    kernel, neighboor_coords = get_kernel(positions, config, mass)

    # The code below does:
    # value = jnp.zeros_like(positions)
    # for n in range(len(positions)):
    #    for grid_point, v in zip(neighboor_coords[n], kernel[n]):
    #        idx = tuple(grid_point)
    #        value = value.at[n, 0].add(v * from_mesh[0][idx])
    #        value = value.at[n, 1].add(v * from_mesh[1][idx])
    #        value = value.at[n, 2].add(v * from_mesh[2][idx])

    dnums = lax.GatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1, 2, 3),
        start_index_map=(1, 2, 3),
    )

    mesh_vals = lax.gather(
        from_mesh,
        neighboor_coords.reshape([-1, 8, 3]).astype("int32"),
        dnums,
        (3, 1, 1, 1),
    )

    weighted_vals = mesh_vals * kernel.reshape([-1, 8])
    value = jnp.sum(weighted_vals, axis=-1)
    return value.T


_bspline_deconv_cache: dict = {}


def _bspline_integer_values(order: int) -> np.ndarray:
    """Return ``M_p(1), ..., M_p(p-1)`` for a cardinal B-spline of order ``p``."""
    values = []
    for x_value in np.arange(1, order, dtype=np.float64):
        total = 0.0
        for k in range(order + 1):
            total += (
                ((-1.0) ** k)
                * comb(order, k)
                * max(float(x_value) - k, 0.0) ** (order - 1)
            )
        values.append(total / factorial(order - 1))
    return np.asarray(values, dtype=np.float64)

def _bspline_deconv_1d(M: int, order: int, rfft: bool = False) -> Array:
    r"""Compute the 1-D SPME B-spline deconvolution factor for mesh size *M*.

    Uses the discrete cardinal B-spline modulus used by GROMACS/SPME,
    ``|sum_k M_p(k+1) exp(2*pi*i*m*k/M)|^-2``.  The global phase present in
    some SPME derivations cancels in the modulus.  When *rfft* is True the
    array covers indices ``0 ... M//2`` (matching ``jnp.fft.rfftfreq``).

    Results are cached as numpy arrays to avoid JAX tracker leaks across
    different JIT trace contexts.
    """
    key = (M, order, rfft)
    if key not in _bspline_deconv_cache:
        if rfft:
            m = np.arange(M // 2 + 1, dtype=np.float64)
        else:
            m = np.fft.fftfreq(M) * M
        values = _bspline_integer_values(order)
        offsets = np.arange(order - 1, dtype=np.float64)
        phase = np.exp(2j * np.pi * m[:, None] * offsets[None, :] / M)
        modulus = np.abs(np.sum(values[None, :] * phase, axis=1)) ** 2
        # Even orders (the practical p=4/6/8 case) have a strictly positive
        # modulus floor (>= 1.7e-2), so the threshold never triggers. Odd
        # orders (p=3,5,7) have an exact zero at the Nyquist frequency
        # m = M/2; clamping there to 1.0 matches GROMACS' bsp_mod safeguard.
        _bspline_deconv_cache[key] = np.where(modulus < 1e-14, 1.0, 1.0 / modulus)
    return jnp.array(_bspline_deconv_cache[key])


@jit
def filter_density(phi: Array, config: Config) -> Array:
    phi_rescale = phi / config.volume_per_cell
    phi_fourier = jnp.fft.rfftn(phi_rescale, norm="forward")
    window = config.window()

    order = config.pme_order
    if order > 2:
        # Apply B-spline deconvolution factor  1 / |b(k)|^{2}
        Mx = int(config.empty_mesh.shape[0])
        My = int(config.empty_mesh.shape[1])
        Mz = int(config.empty_mesh.shape[2])
        dx = _bspline_deconv_1d(Mx, order, rfft=False)  # (Mx,)
        dy = _bspline_deconv_1d(My, order, rfft=False)  # (My,)
        dz = _bspline_deconv_1d(Mz, order, rfft=True)   # (Mz//2+1,)
        deconv = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        window = window * deconv

    return phi_fourier * window


@jit
def reaction_field_potential(
    r_vec: Array,
    erf: float,
    er: float,
    rc: float,
    f: float
) -> Array:

    r = jnp.linalg.norm(r_vec, axis=0)
    krf = (erf - er) / (2*erf + er) / rc**3
    crf = 1/rc + krf*rc**2

    return f * (1/r - crf + krf*r**2)


@jit
def rf_potential_excluded_pairs(
    r_vec: Array,
    erf: float,
    er: float,
    rc: float,
    f: float
) -> Array:

    r = jnp.linalg.norm(r_vec, axis=0)
    krf = (erf - er) / (2*erf + er) / rc**3
    crf = 1/rc + krf*rc**2

    return f * (- crf + krf*r**2)


@jit
def get_rf_excluded_pairs_energy_and_forces(
    excl_pair_param,
    config,
    forces,
    potentials,
    energy
) -> Tuple[float, float, Array]:
    r_vec, _, neigh_i, neigh_j, q_i, q_j, _, _ = excl_pair_param

    func = vmap(value_and_grad(rf_potential_excluded_pairs), (0, None, None, None, None))
    phi_contributions, grads = func(r_vec, config.epsilon_rf, config.dielectric_const, config.rc, config.elec_conversion)

    # Forces
    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i_col = q_i.reshape(-1, 1)
    q_j_col = q_j.reshape(-1, 1)
    forces = forces.at[neigh_i].add(-grads*q_i_col*q_j_col)
    forces = forces.at[neigh_j].add(grads*q_i_col*q_j_col)

    q_i = jnp.squeeze(q_i)
    q_j = jnp.squeeze(q_j)

    # Potential
    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy += jnp.sum(phi_contributions*q_j*q_i)

    return energy, potential, forces, grads, r_vec, jnp.expand_dims(q_i, 1), jnp.expand_dims(q_j, 1)


@jit
def get_reaction_field_energy_and_forces(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[float, float, Array]:

    forces = jnp.zeros((config.n_particles, 3))
    potentials = jnp.zeros((config.n_particles))

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, r, q_i, q_j, neigh_i, neigh_j =  apply_cutoff_elec(r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc)

    potential = vmap(value_and_grad(reaction_field_potential), (0, None, None, None, None))
    phi_contributions, grads = potential(r_vec, config.epsilon_rf, config.dielectric_const, config.rc, config.elec_conversion)

    # Potential
    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy = jnp.sum(phi_contributions*q_j*q_i)

    # Forces
    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    if excl_pair_param is not None:
      energy, potential, forces, _, _, _, _ = get_rf_excluded_pairs_energy_and_forces(
          excl_pair_param,
          config,
          forces,
          potentials,
          energy,
      )

    return energy-config.self_energy, potential, forces


@jit
def get_reaction_field_pair_energy_and_forces(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    scale: float = 1.0,
) -> Tuple[float, float, Array]:

    forces = jnp.zeros((config.n_particles, 3))
    potentials = jnp.zeros((config.n_particles))

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, _, q_i, q_j, neigh_i, neigh_j = apply_cutoff_elec(
        r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc
    )

    potential = vmap(
        value_and_grad(reaction_field_potential), (0, None, None, None, None)
    )
    phi_contributions, grads = potential(
        r_vec,
        config.epsilon_rf,
        config.dielectric_const,
        config.rc,
        config.elec_conversion,
    )

    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(scale * phi_contributions * q_j)
    potentials = potentials.at[neigh_j].add(scale * phi_contributions * q_i)
    potential = jnp.sum(potentials)

    energy = scale * jnp.sum(phi_contributions * q_j * q_i)

    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-scale * grads * q_i * q_j)
    forces = forces.at[neigh_j].add(scale * grads * q_i * q_j)

    return energy, potential, forces


@jit
def get_rf_pressure(
    energy: float,
    grads: Array,
    r_vec: Array,
    q_i: Array,
    q_j: Array,
    potentials: Array,
    forces: Array,
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
):
    """Calculate internal virial contribution from forces."""

    pressure = jnp.sum(-grads*q_i*q_j * r_vec, axis=0)

    if excl_pair_param is not None:
      energy, potential, forces, grads, r_vec, q_i, q_j = get_rf_excluded_pairs_energy_and_forces(
          excl_pair_param,
          config,
          forces,
          potentials,
          energy,
      )
      pressure += jnp.sum(-grads*q_i*q_j * r_vec, axis=0)

    return energy, potential, forces, pressure


@jit
def get_reaction_field_energy_and_forces_npt(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[float, float, Array]:

    energy = 0.0
    forces = forces.at[...].set(0.0) # TODO: Test if this is slow
    potentials = jnp.zeros((config.n_particles))

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, r, q_i, q_j, neigh_i, neigh_j = apply_cutoff_elec(r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc)

    func_grad = vmap(value_and_grad(reaction_field_potential), (0, None, None, None, None))
    phi_contributions, grads = func_grad(r_vec, config.epsilon_rf, config.dielectric_const, config.rc, config.elec_conversion)

    # Potential
    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy = jnp.sum(phi_contributions*q_j*q_i)

    # Forces
    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    # Add pressure contribution
    energy, potential, forces, pressure = get_rf_pressure(
        energy,
        grads,
        r_vec,
        q_i, q_j,
        potentials,
        forces,
        config,
        excl_pair_param
    )

    return energy-config.self_energy, potential, forces, pressure


@jit
def get_reaction_field_pair_energy_and_forces_npt(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    scale: float = 1.0,
) -> Tuple[float, float, Array, Array]:

    forces = forces.at[...].set(0.0)
    potentials = jnp.zeros((config.n_particles))

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, _, q_i, q_j, neigh_i, neigh_j = apply_cutoff_elec(
        r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc
    )

    func_grad = vmap(
        value_and_grad(reaction_field_potential), (0, None, None, None, None)
    )
    phi_contributions, grads = func_grad(
        r_vec,
        config.epsilon_rf,
        config.dielectric_const,
        config.rc,
        config.elec_conversion,
    )

    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(scale * phi_contributions * q_j)
    potentials = potentials.at[neigh_j].add(scale * phi_contributions * q_i)
    potential = jnp.sum(potentials)

    energy = scale * jnp.sum(phi_contributions * q_j * q_i)

    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-scale * grads * q_i * q_j)
    forces = forces.at[neigh_j].add(scale * grads * q_i * q_j)

    pressure = jnp.sum(-scale * grads * q_i * q_j * r_vec, axis=0)

    return energy, potential, forces, pressure


@jit
def coulomb_pair_potential(
    r_vec: Array,
    f: float,
) -> Array:
    r = jnp.linalg.norm(r_vec, axis=0)
    return f / r


@jit
def get_coulomb_pair_energy_and_forces(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    scale: float = 1.0,
    elec_conversion: float = 1.0,
) -> Tuple[float, Array]:

    forces = forces.at[...].set(0.0)

    r_vec, _, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param

    potential = vmap(value_and_grad(coulomb_pair_potential), (0, None))
    phi_contributions, grads = potential(r_vec, elec_conversion)

    phi_contributions = jnp.nan_to_num(phi_contributions, nan=0.0)
    grads = jnp.nan_to_num(grads, nan=0.0)

    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)

    energy = scale * jnp.sum(phi_contributions * q_j.squeeze() * q_i.squeeze())
    forces = forces.at[neigh_i].add(-scale * grads * q_i * q_j)
    forces = forces.at[neigh_j].add(scale * grads * q_i * q_j)

    return energy, forces


@jit
def get_coulomb_pair_energy_and_forces_npt(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    scale: float = 1.0,
    elec_conversion: float = 1.0,
) -> Tuple[float, Array, Array]:

    forces = forces.at[...].set(0.0)

    r_vec, _, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param

    potential = vmap(value_and_grad(coulomb_pair_potential), (0, None))
    phi_contributions, grads = potential(r_vec, elec_conversion)

    phi_contributions = jnp.nan_to_num(phi_contributions, nan=0.0)
    grads = jnp.nan_to_num(grads, nan=0.0)

    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)

    energy = scale * jnp.sum(phi_contributions * q_j.squeeze() * q_i.squeeze())
    forces = forces.at[neigh_i].add(-scale * grads * q_i * q_j)
    forces = forces.at[neigh_j].add(scale * grads * q_i * q_j)
    pressure = jnp.sum(-scale * grads * q_i * q_j * r_vec, axis=0)

    return energy, forces, pressure


@jit
def ewald_rec_intramol(
    r_vec: Array,
    alpha: float,
    f: float,
):
    r = jnp.linalg.norm(r_vec, axis=0)
    return - f * lax.erf(jnp.sqrt(alpha) * r) / r


@jit
def ewald_masked_pairs(
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    forces: Array,
    potentials: Array,
    energy: float,
) -> Tuple[float, float, Array]:

    r_vec, _, neigh_i, neigh_j, q_i, q_j, _, _ = excl_pair_param
    q_i = jnp.ravel(q_i)
    q_j = jnp.ravel(q_j)

    func = vmap(value_and_grad(ewald_rec_intramol), (0, None, None))
    phi_contributions, grads = func(r_vec, 1/(2*config.sigma**2), config.elec_conversion)

    # Forces
    grads = jnp.nan_to_num(grads, nan=0.0)
    # grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    qij = (q_i * q_j).reshape(-1, 1)
    forces = forces.at[neigh_i].add(-grads * qij)
    forces = forces.at[neigh_j].add(grads * qij)

    # Potential
    # phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    phi_contributions = jnp.nan_to_num(phi_contributions, nan=0.0)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy += jnp.sum(phi_contributions * q_j * q_i)

    return energy, potential, forces, grads, r_vec, q_i, q_j


@jit
def phi_real_space(
    r_vec: Array,
    alpha: float,
    f: float,
):
    r = jnp.linalg.norm(r_vec, axis=0)
    return f * lax.erfc(jnp.sqrt(alpha) * r) / r


@jit
def phi_real_space_value_and_grad(
    r_vec: Array,
    alpha: float,
    f: float,
) -> Tuple[Array, Array]:
    """Hand-vectorized closed form of (phi, dphi/dr_vec) for the Ewald
    real-space potential ``phi(r_vec) = f * erfc(sqrt(alpha)*r) / r``,
    where ``r = |r_vec|``.

    Equivalent to ``vmap(value_and_grad(phi_real_space), (0, None, None))``
    but eliminates the per-pair AD trace and saves the corresponding
    residuals on reverse-mode passes.

    Accepts ``r_vec`` of shape ``(num_pairs, 3)`` (the production usage)
    or ``(3,)`` (single-pair tests); the operations broadcast on the last
    axis.
    """
    r2 = jnp.sum(r_vec * r_vec, axis=-1)
    r = jnp.sqrt(r2)
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r

    sqrt_alpha = jnp.sqrt(alpha)
    e_alpha_r2 = jnp.exp(-alpha * r2)
    erfc_term = lax.erfc(sqrt_alpha * r)

    phi = f * erfc_term * inv_r

    # dphi/dr_vec = -f * (2*sqrt(alpha/pi)*exp(-alpha*r^2) + erfc(sqrt(alpha)*r)/r)
    #              * r_vec / r^2
    two_sqrt_alpha_over_sqrt_pi = 2.0 * sqrt_alpha / jnp.sqrt(jnp.pi)
    bracket = two_sqrt_alpha_over_sqrt_pi * e_alpha_r2 + erfc_term * inv_r
    grad_phi = (-f * bracket * inv_r2)[..., None] * r_vec

    return phi, grad_phi


@jit
def ewald_real_space(
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[float, Array]:

    alpha = 1/(2*config.sigma**2)

    num_particles = config.n_particles
    energy = 0.0
    forces = jnp.zeros((num_particles, 3))
    potentials = jnp.zeros(num_particles)

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, r, q_i, q_j, neigh_i, neigh_j = apply_cutoff_elec(r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc)

    phi_contributions, grads = phi_real_space_value_and_grad(
        r_vec, alpha, config.elec_conversion,
    )

    # Potential
    phi_contributions = jnp.nan_to_num(phi_contributions, nan=0.0)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy = jnp.sum(phi_contributions*q_j*q_i)

    # Forces
    grads = jnp.nan_to_num(grads, nan=0.0)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    if excl_pair_param is not None:
        energy, potential, forces, _, _, _, _ = ewald_masked_pairs(
            excl_pair_param,
            config,
            forces,
            potentials,
            energy,
        )

    return energy, forces, potential


@jit
def ewald_real_space_npt(
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[float, Array]:

    alpha = 1/(2*config.sigma**2)

    num_particles = config.n_particles
    energy = 0.0
    potential = 0.0
    forces = jnp.zeros((num_particles, 3))
    potentials = jnp.zeros(num_particles)

    r_vec, r, neigh_i, neigh_j, q_i, q_j, _, _ = elec_param
    r_vec, r, q_i, q_j, neigh_i, neigh_j = apply_cutoff_elec(r_vec, r, q_i, q_j, neigh_i, neigh_j, config.rc)

    phi_contributions, grads = phi_real_space_value_and_grad(
        r_vec, alpha, config.elec_conversion,
    )

    # Potential
    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy = jnp.sum(phi_contributions*q_j*q_i)

    # Forces
    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    q_i = q_i.reshape(len(q_i), 1)
    q_j = q_j.reshape(len(q_j), 1)
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    # Pressure
    pressure = jnp.sum(-grads*q_i*q_j * r_vec, axis=0)

    if excl_pair_param is not None:
        energy, potential, forces, grads, r_vec, q_i, q_j = ewald_masked_pairs(
            excl_pair_param,
            config,
            forces,
            potentials,
            energy,
        )
        q_i = q_i.reshape(-1, 1)
        q_j = q_j.reshape(-1, 1)
        pressure += jnp.sum(-grads*q_i*q_j * r_vec, axis=0)

    return energy, forces, potential, pressure


@jit
def get_elec_potential_and_energy(
    phi_q: Array, phi_q_fourier: Array, config: Config
) -> Tuple[Array, Array]:
    elec_potential_fourier = config.elec_const * phi_q_fourier / config.knorm()
    elec_potential = jnp.fft.irfftn(elec_potential_fourier, norm="forward")

    long_range_energy = 0.5 * jnp.sum(phi_q * elec_potential)
    elec_energy = long_range_energy - config.self_energy
    return elec_potential, elec_energy, long_range_energy


def _recip_pos_grad(positions: Array, pot: Array, charges: Array,
                    config: Config) -> Array:
    """Return ``dE_recip/dr_i`` via the structured B-spline derivative gather.

    For ``E = 0.5 * sum_g phi_q[g] * pot[g]`` and B-spline-painted
    ``phi_q[g] = sum_i Q_i K(r_i - r_g)``,

        dE/dr_i,d = scale_d * Q_i * sum_xi K_d(i, xi) * pot[neighbour_coords[i, xi]]

    where ``K_d`` is the tensor-product B-spline weight with the value
    weights along axis ``d`` replaced by the analytic 1-D derivative.

    The unified order >= 2 path covers CIC (``order == 2``) and higher-order
    B-splines identically: ``test_bspline_pme.TestBSplineOrder2MatchesCIC``
    pins the equivalence for ``order == 2``.
    """
    order = config.pme_order
    scale = config.mesh_size / config.box_size                # (3,)
    u = scale * positions                                     # (N, 3)
    floor_u = jnp.floor(u)
    frac = u - floor_u                                        # (N, 3)

    binom_p = jnp.array(_bspline_binom(order), dtype=frac.dtype)
    from math import comb as _comb
    binom_pm1 = jnp.array(
        [_comb(order - 1, kk) for kk in range(order)], dtype=frac.dtype,
    )

    wx = _cardinal_bspline_weights_1d(frac[:, 0], order, binom_p)
    wy = _cardinal_bspline_weights_1d(frac[:, 1], order, binom_p)
    wz = _cardinal_bspline_weights_1d(frac[:, 2], order, binom_p)
    dwx = _cardinal_bspline_weights_deriv_1d(frac[:, 0], order, binom_pm1)
    dwy = _cardinal_bspline_weights_deriv_1d(frac[:, 1], order, binom_pm1)
    dwz = _cardinal_bspline_weights_deriv_1d(frac[:, 2], order, binom_pm1)

    connection = jnp.array(_bspline_connection(order), dtype=jnp.int32)
    shift = order // 2 - 1
    oi = connection[0, :, 0] + shift
    oj = connection[0, :, 1] + shift
    ok = connection[0, :, 2] + shift

    neighbour_coords = floor_u[:, None, :] + connection                 # (N, p**3, 3)
    neighbour_coords = jnp.mod(neighbour_coords, config.mesh_size).astype(jnp.int32)

    pot_vals = pot[neighbour_coords[..., 0],
                   neighbour_coords[..., 1],
                   neighbour_coords[..., 2]]                             # (N, p**3)

    kx = dwx[:, oi] * wy[:, oj] * wz[:, ok]
    ky = wx[:, oi] * dwy[:, oj] * wz[:, ok]
    kz = wx[:, oi] * wy[:, oj] * dwz[:, ok]

    mass = charges.reshape(-1)                                           # (N,)
    gx = scale[0] * mass * jnp.sum(pot_vals * kx, axis=-1)
    gy = scale[1] * mass * jnp.sum(pot_vals * ky, axis=-1)
    gz = scale[2] * mass * jnp.sum(pot_vals * kz, axis=-1)

    return jnp.stack([gx, gy, gz], axis=-1)                              # (N, 3)


def _make_recip_energy_pos_custom_vjp(charges: Array, config: Config):
    """Build a ``positions -> energy`` function with a hand-written VJP.

    ``charges`` and ``config`` are captured by closure so that
    ``custom_vjp`` only sees ``positions`` as the differentiable argument.
    JAX's ``nondiff_argnums`` cannot accept tracers (charges, config leaves),
    which is why this is built per call rather than at module level. The
    closure is created inside a jitted wrapper, so re-creation only happens
    during tracing and incurs zero runtime cost.

    ``jax.grad`` w.r.t. ``positions`` then uses the analytic structured gather
    in ``_recip_pos_grad`` instead of retaping paint -> rfftn -> multiply ->
    irfftn -> sum.
    """

    @custom_vjp
    def _recip_E(positions):
        phi_q = pme_paint(positions, config, mass=charges)
        phi_q_fourier = filter_density(phi_q, config)
        _, e, _ = get_elec_potential_and_energy(phi_q, phi_q_fourier, config)
        return e

    def _fwd(positions):
        phi_q = pme_paint(positions, config, mass=charges)
        phi_q_fourier = filter_density(phi_q, config)
        pot, e, _ = get_elec_potential_and_energy(phi_q, phi_q_fourier, config)
        return e, (positions, pot)

    def _bwd(residuals, output_cotangent):
        positions, pot = residuals
        grad_pos = _recip_pos_grad(positions, pot, charges, config)
        return (output_cotangent * grad_pos,)

    _recip_E.defvjp(_fwd, _bwd)
    return _recip_E


@partial(jit, static_argnums=(5,))
def get_elec_energy_potential_and_forces(
    positions: Array,
    charges: Array,
    config: Config,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array],
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    compute_potential: bool = True,
) -> Tuple[Array, Array, Array]:
    # Reciprocal energy + forces. ``_recip_E`` carries an analytic
    # ``custom_vjp``; ``value_and_grad`` triggers it instead of retaping
    # the paint/FFT/irfftn chain on every backward pass.
    _recip_E = _make_recip_energy_pos_custom_vjp(charges, config)
    recip_energy, recip_forces = value_and_grad(
        lambda pos: -_recip_E(pos)
    )(positions)
    recip_energy = -recip_energy  # undo the sign flip used for grad->force

    # Real-space Ewald contribution
    elec_energy_real, elec_forces_real, potential_real = ewald_real_space(elec_param, config, excl_pair_param)

    if compute_potential:
        # Reciprocal-space potential for diagnostics — requires a separate
        # forward pass through paint → FFT → irfftn.  Skipped during
        # training where only energy and forces feed into the loss.
        phi_q = pme_paint(positions, config, mass=charges)
        phi_q_fourier = filter_density(phi_q, config)
        recip_potential, _, _ = get_elec_potential_and_energy(
            phi_q, phi_q_fourier, config
        )
        return recip_energy+elec_energy_real, recip_potential+potential_real, recip_forces+elec_forces_real

    return recip_energy+elec_energy_real, 0.0, recip_forces+elec_forces_real


def _recip_strain_grad(phi_q: Array, pot: Array, pot_F: Array,
                       config: Config) -> Array:
    """Closed-form ``dE_recip/dscale`` evaluated at ``scale = 1``.

    The reciprocal energy ``E = 0.5 sum_g phi_q[g] * pot[g] - self_energy``
    has three scale-dependent factors hidden inside ``pot_F`` (they all
    enter ``filter_density`` and ``get_elec_potential_and_energy``):

        pot_F(s, k) = (1 / V_cell_s) * D[k] * phi_q_F_raw[k] * window_s[k] / |K_s|^2

    where ``V_cell_s = (s_x s_y s_z) V_cell_orig``, ``window_s[k] =
    exp(-0.5 sigma^2 |K_s|^2)``, and ``|K_s|^2 = sum_d K_d^2 / s_d^2``.

    Because ``phi_q`` itself is independent of scale (the cancellation in
    ``u = (mesh_size / box_s)*(scale*positions)`` makes the painted
    fractional coordinates invariant), differentiating only ``pot_F``
    yields the closed form

        dE/ds_d (s=1) = 0.5 sum_g phi_q[g] * irfftn(d(pot_F)/ds_d)[g]

    with d(log pot_F)/ds_d (at s=1) = -1 + sigma^2 K_d^2 + 2 K_d^2/|K|^2.
    The first term is axis-independent and reduces to ``-0.5 sum phi*pot
    = -long_range_energy``. The remaining two require the K-dependent
    factors and are batched into a single rank-3 irfftn.
    """
    kx, ky, kz = config.k_vector
    knorm = config.knorm()
    target_shape = config.fft_shape

    KX2 = jnp.broadcast_to(kx * kx, target_shape)
    KY2 = jnp.broadcast_to(ky * ky, target_shape)
    KZ2 = jnp.broadcast_to(kz * kz, target_shape)

    sigma2 = config.sigma * config.sigma
    # Per-axis k-dependent factor inside the irfftn:
    #   factor_d = sigma^2 K_d^2 + 2 K_d^2/|K|^2
    # (At k=0 the 2/|K|^2 hack gives 2/3 * 0 = 0, harmless because K_d^2 = 0
    # there too.)
    factor_x = sigma2 * KX2 + 2.0 * KX2 / knorm
    factor_y = sigma2 * KY2 + 2.0 * KY2 / knorm
    factor_z = sigma2 * KZ2 + 2.0 * KZ2 / knorm

    fog_F = jnp.stack([factor_x, factor_y, factor_z]) * pot_F[None, ...]
    mesh_shape = config.empty_mesh.shape
    fog_real = jnp.fft.irfftn(
        fog_F, s=mesh_shape, axes=(-3, -2, -1), norm="forward",
    )                                                            # (3, Mx, My, Mz)

    # K-dependent contribution: 0.5 * sum_g phi_q * irfftn(...)
    k_term = 0.5 * jnp.sum(phi_q[None, ...] * fog_real, axis=(1, 2, 3))   # (3,)

    # Axis-independent V_cell contribution: -0.5 * sum phi_q * pot.
    long_range_energy = 0.5 * jnp.sum(phi_q * pot)
    v_cell_term = -long_range_energy

    return v_cell_term + k_term


def _make_recip_energy_box_custom_vjp(charges: Array, config: Config):
    """Build a ``(positions, scale) -> energy`` function with a hand-written
    VJP that returns both ``dE/dpositions`` (via the structured gather from
    :func:`_recip_pos_grad`) and ``dE/dscale`` (via :func:`_recip_strain_grad`).

    The custom VJP replaces the previous AD path that traced through
    ``paint -> rfftn -> multiply -> irfftn -> sum`` and saved the full tape
    on the reverse pass. The new backward only needs ``(positions, scale,
    phi_q, pot, pot_F)`` as residuals plus a single batched rank-3 irfftn.
    """

    def _build_config_s(scale):
        new_box = scale * config.box_size
        new_vol = jnp.prod(new_box)
        new_vpc = new_vol / config.n_mesh_cells
        step = new_box / (2 * jnp.pi * config.mesh_size)
        kx = jnp.fft.fftfreq(config.empty_mesh.shape[0], step[0])
        ky = jnp.fft.fftfreq(config.empty_mesh.shape[1], step[1])
        kz = jnp.fft.rfftfreq(config.empty_mesh.shape[2], step[2])
        m_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")
        k_vector = (
            kx.reshape(config.fft_shape[0], 1, 1),
            ky.reshape(1, config.fft_shape[1], 1),
            kz.reshape(1, 1, config.fft_shape[2]),
        )
        return config.replace(
            box_size=new_box, volume=new_vol, volume_per_cell=new_vpc,
            k_vector=k_vector, k_meshgrid=m_grid,
        )

    @custom_vjp
    def _recip_E(positions, scale):
        config_s = _build_config_s(scale)
        scaled_pos = positions * scale
        phi_q = pme_paint(scaled_pos, config_s, mass=charges)
        phi_q_fourier = filter_density(phi_q, config_s)
        _, e, _ = get_elec_potential_and_energy(phi_q, phi_q_fourier, config_s)
        return e

    def _fwd(positions, scale):
        config_s = _build_config_s(scale)
        scaled_pos = positions * scale
        phi_q = pme_paint(scaled_pos, config_s, mass=charges)
        phi_q_fourier = filter_density(phi_q, config_s)
        # Inline get_elec_potential_and_energy so we keep both pot and pot_F
        # for the backward without an extra forward call.
        pot_F = config_s.elec_const * phi_q_fourier / config_s.knorm()
        pot = jnp.fft.irfftn(
            pot_F, s=config_s.empty_mesh.shape, norm="forward",
        )
        e = 0.5 * jnp.sum(phi_q * pot) - config_s.self_energy
        return e, (positions, scale, scaled_pos, phi_q, pot, pot_F, config_s)

    def _bwd(residuals, output_cotangent):
        positions, scale, scaled_pos, phi_q, pot, pot_F, config_s = residuals
        # dE/d(scaled_pos) via the structured B-spline derivative gather.
        grad_scaled_pos = _recip_pos_grad(scaled_pos, pot, charges, config_s)
        # Chain rule: scaled_pos = scale * positions => dE/dpos = scale * dE/dscaled_pos
        grad_pos = scale[None, :] * grad_scaled_pos
        # dE/dscale via the analytic Fourier-space identity.
        grad_scale = _recip_strain_grad(phi_q, pot, pot_F, config_s)
        return (output_cotangent * grad_pos, output_cotangent * grad_scale)

    _recip_E.defvjp(_fwd, _bwd)
    return _recip_E


@partial(jit, static_argnums=(5,))
def get_elec_energy_potential_and_forces_npt(
    positions: Array,
    charges: Array,
    config: Config,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array],
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    compute_potential: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """PME energy, forces and virial pressure for NPT.

    Returns (energy, potential, forces, pressure) where pressure is a
    shape-(3,) virial combining the real-space Ewald pair virial and the
    reciprocal-space strain virial (box-scaling derivative).

    The reciprocal-space pair (forces, virial) come from a hand-written
    ``custom_vjp`` (:func:`_make_recip_energy_box_custom_vjp`) that
    replaces the previous AD trace through paint/FFT/irfftn with a
    structured-gather position gradient and an analytic Fourier-space
    strain gradient.
    """
    _recip_E = _make_recip_energy_box_custom_vjp(charges, config)
    neg_e, (recip_forces, recip_virial) = value_and_grad(
        lambda p, s: -_recip_E(p, s), argnums=(0, 1),
    )(positions, jnp.ones(3))
    recip_energy = -neg_e

    # Real-space Ewald contribution (returns pair virial)
    elec_energy_real, elec_forces_real, potential_real, real_pressure = ewald_real_space_npt(
        elec_param, config, excl_pair_param
    )

    # Total electrostatic virial = real pair virial + reciprocal strain virial
    total_elec_pressure = real_pressure + recip_virial

    if compute_potential:
        phi_q = pme_paint(positions, config, mass=charges)
        phi_q_fourier = filter_density(phi_q, config)
        recip_potential, _, _ = get_elec_potential_and_energy(
            phi_q, phi_q_fourier, config
        )
        return (
            recip_energy + elec_energy_real,
            recip_potential + potential_real,
            recip_forces + elec_forces_real,
            total_elec_pressure,
        )

    return (
        recip_energy + elec_energy_real,
        0.0,
        recip_forces + elec_forces_real,
        total_elec_pressure,
    )


@partial(jit, static_argnums=(3,))
def get_dipole_forces(
    dip_positions: Array, dip_charges: Array, dip_fog: Array, n_dip: int, config: Config
) -> Tuple[Array, Array]:
    # Use ``value_and_grad`` through the reciprocal energy chain so that
    # the forces are the exact negative gradient of the energy, matching
    # the fix applied to ``get_elec_energy_potential_and_forces``.
    @checkpoint
    def _dip_recip_energy(pos):
        phi_dip = pme_paint(pos, config, mass=dip_charges)
        phi_dip_fourier = filter_density(phi_dip, config)
        _, e, _ = get_elec_potential_and_energy(
            phi_dip, phi_dip_fourier, config
        )
        return e

    dip_forces = -grad(_dip_recip_energy)(dip_positions)
    return (
        dip_forces[:n_dip] + dip_forces[n_dip:],
        dip_forces[:n_dip] - dip_forces[n_dip:],
    )
