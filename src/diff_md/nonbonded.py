from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import Array, jit, lax, vmap, value_and_grad
from jax.typing import ArrayLike

from .config import Config

from .neighbor_list import apply_cutoff, apply_cutoff_elec



@jit
def LJ_energy(r_vec, s, e):
    r = jnp.linalg.norm(r_vec)
    r6 = (s/r)**6
    return 4.0 * e * r6 * (r6 - 1.0)


@jit
def LJ_force(r, e, s):
    r = jnp.linalg.norm(r, axis=1)
    r2 = r**2
    r6 = r**6
    s6 = s**6
    return 48 * e * (s6/r6) * (s6/r6 - 0.5) * (1 / r2)


# # Without auto diff
# @jit
# def get_LJ_energy_and_forces(
#     forces: Array,
#     pair_params: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
#     config: Config
# ) -> Tuple[float, Array]:

#     forces = forces.at[...].set(0.0)

#     r_vec, r, neigh_i, neigh_j, _, _, s_ij, e_ij = pair_params
#     r_vec, s_ij, e_ij, neigh_i, neigh_j = apply_cutoff(r_vec, r, s_ij, e_ij, neigh_i, neigh_j, config.rlj)

#     energies = LJ_energy(r_vec, s_ij, e_ij)
#     grads = LJ_force(r_vec, s_ij, e_ij)
#     # LJ_grad = vmap(value_and_grad(LJ_energy), (0, 0, 0))
#     # energies, grads = LJ_grad(r_vec, s_ij, e_ij)

#     grads = jnp.nan_to_num(grads, nan=0.0)
#     grads = jnp.reshape(len(grads), 1)
#     forces = forces.at[neigh_i].add(-grads*r_vec)
#     forces = forces.at[neigh_j].add(grads*r_vec)

#     e_cut =  LJ_energy(config.rlj, s_ij, e_ij)
#     energy = energies - e_cut
#     energy = jnp.nan_to_num(energy, nan=0.0)

#     return jnp.sum(energy), forces


@jit
def get_LJ_energy_and_forces(
    forces: Array,
    pair_params: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config
) -> Tuple[float, Array]:

    forces = forces.at[...].set(0.0)

    r_vec, r, neigh_i, neigh_j, _, _, s_ij, e_ij = pair_params
    r_vec, s_ij, e_ij, neigh_i, neigh_j = apply_cutoff(r_vec, r, s_ij, e_ij, neigh_i, neigh_j, config.rlj)

    LJ_grad = vmap(value_and_grad(LJ_energy), (0, 0, 0))
    energies, grads = LJ_grad(r_vec, s_ij, e_ij)

    grads = jnp.nan_to_num(grads, nan=0.0)
    forces = forces.at[neigh_i].add(-grads)
    forces = forces.at[neigh_j].add(grads)

    e_cut =  LJ_energy(config.rlj, s_ij, e_ij)
    energy = energies - e_cut
    energy = jnp.nan_to_num(energy, nan=0.0)

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

    LJ_grad = vmap(value_and_grad(LJ_energy), (0, 0, 0))
    energies, grads = LJ_grad(r_vec, s_ij, e_ij)

    grads = jnp.nan_to_num(grads, nan=0.0)
    forces = forces.at[neigh_i].add(-grads)
    forces = forces.at[neigh_j].add(grads)

    e_cut =  LJ_energy(config.rlj, s_ij, e_ij)
    energy = energies - e_cut
    energy = jnp.nan_to_num(energy, nan=0.0)

    # virial = - 0.5 * jnp.sum(
    #     (jnp.expand_dims(r_vec, 2) * jnp.expand_dims(-grads, 1)),
    #     axis=0
    # )

    # virial = jnp.trace(virial)

    virial = - 0.5 * jnp.sum(-grads*r_vec, axis=0)

    return jnp.sum(energy), forces, virial


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

    # The code below does the following
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
        config.empty_mesh,  # jnp.zeros(mesh_size)
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

    # The code below does the same as the following
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


@jit
def filter_density(phi: Array, config: Config) -> Array:
    phi_rescale = phi / config.volume_per_cell
    phi_fourier = jnp.fft.rfftn(phi_rescale, norm="forward")
    return phi_fourier * config.window()


@jit
def fft_density(phi: Array, config: Config) -> Array:
    phi_fourier = filter_density(phi, config)
    phi_tilde = jnp.fft.irfftn(phi_fourier, norm="forward")
    return phi_tilde


@jit
def paint(positions: Array, config: Config, mass: float | Array = 1.0):
    phi = cic_paint(positions, config, mass)
    phi = fft_density(phi, config)
    return phi


@jit
def calculate_potential(phi: Array, chi: Array, config: Config) -> Array:
    """Calculate explicitly the external potential for a single type"""
    # chi has shape (n, 1, 1, 1), phi has shape (n, *mesh_size)
    interaction = jnp.sum(chi * phi, axis=0)
    incompressibility = (jnp.sum(phi, axis=0) - config.a) / config.kappa
    dw_dphi = interaction + incompressibility
    return dw_dphi / config.rho0


@jit
def get_fog(fog: Array, phi: Array, chi: Array, config: Config) -> Array:
    for i in range(config.n_types):
        v_ext_real = calculate_potential(phi, chi[i], config)
        v_ext_fourier = jnp.fft.rfftn(v_ext_real, norm="forward") * config.window()
        for d in range(3):
            fog = fog.at[i, d].set(
                jnp.fft.irfftn(
                    -1j * config.k_vector[d] * v_ext_fourier,
                    norm="forward",
                )
            )
    return fog


@jit
def get_field_energy_and_forces_npt(
    positions: Array,
    phi: Array,
    phi_fourier: Array,
    field_forces: Array,
    fog: Array,
    chi: Array,
    type_mask: dict[int, Array],
    elec_potential: Array,
    config: Config,
):
    for i, ti in enumerate(config.unique_types):
        phi_cic = cic_paint(positions, config, mass=type_mask[ti])
        phi_fourier = phi_fourier.at[i].set(filter_density(phi_cic, config))
        phi = phi.at[i].set(jnp.fft.irfftn(phi_fourier[i], norm="forward"))

    fog, field_pressure = get_fog_and_pressure(
        fog, phi, phi_fourier, chi, elec_potential, config
    )
    # We save a loop by calculating the χ interactions here
    chi_interaction = 0.0
    field_forces = field_forces.at[...].set(0.0)
    for i, ti in enumerate(config.unique_types):
        chi_interaction += jnp.sum(chi[i] * phi[i] * phi, axis=0)
        field_forces = field_forces.at[...].add(
            cic_readout(positions, fog[i], config, mass=type_mask[ti])
        )

    incompressibility = 0.5 / config.kappa * (jnp.sum(phi, axis=0) - config.a) ** 2
    # Interactions are double counted so we divide by 2
    w = 0.5 * chi_interaction + incompressibility
    field_energy = config.volume_per_cell / config.rho0 * jnp.sum(w)
    return field_energy, field_forces, field_pressure


@jit
def get_fog_and_pressure(
    fog: Array,
    phi: Array,
    phi_fourier: Array,
    chi: Array,
    elec_potential: Array,
    config: Config,
) -> Tuple[Array, Array]:
    """Laplacian is computed in Fourier space as -k^2 phi."""
    iso_pressure = 0.0
    aniso_pressure = jnp.zeros(3)
    for i, ti in enumerate(config.unique_types):
        v_ext_real = calculate_potential(phi, chi[i], config)  # V_bar
        v_ext_fourier = jnp.fft.rfftn(v_ext_real, norm="forward") * config.window()
        for d in range(3):
            # Calculate forces on grid
            fog = fog.at[i, d].set(
                jnp.fft.irfftn(
                    -1j * config.k_vector[d] * v_ext_fourier,
                    norm="forward",
                )
            )

        v_ext_real += elec_potential * config.type_to_charge_map[ti]
        iso_pressure += jnp.sum(v_ext_real * phi[i])
        for d in range(3):
            # Evaluate laplacian of phi in fourier space (single type and direction)
            laplacian = jnp.fft.irfftn(
                -config.k_vector[d] * config.k_vector[d] * phi_fourier[i],
                norm="forward",
            )
            aniso_pressure = aniso_pressure.at[d].add(jnp.sum(v_ext_real * laplacian))
    iso_pressure *= config.volume_per_cell
    aniso_pressure *= config.volume_per_cell * config.sigma**2
    return fog, (iso_pressure + aniso_pressure)


@jit
def get_field_energy(phi: Array, chi: Array, config: Config) -> Array:
    interaction = 0.0
    for i, _ in enumerate(config.unique_types):
        interaction += jnp.sum(chi[i] * phi[i] * phi, axis=0)

    incompressibility = 0.5 / config.kappa * (jnp.sum(phi, axis=0) - config.a) ** 2
    # Interactions are double counted so we divide by 2
    w = 0.5 * interaction + incompressibility
    return config.volume_per_cell / config.rho0 * jnp.sum(w)


@jit
def get_field_energy_and_forces(
    positions: Array,
    phi: Array,
    field_forces: Array,
    fog: Array,
    chi: Array,
    type_mask: dict[int, Array],
    config: Config,
) -> Tuple[ArrayLike, Array]:
    for i, ti in enumerate(config.unique_types):
        phi = phi.at[i].set(paint(positions, config, mass=type_mask[ti]))

    # We save a loop by calculating the φ interactions here
    interaction = 0.0
    field_forces = field_forces.at[...].set(0.0)
    fog = get_fog(fog, phi, chi, config)
    for i, ti in enumerate(config.unique_types):
        interaction += jnp.sum(chi[i] * phi[i] * phi, axis=0)
        field_forces = field_forces.at[...].add(
            cic_readout(positions, fog[i], config, mass=type_mask[ti])
        )

    incompressibility = 0.5 / config.kappa * (jnp.sum(phi, axis=0) - config.a) ** 2
    # Interactions are double counted so we divide by 2
    w = 0.5 * interaction + incompressibility
    field_energy = config.volume_per_cell / config.rho0 * jnp.sum(w)
    return field_energy, field_forces


@jit
def get_field_forces(
    positions: Array,
    phi: Array,
    field_forces: Array,
    fog: Array,
    chi: Array,
    mass: dict[int, Array],
    config: Config,
) -> Array:
    """Deprecated / Not used"""
    for i, _ in enumerate(config.unique_types):
        phi = phi.at[i].set(paint(positions, config, mass=mass[i]))

    field_forces = field_forces.at[...].set(0.0)
    fog = get_fog(fog, phi, chi, config)
    for i, _ in enumerate(config.unique_types):
        field_forces = field_forces.at[...].add(
            cic_readout(positions, fog[i], config, mass=mass[i])
        )
    return field_forces


@jit
def get_Vbar_elec(Vbar: Array, elec_potential: Array, config: Config) -> Array:
    """Deprecated / Not used"""
    for i, ti in enumerate(config.unique_types):
        Vbar = Vbar.at[i, ...].add(
            elec_potential * config.type_to_charge_map[ti],
        )
    return Vbar


@jit
def comp_laplacian(laplacian: Array, phi_fourier: Array, config: Config) -> Array:
    """Laplacian is computed in Fourier space as -k**2 phi."""
    for t in range(config.n_types):
        # Evaluate laplacian of phi in fourier space
        for d in range(3):
            laplacian = laplacian.at[t, d].set(
                jnp.fft.irfftn(
                    -config.k_vector[d] * config.k_vector[d] * phi_fourier[t],
                    norm="forward",
                )
            )
    return laplacian


@jit
def reaction_field_force(
    r: Array,
    q_i: Array,
    q_j: Array,
    erf: float,
    er: float,
    rc: float) -> Array:

    krf = (erf - er) / (2*erf + er) * 1/rc**3
    c = 138.935458*q_i*q_j/er
    return c * (1/r**2 - 2*krf*r) / r


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
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    q_i = jnp.squeeze(q_i) # TODO: Try not to use this
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

    energy = 0.0
    forces = forces.at[...].set(0.0)
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
def get_reaction_field_energy_and_forces_npt(
    forces: Array,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    config: Config,
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[float, float, Array]:

    energy = 0.0
    forces = forces.at[...].set(0.0) # TODO: Test if this is sllow
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

    # Pressure terms
    pressure = jnp.sum(-grads*q_i*q_j * r_vec)

    if excl_pair_param is not None:
      energy, potential, forces, grads, r_vec, q_i, q_j = get_rf_excluded_pairs_energy_and_forces(
          excl_pair_param,
          config,
          forces,
          potentials,
          energy,
      )
      pressure += jnp.sum(-grads*q_i*q_j * r_vec)

    return energy-config.self_energy, potential, forces, pressure


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

    func = vmap(value_and_grad(ewald_rec_intramol), (0, None, None))
    phi_contributions, grads = func(r_vec, 1/(2*config.sigma**2), config.elec_conversion)

    # Forces
    grads = jnp.where((neigh_i == -1)[:, None], jnp.zeros(3), grads)
    forces = forces.at[neigh_i].add(-grads*q_i*q_j)
    forces = forces.at[neigh_j].add(grads*q_i*q_j)

    q_i = jnp.squeeze(q_i)
    q_j = jnp.squeeze(q_j)
    
    # Potential
    phi_contributions = jnp.where(neigh_i == -1, 0, phi_contributions)
    potentials = potentials.at[neigh_i].add(phi_contributions*q_j)
    potentials = potentials.at[neigh_j].add(phi_contributions*q_i)
    potential = jnp.sum(potentials)

    # Energy
    energy += jnp.sum(phi_contributions*q_j*q_i)

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

    potential = vmap(value_and_grad(phi_real_space), (0, None, None))
    phi_contributions, grads = potential(r_vec, alpha, config.elec_conversion)

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

    potential = vmap(value_and_grad(phi_real_space), (0, None, None))
    phi_contributions, grads = potential(r_vec, alpha, config.elec_conversion)

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


@jit
def get_elec_fog(elec_fog: Array, phi_q_fourier: Array, config: Config) -> Array:
    for d in range(3):
        # CHECK: if there's a way to do this without loop and assignment
        elec_fog = elec_fog.at[d].set(
            jnp.fft.irfftn(
                -1j
                * config.elec_const
                / config.knorm()
                * config.k_vector[d]
                * phi_q_fourier,
                norm="forward",
            )
        )
    return elec_fog


@jit
def get_elec_forces(
    elec_fog: Array,
    phi_q_fourier: Array,
    positions: Array,
    charges: Array,
    config: Config,
) -> Array:
    elec_fog = get_elec_fog(elec_fog, phi_q_fourier, config)
    elec_forces = cic_readout(positions, elec_fog, config, mass=charges)
    return elec_forces


@jit
def get_elec_energy_potential_and_forces(
    positions: Array,
    elec_fog: Array,
    charges: Array,
    config: Config,
    elec_param: Tuple[Array, Array, Array, Array, Array, Array],
    excl_pair_param: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
) -> Tuple[Array, Array, Array]:
    phi_q = cic_paint(positions, config, mass=charges)
    phi_q_fourier = filter_density(phi_q, config)
    elec_forces = get_elec_forces(elec_fog, phi_q_fourier, positions, charges, config)
    elec_energy_real, elec_forces_real, potential_real = ewald_real_space(elec_param, config, excl_pair_param)
    elec_potential, elec_energy, long_range_energy = get_elec_potential_and_energy(
        phi_q, phi_q_fourier, config
    )
    return elec_energy+elec_energy_real, elec_potential+potential_real, elec_forces+elec_forces_real#, elec_energy_real, long_range_energy


@partial(jit, static_argnums=(3,))
def get_dipole_forces(
    dip_positions: Array, dip_charges: Array, dip_fog: Array, n_dip: int, config: Config
) -> Tuple[Array, Array]:
    phi_dip = cic_paint(dip_positions, config, mass=dip_charges)
    phi_dip_fourier = filter_density(phi_dip, config)
    dip_forces = get_elec_forces(
        dip_fog, phi_dip_fourier, dip_positions, dip_charges, config
    )
    return (
        dip_forces[:n_dip] + dip_forces[n_dip:],
        dip_forces[:n_dip] - dip_forces[n_dip:],
    )
