from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import Array, jit, lax, value_and_grad
from jax.typing import ArrayLike

from .config import Config

from .neighbor_list import apply_cutoff

@jit
def LJ_potential(r, e, s):
    rnorm = jnp.linalg.norm(r)
    r6 = (s/rnorm)**6
    return 4.0 * e * r6 * (r6 - 1.0)

@jit
def LJ_forces(r, e, s):
    rnorm = jnp.linalg.norm(r)
    r2i = 1.0/(rnorm*rnorm)
    r6 = (s/rnorm)**6
    ff = 48.0*e*r6 * ( r6 - 0.5 ) * r2i
    return r*ff

# @jit
# def get_LJ_energy_and_forces(
#     positions: Array, types: Array, sigma: Array, epsilon: Array,
# ) -> Tuple[Array, Array]:
#     """Compute Lennard-Jones forces for a set of particles."""
#     num_particles = len(positions)
#     forces = jnp.zeros((num_particles, 3))  # Initialize forces array
#     energies = jnp.zeros((num_particles, 3))  # Initialize energies array
#     for idx_i, i in enumerate(positions):
#         for idx_j, j in enumerate(positions):
#             if idx_j <= idx_i:  
#                 continue

#             # Vector difference 
#             r_vec = i - j

#             # Take interaction parameters            
#             s, e = sigma[types[idx_i],types[idx_j]], epsilon[types[idx_i],types[idx_j]] 

#             # Compute gradients
#             LJ_grad = value_and_grad(LJ_potential)
#             energy, grad = LJ_grad(r_vec, e, s)

#             # Update forces and energies symmetrically
#             forces = forces.at[idx_i].add(-grad)
#             forces = forces.at[idx_j].add(grad)
           
#             energies = energies.at[idx_i].add(-energy)
#             energies = energies.at[idx_j].add(energy) 

#     return energies, jnp.sum(energies), forces


@jit
def get_LJ_energy_and_forces(
    positions: Array,
    types: Array,
    sigma: Array, 
    epsilon: Array, 
    neigh_i: Array, 
    neigh_j: Array,
    box: Array,
    rc: float
) -> Tuple[float, Array]:

    num_particles = len(positions)
    forces = jnp.zeros((num_particles, 3))  # Initialize forces array
    energy = 0.0  # Initialize energy

    for i, j in zip(neigh_i, neigh_j):
        i_r = positions[i]
        j_r = positions[j]
        r_vec = i_r - j_r
        r_vec = r_vec - box * jnp.round(r_vec / box)
        r_norm = jnp.linalg.norm(r_vec)

        sigma_ij = sigma[types[i], types[j]]
        epsilon_ij = epsilon[types[i], types[j]]

        # compute the energy and force for each particle pair i, j
        en_pair = LJ_potential(r_vec, epsilon_ij, sigma_ij)
        ecut = LJ_potential(rc, epsilon_ij, sigma_ij) # TODO: make a table for this
        f_pair = LJ_forces(r_vec, epsilon_ij, sigma_ij)

        forces = forces.at[i].add(f_pair)
        forces = forces.at[j].add(-f_pair)
        energy = en_pair - ecut
    print(f'{energy}\n{forces}')
    return energy, forces


@jit
def get_LJ_energy_and_forces_npt(
    positions: Array, types: Array, sigma: Array, epsilon: Array,
) -> Tuple[Array, Array, Array]:
    """Compute Lennard-Jones forces for a set of particles."""
    num_particles = len(positions)
    forces = jnp.zeros((num_particles, 3))  # Initialize forces array
    energies = jnp.zeros((num_particles, 3))  # Initialize energies array

    for idx_i, i in enumerate(positions):
        for idx_j, j in enumerate(positions):
            if idx_j <= idx_i:  
                continue

            # Vector difference 
            r_vec = i - j

            # Take interaction parameters            
            s, e = sigma[types[idx_i],types[idx_j]], epsilon[types[idx_i],types[idx_j]] 

            # Compute gradients
            LJ_grad = value_and_grad(LJ_potential)
            energy, grad = LJ_grad(r_vec, e, s)

            # Update forces and energies symmetrically
            forces = forces.at[idx_i].add(-grad)
            forces = forces.at[idx_j].add(grad)
           
            energies = energies.at[idx_i].add(-energy)
            energies = energies.at[idx_j].add(energy) 

    return energies, forces#, vdw_pressure

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
def get_elec_potential_and_energy(
    phi_q: Array, phi_q_fourier: Array, config: Config
) -> Tuple[Array, Array]:
    elec_potential_fourier = config.elec_const * phi_q_fourier / config.knorm()
    elec_potential = jnp.fft.irfftn(elec_potential_fourier, norm="forward")

    long_range_energy = 0.5 * jnp.sum(phi_q * elec_potential)
    elec_energy = long_range_energy - config.self_energy
    return elec_potential, elec_energy


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
    config: Config
) -> Tuple[Array, Array, Array]:
    phi_q = cic_paint(positions, config, mass=charges)
    phi_q_fourier = filter_density(phi_q, config)
    elec_forces = get_elec_forces(elec_fog, phi_q_fourier, positions, charges, config)
    elec_potential, elec_energy = get_elec_potential_and_energy(
        phi_q, phi_q_fourier, config
    )
    return elec_energy, elec_potential, elec_forces


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
