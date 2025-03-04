import jax
import jax.numpy as jnp
import numpy as onp

from .barostat import berendsen, c_rescale
from .force import (
    get_angle_energy_and_forces,
    get_bond_energy_and_forces,
    get_dihedral_energy_and_forces,
    get_impropers_energy_and_forces,
    get_protein_dipoles,
    redistribute_dipole_forces,
)
from .integrator import integrate_position, integrate_velocity
from .nonbonded import (
    get_dipole_forces,
    get_elec_energy_potential_and_forces,
    get_field_energy_and_forces,
    get_field_energy_and_forces_npt,
)
from .thermostat import (
    cancel_com_momentum,
    csvr_thermostat,
    generate_initial_velocities,
)


# @jit # takes too much time and memory to compile, at least on cpu
def simulator(
    model,
    positions,
    velocities,
    type_mask,
    charges,
    chi,
    key,
    topol,
    config,
    start_temperature,
    equilibration=0,
):
    # Dict to save trajectory
    trj = {}

    # Arrays to store dihedral angle information for fitting 2d distribution
    # if protein_flag:
    #     dihedral_phi = jnp.empty(0)
    #     dihedral_theta = jnp.empty(0)
    # dihedral_phi = jnp.empty(0)
    # dihedral_theta = jnp.empty(0)

    if start_temperature:
        key, subkey = jax.random.split(key)
        velocities = generate_initial_velocities(velocities, subkey, config)
        velocities = cancel_com_momentum(velocities, config)

    positions = jnp.mod(positions, config.box_size)

    # Init bonded forces
    bond_forces = jnp.zeros_like(positions)
    angle_forces = jnp.zeros_like(positions)
    dihedral_forces = jnp.zeros_like(positions)
    improper_forces = jnp.zeros_like(positions)

    # Init non-bonded forces
    field_fog = jnp.zeros((config.n_types, 3, *config.empty_mesh.shape))
    field_forces = jnp.zeros_like(positions)
    elec_forces = jnp.zeros_like(positions)
    elec_potential = jnp.zeros(config.empty_mesh.shape)
    reconstr_forces = jnp.zeros_like(positions)

    phi = jnp.zeros((config.n_types, *config.empty_mesh.shape))

    # Init energies
    bond_energy, angle_energy, dihedral_energy, field_energy, elec_energy = 0, 0, 0, 0, 0  # fmt:skip
    bond_pressure, angle_pressure, dihedral_pressure, field_pressure = 0, 0, 0, 0

    # Calculate initial bonded energies and forces
    if topol.bonds:
        bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
            bond_forces, positions, config.box_size, *topol.bonds_2
        )

    if topol.angles:
        angle_energy, angle_forces, angle_pressure = get_angle_energy_and_forces(
            angle_forces, positions, config.box_size, *topol.bonds_3
        )
    if topol.dihedrals:
        (
            dihedral_energy,
            dihedral_forces,
            # (phi, theta),
            _,
            dihedral_pressure,
        ) = get_dihedral_energy_and_forces(
            dihedral_forces, positions, config.box_size, *topol.bonds_4
        )
        # if protein_flag:
        #     dihedral_phi = jnp.append(dihedral_phi, phi)
        #     dihedral_theta = jnp.append(dihedral_theta, theta)

        # Init protein backbone dipoles
        # TODO: should only happen when we actually have proteins
        # protein_flag = hasattr(model, "dihedrals") and isinstance(model.dihedrals, dict)
        dip_fog = jnp.zeros((3, *config.empty_mesh.shape))
        n_dip = topol.dihedrals + 1
        dip_charges = jnp.hstack((jnp.full(n_dip, 0.25), jnp.full(n_dip, -0.25)))
        dip_charges = dip_charges.reshape((2 * n_dip, 1))

        transfer_matrices, dip_positions = get_protein_dipoles(
            positions, config.box_size, *topol.bonds_d
        )
        dip_forces = get_dipole_forces(
            dip_positions, dip_charges, dip_fog, n_dip, config
        )
        reconstr_forces = redistribute_dipole_forces(
            reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
        )

    if topol.impropers:
        improper_energy, improper_forces = get_impropers_energy_and_forces(
            improper_forces, positions, config.box_size, *topol.bonds_impr
        )
        dihedral_energy += improper_energy

    # Calculate initial electrostatic energy and forces
    if config.coulombtype:
        elec_fog = jnp.zeros((3, *config.empty_mesh.shape))
        elec_energy, elec_potential, elec_forces = get_elec_energy_potential_and_forces(
            positions, elec_fog, charges, config
        )

    # Calculate initial non bonded energy and forces
    field_energy, field_forces = get_field_energy_and_forces(
        positions, phi, field_forces, field_fog, chi, type_mask, config
    )

    if config.barostat:
        ctype = jnp.complex128 if phi.dtype == "float64" else jnp.complex64
        phi_fourier = jnp.zeros((config.n_types, *config.fft_shape), dtype=ctype)

    # Save step 0 to trajectory
    if config.n_print > 0:
        # NOTE: we don't need to save all this stuff for the differentiable MD
        kinetic_energy = 0.5 * config.mass * jnp.sum(velocities**2)
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
        trj["angle energy"] = [angle_energy]
        trj["bond energy"] = [bond_energy]
        trj["box"] = [config.box_size]
        trj["dihedral energy"] = [dihedral_energy]
        trj["elec energy"] = [elec_energy]
        trj["field energy"] = [field_energy]
        trj["forces"] = [
            bond_forces
            + angle_forces
            + dihedral_forces
            + improper_forces
            + field_forces
            + reconstr_forces
            + elec_forces
        ]
        trj["kinetic energy"] = [kinetic_energy]
        trj["positions"] = [positions]
        trj["temperature"] = [temperature]
        trj["velocities"] = [velocities]

    # MD loop
    n_steps = equilibration if equilibration else config.n_steps
    for step in range(1, n_steps + 1):
        # First outer rRESPA velocity step
        velocities = integrate_velocity(
            velocities,
            (field_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Inner rRESPA steps
        for _ in range(config.respa_inner):
            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces + dihedral_forces + improper_forces)
                / config.mass,
                config.inner_ts,
            )
            # Update positions
            positions = integrate_position(positions, velocities, config.inner_ts)
            positions = jnp.mod(positions, config.box_size)

            # Update fast bonded forces
            if topol.bonds:
                bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
                    bond_forces, positions, config.box_size, *topol.bonds_2
                )
            if topol.angles:
                (
                    angle_energy,
                    angle_forces,
                    angle_pressure,
                ) = get_angle_energy_and_forces(
                    angle_forces, positions, config.box_size, *topol.bonds_3
                )
            if topol.dihedrals:
                (
                    dihedral_energy,
                    dihedral_forces,
                    # (phi, theta),
                    _,
                    dihedral_pressure,
                ) = get_dihedral_energy_and_forces(
                    dihedral_forces, positions, config.box_size, *topol.bonds_4
                )
                # if protein_flag:
                #     dihedral_phi = jnp.append(dihedral_phi, phi)
                #     dihedral_theta = jnp.append(dihedral_theta, theta)
            if topol.impropers:
                improper_energy, improper_forces = get_impropers_energy_and_forces(
                    improper_forces, positions, config.box_size, *topol.bonds_impr
                )
                dihedral_energy += improper_energy

            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces + dihedral_forces + improper_forces)
                / config.mass,
                config.inner_ts,
            )

        # Append only last inner loop angles
        # if protein_flag:
        #     dihedral_phi = jnp.append(dihedral_phi, phi)
        #     dihedral_theta = jnp.append(dihedral_theta, theta)

        if config.barostat:
            # Get electrostatic potential
            if config.coulombtype:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, elec_fog, charges, config
                )

            # Get field pressure terms
            (
                field_energy,
                field_forces,
                field_pressure,
            ) = get_field_energy_and_forces_npt(
                positions,
                phi,
                phi_fourier,
                field_forces,
                field_fog,
                chi,
                type_mask,
                elec_potential,
                config,
            )

            # Calculate pressure
            kinetic_energy = 0.5 * config.mass * jnp.sum(velocities**2)
            kinetic_pressure = 2.0 / 3.0 * kinetic_energy
            pressure = (
                kinetic_pressure
                + (field_pressure - field_energy - elec_energy)
                + bond_pressure
                + angle_pressure
                + dihedral_pressure
            ) / config.volume

            # Call barostat
            if config.barostat == 1:
                positions, config = berendsen(
                    pressure,
                    positions,
                    config,
                )
            elif config.barostat == 2:
                positions, velocities, config, key = c_rescale(
                    pressure,
                    positions,
                    velocities,
                    config,
                    key,
                )

        # Recompute after barostat
        field_energy, field_forces = get_field_energy_and_forces(
            positions, phi, field_forces, field_fog, chi, type_mask, config
        )

        if config.coulombtype == 1:
            (
                elec_energy,
                elec_potential,
                elec_forces,
            ) = get_elec_energy_potential_and_forces(
                positions, elec_fog, charges, config
            )
        # elif config.coulombtype == 2:
        #     (
        #         elec_energy,
        #         elec_potential,
        #         elec_forces,
        #     ) = get_elec_energy_potential_and_forces(
        #         positions, elec_fog, charges, config
        #     )

        if topol.dihedrals:
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, config.box_size, *topol.bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, dip_charges, dip_fog, n_dip, config
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
            )

        # Second outer rRESPA velocity step
        velocities = integrate_velocity(
            velocities,
            (field_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Apply thermostat
        if config.target_temperature:
            velocities, key = csvr_thermostat(velocities, key, config)

        # Remove total linear momentum
        if config.cancel_com_momentum:
            if jnp.mod(step, config.cancel_com_momentum) == 0:
                velocities = cancel_com_momentum(velocities, config)

        # Update trajectory dict, print later after calculating grads
        if config.n_print > 0:
            if onp.mod(step, config.n_print) == 0 and step != 0:
                frame = step // config.n_print
                kinetic_energy = 0.5 * config.mass * jnp.sum(velocities**2)
                temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
                trj["angle energy"].append(angle_energy)
                trj["bond energy"].append(bond_energy)
                trj["box"].append(config.box_size)
                trj["dihedral energy"].append(dihedral_energy)
                trj["elec energy"].append(elec_energy)
                trj["field energy"].append(field_energy)
                trj["forces"].append(
                    bond_forces
                    + angle_forces
                    + dihedral_forces
                    + field_forces
                    + reconstr_forces
                    + elec_forces
                )
                trj["kinetic energy"].append(kinetic_energy)
                trj["positions"].append(positions)
                trj["temperature"].append(temperature)
                trj["velocities"].append(velocities)

    # if protein_flag:
    #     return (dihedral_phi, dihedral_theta), trj, key

    return trj, key, config
