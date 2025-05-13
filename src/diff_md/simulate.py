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
from .integrator import integrate_position, integrate_velocity, zero_forces
from .nonbonded import (
    get_dipole_forces,
    get_elec_energy_potential_and_forces,
    get_LJ_energy_and_forces, get_LJ_energy_and_forces_npt,
    get_reaction_field_energy_and_forces, get_reaction_field_energy_and_forces_npt
)
from .thermostat import (
    cancel_com_momentum,
    csvr_thermostat,
    generate_initial_velocities
)
from .neighbor_list import (
    nlist,
    apply_nlist, 
    apply_nlist_elec, 
    exclude_bonded_neighbors
)

from jax import config 

config.update("jax_debug_nans", True)
# config.update('jax_enable_x64', True)

# @jit # takes too much time and memory to compile, at least on cpu
def simulator(
    model,
    positions,
    velocities,
    types,
    masses,
    charges,
    epsl_table,
    key,
    topol,
    config,
    start_temperature,
    equilibration=0,
):
    # Dict to save trajectory
    trj = {}

    # print(epsl_table)

    # Arrays to store dihedral angle information for fitting 2d distribution
    # if protein_flag:
    #     dihedral_phi = jnp.empty(0)
    #     dihedral_theta = jnp.empty(0)
    # dihedral_phi = jnp.empty(0)
    # dihedral_theta = jnp.empty(0)

    if start_temperature:
        key, subkey = jax.random.split(key)
        velocities = generate_initial_velocities(velocities, subkey, config, masses)
        velocities = cancel_com_momentum(velocities, config)

    positions = jnp.mod(positions, config.box_size)

    # Init bonded forces
    bond_forces = jnp.zeros_like(positions)
    angle_forces = jnp.zeros_like(positions)
    dihedral_forces = jnp.zeros_like(positions)
    improper_forces = jnp.zeros_like(positions)

    # Init non-bonded forces
    LJ_forces = jnp.zeros_like(positions)
    elec_forces = jnp.zeros_like(positions)
    elec_potential = jnp.zeros(config.empty_mesh.shape)
    reconstr_forces = jnp.zeros_like(positions)

    phi = jnp.zeros((config.n_types, *config.empty_mesh.shape))

    restr_atoms = topol.restraints

    # Make neighbor list
    rv = config.rv 
    ns_nlist = config.ns_nlist 
    dens = config.n_particles / config.box_size.prod()
    max_neighbors = int((1/2) * config.n_particles * ( 4 * jnp.pi * rv**3 / 3 ) * dens)
    max_neighbors += 5000 # Add a buffer for safety

    # Inicialize neighbor list
    neigh_i = jnp.full(max_neighbors, -1, dtype=int)
    neigh_j = jnp.full(max_neighbors, -1, dtype=int)
    # neigh_i, neigh_j = jax.lax.stop_gradient(nlist(positions, config.box_size, rv, neigh_i, neigh_j))
    neigh_i, neigh_j = nlist(positions, config.box_size, rv, neigh_i, neigh_j)

    if topol.excluded_pairs is not None:
        neigh_i, neigh_j = exclude_bonded_neighbors(neigh_i, neigh_j, topol.excluded_pairs[0], topol.excluded_pairs[1])

    # NOTE: This can probably be cleaned up
    if config.coulombtype and charges is not None:
        pair_params = apply_nlist_elec(
            neigh_i, 
            neigh_j, 
            positions, 
            charges, 
            config.box_size, 
            config.sgm_table, 
            epsl_table, 
            types
        )
        if topol.excluded_pairs is not None:
            excl_pair_params = apply_nlist_elec(
                topol.excluded_pairs[0],
                topol.excluded_pairs[1],
                positions,
                charges,
                config.box_size,
                config.sgm_table,
                epsl_table,
                types,
            )
        else:
            excl_pair_params = None
    else:
        pair_params = apply_nlist(
            neigh_i, 
            neigh_j, 
            positions, 
            config.box_size, 
            config.sgm_table, 
            epsl_table, 
            types
        )

    # Init energies
    bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy = 0, 0, 0, 0, 0  # fmt:skip
    bond_pressure, angle_pressure, dihedral_pressure, LJ_pressure = 0, 0, 0, 0

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
    if charges is not None:
        if config.coulombtype == 1:
            elec_fog = jnp.zeros((3, *config.mesh_size))
            elec_energy, elec_potential, elec_forces = get_elec_energy_potential_and_forces(
                positions, elec_fog, charges, config, pair_params, excl_pair_params
            )
        elif config.coulombtype == 2:
            elec_energy, elec_potential, elec_forces = get_reaction_field_energy_and_forces(
                elec_forces, pair_params, config, excl_pair_params
            )    

    # Calculate initial non bonded energy and forces
    LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config
        )
        
    if config.barostat:
        ctype = jnp.complex128 if phi.dtype == "float64" else jnp.complex64
        phi_fourier = jnp.zeros((config.n_types, *config.fft_shape), dtype=ctype)

    # Save step 0 to trajectory
    if config.n_print > 0:
        # NOTE: we don't need to save all this stuff for the differentiable MD
        kinetic_energy = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
        #kinetic_energy = 0.5 * jnp.sum(masses * jnp.linalg.norm(velocities, axis=1)**2)
        # kinetic_energy = 0.5 * config.mass * jnp.sum(velocities * velocities)
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
        trj["angle energy"] = [angle_energy]
        trj["bond energy"] = [bond_energy]
        trj["box"] = [config.box_size]
        trj["dihedral energy"] = [dihedral_energy]
        trj["elec energy"] = [elec_energy]
        trj["LJ energy"] = [LJ_energy]

        trj["forces"] = [
            bond_forces
            + angle_forces
            + dihedral_forces
            + improper_forces
            + LJ_forces
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
        if step%ns_nlist == 0:
            neigh_i = jnp.full(max_neighbors, -1, dtype=int)
            neigh_j = jnp.full(max_neighbors, -1, dtype=int)
            neigh_i, neigh_j = nlist(positions, config.box_size, rv, neigh_i, neigh_j)

            if topol.excluded_pairs is not None:
                neigh_i, neigh_j = exclude_bonded_neighbors(neigh_i, neigh_j, topol.excluded_pairs[0], topol.excluded_pairs[1])

        if len(restr_atoms) > 0 :
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            reconstructed_forces = zero_forces(reconstructed_forces, restr_atoms)

        velocities = integrate_velocity(
            velocities,
            (LJ_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Inner rRESPA steps
        for _ in range(config.respa_inner):
            if len(restr_atoms) > 0 :
                bond_forces = zero_forces(bond_forces, restr_atoms)
                angle_forces = zero_forces(angle_forces, restr_atoms)
                dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
                improper_forces = zero_forces(improper_forces, restr_atoms)

            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces + dihedral_forces + improper_forces)
                / config.mass,
                config.inner_ts,
            ) # TODO: Make sure this is right for system with different masses
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

        # Second rRESPA velocity step
        if config.barostat:
            # Get electrostatic potential
            if config.coulombtype and charges is not None:
                pair_params = apply_nlist_elec(
                    neigh_i, 
                    neigh_j, 
                    positions, 
                    charges, 
                    config.box_size, 
                    config.sgm_table, 
                    epsl_table, 
                    types
                )
                if topol.excluded_pairs is not None:
                    excl_pair_params = apply_nlist_elec(
                        topol.excluded_pairs[0],
                        topol.excluded_pairs[1],
                        positions,
                        charges,
                        config.box_size,
                        config.sgm_table,
                        epsl_table,
                        types,
                    )
            else:
                pair_params = apply_nlist(
                    neigh_i, 
                    neigh_j, 
                    positions, 
                    config.box_size, 
                    config.sgm_table, 
                    epsl_table, 
                    types
                )            

            if charges is not None:
                if config.coulombtype == 1:
                    (
                        elec_energy,
                        elec_potential,
                        elec_forces,
                    ) = get_elec_energy_potential_and_forces(
                        positions, elec_fog, charges, config, pair_params, excl_pair_params
                    )
                elif config.coulombtype == 2:
                    (
                        elec_energy, 
                        elec_potential, 
                        elec_forces,
                        ele_pressure
                    ) = get_reaction_field_energy_and_forces_npt(
                        elec_forces, pair_params, config, excl_pair_params
                    )

            (
                LJ_energy,
                LJ_forces,
                LJ_pressure
            ) = get_LJ_energy_and_forces_npt(
                LJ_forces, pair_params, config
            )            

            # Calculate pressure
            kinetic_energy = 0.5 * jnp.sum(masses * jnp.linalg.norm(velocities, axis=1)**2)
            kinetic_pressure = 2.0 / 3.0 * kinetic_energy
            pressure = (
                kinetic_pressure
                + LJ_pressure # 
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


        if config.coulombtype and charges is not None:
            pair_params = apply_nlist_elec(
                neigh_i, 
                neigh_j, 
                positions, 
                charges, 
                config.box_size, 
                config.sgm_table, 
                epsl_table, 
                types
            )
            if topol.excluded_pairs is not None: # This should be applied when we don't have charges too
                excl_pair_params = apply_nlist_elec(
                    topol.excluded_pairs[0],
                    topol.excluded_pairs[1],
                    positions,
                    charges,
                    config.box_size,
                    config.sgm_table,
                    epsl_table,
                    types,
                )
        else:
            pair_params = apply_nlist(
                neigh_i, 
                neigh_j, 
                positions, 
                config.box_size, 
                config.sgm_table, 
                epsl_table, 
                types
            )      
        
        # Recompute after barostat
        LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config 
        )

        if charges is not None:
            if config.coulombtype == 1:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, elec_fog, charges, config, pair_params, excl_pair_params
                )
            elif config.coulombtype == 2:
                (
                    elec_energy, 
                    elec_potential, 
                    elec_forces,
                ) = get_reaction_field_energy_and_forces(
                    elec_forces, pair_params, config, excl_pair_params
                )

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
            (LJ_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Apply thermostat
        if config.target_temperature:
            velocities, key = csvr_thermostat(velocities, key, config, masses)

        # Remove total linear momentum
        if config.cancel_com_momentum:
            if jnp.mod(step, config.cancel_com_momentum) == 0:
                velocities = cancel_com_momentum(velocities, config)

        # Update trajectory dict, print later after calculating grads
        if config.n_print > 0:
            if onp.mod(step, config.n_print) == 0 and step != 0:
                frame = step // config.n_print
                kinetic_energy = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
                # kinetic_energy = 0.5 * jnp.sum(masses * jnp.linalg.norm(velocities, axis=1)**2)
                temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
                trj["angle energy"].append(angle_energy)
                trj["bond energy"].append(bond_energy)
                trj["box"].append(config.box_size)
                trj["dihedral energy"].append(dihedral_energy)
                trj["elec energy"].append(elec_energy)
                trj["LJ energy"].append(LJ_energy)
                trj["forces"].append(
                    bond_forces
                    + angle_forces
                    + dihedral_forces
                    + LJ_forces
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
