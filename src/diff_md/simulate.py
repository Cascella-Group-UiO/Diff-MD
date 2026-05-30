import jax
import jax.numpy as jnp
import numpy as onp
from collections import OrderedDict

from .barostat import berendsen, c_rescale, kinetic_pressure
from .config import BoxState
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
    get_coulomb_pair_energy_and_forces,
    get_coulomb_pair_energy_and_forces_npt,
    get_dipole_forces,
    get_elec_energy_potential_and_forces,
    get_elec_energy_potential_and_forces_npt,
    get_LJ_energy_and_forces, get_LJ_energy_and_forces_npt,
    get_reaction_field_pair_energy_and_forces,
    get_reaction_field_pair_energy_and_forces_npt,
    get_reaction_field_energy_and_forces, get_reaction_field_energy_and_forces_npt
)
from .thermostat import (
    apply_thermostat,
    cancel_com_momentum,
    generate_initial_velocities
)
from .neighbor_list import (
    build_neighbor_list_cell,
    maybe_rebuild_verlet_list_jax,
    apply_nlist, 
    apply_nlist_elec, 
    apply_nlist_general,
    compute_pair_distances,
    apply_nlist_precomputed,
    apply_nlist_elec_precomputed,
    exclude_bonded_neighbors,
    init_jaxmd_neighbor_list,
    maybe_update_jaxmd_nlist,
    resolve_verlet_radii,
)

from jax import config

# jax_debug_nans inserts a device->host sync after every operation — catastrophic
# for performance inside lax.scan.  Enable only for targeted local debugging:
#   jax.config.update("jax_debug_nans", True)
# config.update('jax_enable_x64', True)

# @jit # takes too much time and memory to compile, at least on cpu
def simulator(
    model,
    positions,
    velocities,
    types,
    masses,
    charges,
    sgm_table,
    epsl_table,
    key,
    topol,
    config,
    start_temperature,
    equilibration=0,
    differentiable=True,
):
    # Dict to save trajectory
    # trj = {}
    trj = OrderedDict()


    # All frames are kept in RAM and unwraping the whole trajectory takes long.
    if equilibration:
        n_print = 100
    else:
        n_print = config.n_print
    # n_print controls both trajectory output frequency AND reverse-mode memory:
    # each lax.scan chunk covers n_print steps, so the backward pass stores
    # exactly n_print × carry_size carries simultaneously.
    # For N=15K f64 the per-step carry is ~40 MB (pos+vel+7 forces+nlist);
    # n_print=10 → ~400 MB transient peak, n_print=400 → ~16 GB.  Tune accordingly.

    # Arrays to store dihedral angle information for fitting 2d distribution
    # if protein_flag:
    #     dihedral_phi = jnp.empty(0)
    #     dihedral_theta = jnp.empty(0)
    # dihedral_phi = jnp.empty(0)
    # dihedral_theta = jnp.empty(0)

    if start_temperature:
        key, subkey = jax.random.split(key)
        velocities = generate_initial_velocities(velocities, subkey, config, masses)
        velocities = cancel_com_momentum(velocities, masses)

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

    use_14_scaling = (
        config.ff_family == "amber_like"
        and topol.one_four_pairs is not None
        and (config.lj14_scale != 1.0 or config.coulomb14_scale != 1.0)
    )

    # When nrexcl >= 3, excluded_pairs already contains 1-4 pairs (bond distance 3),
    # so we must NOT concatenate one_four_pairs again to avoid double-counting.
    need_concat_14 = use_14_scaling and config.nrexcl < 3

    if topol.excluded_pairs is not None and need_concat_14:
        excluded_for_main = (
            jnp.concatenate((topol.excluded_pairs[0], topol.one_four_pairs[0])),
            jnp.concatenate((topol.excluded_pairs[1], topol.one_four_pairs[1])),
        )
    elif topol.excluded_pairs is not None:
        excluded_for_main = topol.excluded_pairs
    elif need_concat_14:
        excluded_for_main = topol.one_four_pairs
    else:
        excluded_for_main = None

    if excluded_for_main is not None:
        excluded_nlist_i, excluded_nlist_j = excluded_for_main
    else:
        excluded_nlist_i = jnp.empty((0,), dtype=jnp.int32)
        excluded_nlist_j = jnp.empty((0,), dtype=jnp.int32)

    if topol.excluded_pairs is not None and need_concat_14 and config.coulombtype == 1:
        excluded_for_elec = (
            jnp.concatenate((topol.excluded_pairs[0], topol.one_four_pairs[0])),
            jnp.concatenate((topol.excluded_pairs[1], topol.one_four_pairs[1])),
        )
    elif topol.excluded_pairs is not None:
        excluded_for_elec = topol.excluded_pairs
    elif need_concat_14 and config.coulombtype == 1:
        excluded_for_elec = topol.one_four_pairs
    else:
        excluded_for_elec = None

    # Make neighbor list
    skin = config.skin
    rv = config.rv
    ns_nlist = config.ns_nlist
    nlist_method = config.nlist_method

    force_cutoff, rv, jaxmd_skin = resolve_verlet_radii(
        config.rc, config.rlj, rv, skin
    )

    n_atoms = config.n_particles
    neighbor_fn = None  # only used by jaxmd

    if nlist_method == "jaxmd":
        # jax-md searches pairs within (r_cutoff + dr_threshold).
        # r_cutoff = max(rc, rlj) is the physical cutoff; dr_threshold is
        # chosen so the effective search radius matches Diff-MD's rv.
        r_cutoff_phys = force_cutoff
        cap_mult = getattr(config, 'nlist_capacity_multiplier', 1.25)
        neighbor_fn, nbrs, neigh_i, neigh_j = init_jaxmd_neighbor_list(
            positions, config.box_size, r_cutoff_phys, jaxmd_skin,
            capacity_multiplier=cap_mult,
        )
        nlist_state = nbrs
    else:
        dens = config.n_particles / config.box_size.prod()
        max_neighbors = int((1/2) * config.n_particles * ( 4 * jnp.pi * rv**3 / 3 ) * dens)
        max_neighbors += 50000 # Add a buffer for safety

        # Initialize neighbor list (cell-list accelerated)
        neigh_i, neigh_j, max_neighbors = build_neighbor_list_cell(
            positions, config.box_size, rv, max_neighbors
        )
        nlist_state = jnp.array(positions)  # ref_positions for Verlet

    if excluded_for_main is not None:
        neigh_i, neigh_j = exclude_bonded_neighbors(
            neigh_i, neigh_j, excluded_for_main[0], excluded_for_main[1]
        )

    if config.coulombtype and charges is not None:
        pair_params = apply_nlist_elec(
            neigh_i, 
            neigh_j, 
            positions, 
            charges, 
            config.box_size, 
            sgm_table, 
            epsl_table, 
            types
        )
        if excluded_for_elec is not None:
            excl_pair_params = apply_nlist_elec(
            excluded_for_elec[0],
            excluded_for_elec[1],
                positions,
                charges,
                config.box_size,
                sgm_table,
                epsl_table,
                types,
            )
        else:
            excl_pair_params = None
        if use_14_scaling:
            pair_params_14 = apply_nlist_elec(
                topol.one_four_pairs[0],
                topol.one_four_pairs[1],
                positions,
                charges,
                config.box_size,
                sgm_table,
                epsl_table,
                types,
            )
        else:
            pair_params_14 = None
    else:
        pair_params = apply_nlist(
            neigh_i, 
            neigh_j, 
            positions, 
            config.box_size, 
            sgm_table, 
            epsl_table, 
            types
        )
        excl_pair_params = None
        if use_14_scaling:
            pair_params_14 = apply_nlist(
                topol.one_four_pairs[0],
                topol.one_four_pairs[1],
                positions,
                config.box_size,
                sgm_table,
                epsl_table,
                types,
            )
        else:
            pair_params_14 = None
        

    # Init energies
    bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy = 0, 0, 0, 0, 0  # fmt:skip

    # Calculate initial bonded energies and forces
    if topol.bonds:
        bond_energy, bond_forces, _ = get_bond_energy_and_forces(
            bond_forces, positions, config.box_size, *topol.bonds_2
        )
    if topol.angles:
        angle_energy, angle_forces, _ = get_angle_energy_and_forces(
            angle_forces, positions, config.box_size, *topol.bonds_3,
            only_harmonic=topol.angle_uses_only_harmonic,
        )
    if topol.dihedrals:
        (
            dihedral_energy,
            dihedral_forces,
            _,
            _,
        ) = get_dihedral_energy_and_forces(
            dihedral_forces,
            positions,
            config.box_size,
            *topol.bonds_4,
            ff_family=config.ff_family,
        )
        # if protein_flag:
        #     dihedral_phi = jnp.append(dihedral_phi, phi)
        #     dihedral_theta = jnp.append(dihedral_theta, theta)

        # Init protein backbone dipoles
        # TODO: should only happen when we actually have proteins
        # protein_flag = hasattr(model, "dihedrals") and isinstance(model.dihedrals, dict)
        if config.ff_family != "amber_like":
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
        improper_energy, improper_forces, _ = get_impropers_energy_and_forces(
            improper_forces, positions, config.box_size, *topol.bonds_impr,
            compute_pressure=False,
        )
        dihedral_energy += improper_energy

    # Calculate initial electrostatic energy and forces
    if charges is not None:
        if config.coulombtype == 1:
            elec_energy, _, elec_forces = get_elec_energy_potential_and_forces(
                positions, charges, config, pair_params, excl_pair_params,
                compute_potential=False,
            )
            if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                elec_14_energy, elec_14_forces = get_coulomb_pair_energy_and_forces(
                    elec_forces,
                    pair_params_14,
                    config.coulomb14_scale,
                    config.elec_conversion,
                )
                elec_energy += elec_14_energy
                elec_forces += elec_14_forces
        elif config.coulombtype == 2:
            elec_energy, elec_potential, elec_forces = get_reaction_field_energy_and_forces(
                elec_forces, pair_params, config, excl_pair_params
            )
            if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                elec_14_energy, _, elec_14_forces = get_reaction_field_pair_energy_and_forces(
                    elec_forces,
                    pair_params_14,
                    config,
                    config.coulomb14_scale,
                )
                elec_energy += elec_14_energy
                elec_forces += elec_14_forces

    # Calculate initial non bonded energy and forces
    LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config
        )
    if use_14_scaling and pair_params_14 is not None and config.lj14_scale != 0.0:
        LJ_14_energy, LJ_14_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params_14, config
        )
        LJ_energy += config.lj14_scale * LJ_14_energy
        LJ_forces += config.lj14_scale * LJ_14_forces
        
    if config.barostat:
        ctype = jnp.complex128 if phi.dtype == "float64" else jnp.complex64
        phi_fourier = jnp.zeros((config.n_types, *config.fft_shape), dtype=ctype)

    # Save step 0 to trajectory
    if n_print > 0:
        # NOTE: we don't need to save all this stuff for the differentiable MD
        kinetic_energy = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
        trj["angle energy"] = [angle_energy]
        trj["bond energy"] = [bond_energy]
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
        trj["temperature"] = [temperature]
        trj["positions"] = [positions]
        trj["velocities"] = [velocities]
        trj["box"] = [config.box_size]
        if config.barostat:
            # Initial frame: kinetic-only pressure (virial not yet available).
            # Keep this as a JAX scalar — wrapping in float() aborts under
            # value_and_grad(simulator) on NPT systems.
            _init_kin_p = kinetic_pressure(velocities, masses, config.volume)
            trj["pressure"] = [jnp.mean(_init_kin_p * config.p_conv)]

    ###################
    # # # MD LOOP # # #
    ###################
    n_steps = equilibration if equilibration else config.n_steps

    # ── One velocity-Verlet MD step for lax.scan (NVT — no barostat) ────
    def _md_step(carry, step):
        (positions, velocities, neigh_i, neigh_j, nlist_state, key,
         bond_forces, angle_forces, dihedral_forces, improper_forces,
         LJ_forces, elec_forces, reconstr_forces,
         bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
         any_overflow) = carry

        dt = config.outer_ts

        # Maybe rebuild neighbor list
        if nlist_method == "jaxmd":
            neigh_i, neigh_j, nlist_overflow, nlist_state = (
                maybe_update_jaxmd_nlist(
                    step, ns_nlist, positions, nlist_state, n_atoms,
                    neigh_i, neigh_j, excluded_nlist_i, excluded_nlist_j,
                    excluded_for_main is not None,
                    config.box_size,
                )
            )
        else:
            neigh_i, neigh_j, nlist_overflow, nlist_state = (
                maybe_rebuild_verlet_list_jax(
                    step, ns_nlist, positions, nlist_state,
                    config.box_size, rv, skin,
                    neigh_i, neigh_j, excluded_nlist_i, excluded_nlist_j,
                    excluded_for_main is not None,
                )
            )
        any_overflow = any_overflow | nlist_overflow

        # Zero restrained atoms' forces
        if len(restr_atoms) > 0:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            reconstr_forces = zero_forces(reconstr_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- First velocity half-step (all forces) ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, dt)

        # --- Full position step ---
        positions = integrate_position(positions, velocities, dt)
        positions = jnp.mod(positions, config.box_size)

        # --- Recompute ALL forces at new positions ---

        # Bonded forces  (NVT ⇒ no virial pressure needed)
        if topol.bonds:
            bond_energy, bond_forces, _ = get_bond_energy_and_forces(
                bond_forces, positions, config.box_size, *topol.bonds_2,
                compute_pressure=False,
            )
        if topol.angles:
            angle_energy, angle_forces, _ = get_angle_energy_and_forces(
                angle_forces, positions, config.box_size, *topol.bonds_3,
                only_harmonic=topol.angle_uses_only_harmonic,
                compute_pressure=False,
            )
        if topol.dihedrals:
            dihedral_energy, dihedral_forces, _, _ = get_dihedral_energy_and_forces(
                dihedral_forces, positions, config.box_size, *topol.bonds_4,
                ff_family=config.ff_family,
                compute_pressure=False,
            )
        if topol.impropers:
            _ie, improper_forces, _ = get_impropers_energy_and_forces(
                improper_forces, positions, config.box_size, *topol.bonds_impr,
                compute_pressure=False,
            )
            dihedral_energy = dihedral_energy + _ie

        # Pair parameters from neighbor list  (precomputed distances)
        if config.coulombtype and charges is not None:
            main_rvec, main_r = compute_pair_distances(
                neigh_i, neigh_j, positions, config.box_size)
            pair_params = apply_nlist_elec_precomputed(
                neigh_i, neigh_j, main_rvec, main_r,
                charges, sgm_table, epsl_table, types,
            )
            if excluded_for_elec is not None:
                excl_rvec, excl_r = compute_pair_distances(
                    excluded_for_elec[0], excluded_for_elec[1],
                    positions, config.box_size)
                excl_pair_params = apply_nlist_elec_precomputed(
                    excluded_for_elec[0], excluded_for_elec[1],
                    excl_rvec, excl_r,
                    charges, sgm_table, epsl_table, types,
                )
            else:
                excl_pair_params = None
            if use_14_scaling:
                p14_rvec, p14_r = compute_pair_distances(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    positions, config.box_size)
                pair_params_14 = apply_nlist_elec_precomputed(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    p14_rvec, p14_r,
                    charges, sgm_table, epsl_table, types,
                )
            else:
                pair_params_14 = None
        else:
            main_rvec, main_r = compute_pair_distances(
                neigh_i, neigh_j, positions, config.box_size)
            pair_params = apply_nlist_precomputed(
                neigh_i, neigh_j, main_rvec, main_r,
                sgm_table, epsl_table, types,
            )
            excl_pair_params = None
            if use_14_scaling:
                p14_rvec, p14_r = compute_pair_distances(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    positions, config.box_size)
                pair_params_14 = apply_nlist_precomputed(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    p14_rvec, p14_r,
                    sgm_table, epsl_table, types,
                )
            else:
                pair_params_14 = None

        # LJ forces
        LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config
        )
        if use_14_scaling and pair_params_14 is not None and config.lj14_scale != 0.0:
            LJ_14_energy, LJ_14_forces = get_LJ_energy_and_forces(
                LJ_forces, pair_params_14, config
            )
            LJ_energy += config.lj14_scale * LJ_14_energy
            LJ_forces += config.lj14_scale * LJ_14_forces

        # Electrostatic forces
        if charges is not None:
            if config.coulombtype == 1:
                (
                    elec_energy,
                    _,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, charges, config, pair_params, excl_pair_params,
                    compute_potential=False,
                )
                if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                    elec_14_energy, elec_14_forces = get_coulomb_pair_energy_and_forces(
                        elec_forces,
                        pair_params_14,
                        config.coulomb14_scale,
                        config.elec_conversion,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces
            elif config.coulombtype == 2:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                ) = get_reaction_field_energy_and_forces(
                    elec_forces, pair_params, config, excl_pair_params
                )
                if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                    elec_14_energy, _, elec_14_forces = get_reaction_field_pair_energy_and_forces(
                        elec_forces,
                        pair_params_14,
                        config,
                        config.coulomb14_scale,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces

        # Dipole forces (non-amber proteins)
        if topol.dihedrals and config.ff_family != "amber_like":
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, config.box_size, *topol.bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, dip_charges, dip_fog, n_dip, config
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
            )

        # Zero restrained atoms after recompute
        if len(restr_atoms) > 0:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            reconstr_forces = zero_forces(reconstr_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- Second velocity half-step (all forces) ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, dt)

        # Thermostat
        velocities, key = apply_thermostat(velocities, key, config, masses)

        # Cancel COM momentum
        if config.cancel_com_momentum:
            should_cancel = jnp.mod(step, config.cancel_com_momentum) == 0
            velocities = jax.lax.cond(
                should_cancel,
                lambda v: cancel_com_momentum(v, masses),
                lambda v: v,
                velocities,
            )

        new_carry = (positions, velocities, neigh_i, neigh_j, nlist_state, key,
                     bond_forces, angle_forces, dihedral_forces, improper_forces,
                     LJ_forces, elec_forces, reconstr_forces,
                     bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
                     any_overflow)
        return new_carry, None

    # ── One velocity-Verlet MD step for lax.scan (NPT — with barostat) ──
    def _md_step_npt(carry, step):
        (positions, velocities, neigh_i, neigh_j, nlist_state, key,
         bond_forces, angle_forces, dihedral_forces, improper_forces,
         LJ_forces, elec_forces, reconstr_forces,
         bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
         any_overflow, box_state, _total_pressure) = carry

        dt = config.outer_ts
        box_sz = box_state.box_size

        # Maybe rebuild neighbor list
        if nlist_method == "jaxmd":
            neigh_i, neigh_j, nlist_overflow, nlist_state = (
                maybe_update_jaxmd_nlist(
                    step, ns_nlist, positions, nlist_state, n_atoms,
                    neigh_i, neigh_j, excluded_nlist_i, excluded_nlist_j,
                    excluded_for_main is not None,
                    box_sz,
                )
            )
        else:
            neigh_i, neigh_j, nlist_overflow, nlist_state = (
                maybe_rebuild_verlet_list_jax(
                    step, ns_nlist, positions, nlist_state,
                    box_sz, rv, skin,
                    neigh_i, neigh_j, excluded_nlist_i, excluded_nlist_j,
                    excluded_for_main is not None,
                )
            )
        any_overflow = any_overflow | nlist_overflow

        # Zero restrained atoms' forces
        if len(restr_atoms) > 0:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            reconstr_forces = zero_forces(reconstr_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- First velocity half-step (all forces) ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, dt)

        # --- Full position step ---
        positions = integrate_position(positions, velocities, dt)
        positions = jnp.mod(positions, box_sz)

        # --- Recompute ALL forces at new positions ---

        # Bonded forces (with pressure)
        bond_pressure = jnp.zeros(3)
        angle_pressure = jnp.zeros(3)
        dihedral_pressure = jnp.zeros(3)

        if topol.bonds:
            bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
                bond_forces, positions, box_sz, *topol.bonds_2
            )
        if topol.angles:
            angle_energy, angle_forces, angle_pressure = get_angle_energy_and_forces(
                angle_forces, positions, box_sz, *topol.bonds_3,
                only_harmonic=topol.angle_uses_only_harmonic,
            )
        if topol.dihedrals:
            dihedral_energy, dihedral_forces, _, dihedral_pressure = get_dihedral_energy_and_forces(
                dihedral_forces, positions, box_sz, *topol.bonds_4,
                ff_family=config.ff_family,
            )
        improper_pressure = jnp.zeros(3)
        if topol.impropers:
            _ie, improper_forces, improper_pressure = get_impropers_energy_and_forces(
                improper_forces, positions, box_sz, *topol.bonds_impr
            )
            dihedral_energy = dihedral_energy + _ie

        # Update config with current box for nonbonded functions
        config_npt = config.replace(
            box_size=box_state.box_size,
            volume=box_state.volume,
            volume_per_cell=box_state.volume_per_cell,
            k_vector=box_state.k_vector,
            k_meshgrid=box_state.k_meshgrid,
        )

        # Pair parameters from neighbor list (with current box, precomputed distances)
        if config.coulombtype and charges is not None:
            main_rvec, main_r = compute_pair_distances(
                neigh_i, neigh_j, positions, box_sz)
            pair_params = apply_nlist_elec_precomputed(
                neigh_i, neigh_j, main_rvec, main_r,
                charges, sgm_table, epsl_table, types,
            )
            if excluded_for_elec is not None:
                excl_rvec, excl_r = compute_pair_distances(
                    excluded_for_elec[0], excluded_for_elec[1],
                    positions, box_sz)
                excl_pair_params = apply_nlist_elec_precomputed(
                    excluded_for_elec[0], excluded_for_elec[1],
                    excl_rvec, excl_r,
                    charges, sgm_table, epsl_table, types,
                )
            else:
                excl_pair_params = None
            if use_14_scaling:
                p14_rvec, p14_r = compute_pair_distances(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    positions, box_sz)
                pair_params_14 = apply_nlist_elec_precomputed(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    p14_rvec, p14_r,
                    charges, sgm_table, epsl_table, types,
                )
            else:
                pair_params_14 = None
        else:
            main_rvec, main_r = compute_pair_distances(
                neigh_i, neigh_j, positions, box_sz)
            pair_params = apply_nlist_precomputed(
                neigh_i, neigh_j, main_rvec, main_r,
                sgm_table, epsl_table, types,
            )
            excl_pair_params = None
            if use_14_scaling:
                p14_rvec, p14_r = compute_pair_distances(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    positions, box_sz)
                pair_params_14 = apply_nlist_precomputed(
                    topol.one_four_pairs[0], topol.one_four_pairs[1],
                    p14_rvec, p14_r,
                    sgm_table, epsl_table, types,
                )
            else:
                pair_params_14 = None

        # NPT LJ forces + pressure
        LJ_energy, LJ_forces, LJ_pressure = get_LJ_energy_and_forces_npt(
            LJ_forces, pair_params, config_npt
        )
        if use_14_scaling and pair_params_14 is not None and config.lj14_scale != 0.0:
            LJ_14_energy, LJ_14_forces, LJ_14_pressure = get_LJ_energy_and_forces_npt(
                LJ_forces, pair_params_14, config_npt
            )
            LJ_energy += config.lj14_scale * LJ_14_energy
            LJ_forces += config.lj14_scale * LJ_14_forces
            LJ_pressure += config.lj14_scale * LJ_14_pressure

        # NPT electrostatics + pressure
        elec_pressure = jnp.zeros(3)
        if charges is not None:
            if config.coulombtype == 1:
                (
                    elec_energy,
                    _,
                    elec_forces,
                    pme_real_pressure,
                ) = get_elec_energy_potential_and_forces_npt(
                    positions, charges, config_npt,
                    pair_params, excl_pair_params,
                    compute_potential=False,
                )
                elec_pressure += pme_real_pressure
                if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                    elec_14_energy, elec_14_forces, elec_14_pressure = get_coulomb_pair_energy_and_forces_npt(
                        elec_forces,
                        pair_params_14,
                        config.coulomb14_scale,
                        config.elec_conversion,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces
                    elec_pressure += elec_14_pressure
            elif config.coulombtype == 2:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                    rf_pressure,
                ) = get_reaction_field_energy_and_forces_npt(
                    elec_forces, pair_params, config_npt, excl_pair_params
                )
                elec_pressure += rf_pressure
                if use_14_scaling and pair_params_14 is not None and config.coulomb14_scale != 0.0:
                    elec_14_energy, _, elec_14_forces, rf_14_pressure = get_reaction_field_pair_energy_and_forces_npt(
                        elec_forces,
                        pair_params_14,
                        config_npt,
                        config.coulomb14_scale,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces
                    elec_pressure += rf_14_pressure

        # Dipole forces (non-amber proteins)
        if topol.dihedrals and config.ff_family != "amber_like":
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, box_sz, *topol.bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, dip_charges, dip_fog, n_dip, config_npt
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
            )

        # Zero restrained atoms after recompute
        if len(restr_atoms) > 0:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            reconstr_forces = zero_forces(reconstr_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- Second velocity half-step (all forces) ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, dt)

        # Thermostat
        velocities, key = apply_thermostat(velocities, key, config, masses)

        # ── Compute total instantaneous pressure ──────────────────────────
        volume = box_state.volume
        virial_pressure = (
            bond_pressure + angle_pressure + dihedral_pressure
            + improper_pressure + LJ_pressure + elec_pressure
        )
        kin_pressure = kinetic_pressure(velocities, masses, volume)
        total_pressure = kin_pressure + virial_pressure / volume

        # ── Apply barostat ────────────────────────────────────────────────
        if config.barostat == 1:
            positions, box_state = berendsen(
                total_pressure, positions, box_state, config
            )
        elif config.barostat == 2:
            positions, velocities, box_state, key = c_rescale(
                total_pressure, positions, velocities, box_state, config, key
            )

        # Re-wrap positions into new box
        positions = jnp.mod(positions, box_state.box_size)

        # Cancel COM momentum
        if config.cancel_com_momentum:
            should_cancel = jnp.mod(step, config.cancel_com_momentum) == 0
            velocities = jax.lax.cond(
                should_cancel,
                lambda v: cancel_com_momentum(v, masses),
                lambda v: v,
                velocities,
            )

        new_carry = (positions, velocities, neigh_i, neigh_j, nlist_state, key,
                     bond_forces, angle_forces, dihedral_forces, improper_forces,
                     LJ_forces, elec_forces, reconstr_forces,
                     bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
                     any_overflow, box_state, total_pressure)
        return new_carry, None

    # ── Build initial carry ───────────────────────────────────────────────
    if config.barostat:
        box_state = BoxState.from_config(config)
        carry = (
            positions, velocities, neigh_i, neigh_j, nlist_state, key,
            bond_forces, angle_forces, dihedral_forces, improper_forces,
            LJ_forces, elec_forces, reconstr_forces,
            bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
            jnp.array(False),  # any_overflow
            box_state,
            jnp.zeros(3),  # total_pressure
        )
        # ``jax.checkpoint`` recomputes the step body in the backward pass
        # to save memory.  Skip it on forward-only callers (mdrun, MD-quality
        # tools) — re-tracing it costs ~2x forward-only wall time.
        _md_step_ckpt = jax.checkpoint(_md_step_npt) if differentiable else _md_step_npt
    else:
        carry = (
            positions, velocities, neigh_i, neigh_j, nlist_state, key,
            bond_forces, angle_forces, dihedral_forces, improper_forces,
            LJ_forces, elec_forces, reconstr_forces,
            bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
            jnp.array(False),  # any_overflow
        )
        _md_step_ckpt = jax.checkpoint(_md_step) if differentiable else _md_step

    # ── Chunked lax.scan — collect trajectory every n_print steps ─────────
    if n_print > 0:
        chunk_size = n_print
        n_chunks = n_steps // chunk_size
        remainder = n_steps % chunk_size
    else:
        chunk_size = n_steps
        n_chunks = 1
        remainder = 0

    for chunk in range(n_chunks):
        step_indices = jnp.arange(
            chunk * chunk_size + 1, (chunk + 1) * chunk_size + 1
        )
        carry, _ = jax.lax.scan(_md_step_ckpt, carry, step_indices)

        if n_print > 0:
            if config.barostat:
                (pos_c, vel_c, _ni, _nj, _nls, _k,
                 bf_c, af_c, df_c, impf_c,
                 ljf_c, ef_c, rf_c,
                 be_c, ae_c, de_c, lje_c, ee_c,
                 _ovf, box_state_c, total_pressure_c) = carry
                current_box = box_state_c.box_size
            else:
                (pos_c, vel_c, _ni, _nj, _nls, _k,
                 bf_c, af_c, df_c, impf_c,
                 ljf_c, ef_c, rf_c,
                 be_c, ae_c, de_c, lje_c, ee_c,
                 _ovf) = carry
                current_box = config.box_size
            ke = 0.5 * jnp.sum(masses * jnp.sum(vel_c**2, axis=1))
            temp = (2 / 3) * ke / (config.R * config.n_particles)
            trj["angle energy"].append(ae_c)
            trj["bond energy"].append(be_c)
            trj["dihedral energy"].append(de_c)
            trj["elec energy"].append(ee_c)
            trj["LJ energy"].append(lje_c)
            trj["forces"].append(
                bf_c + af_c + df_c + impf_c + ljf_c + rf_c + ef_c
            )
            trj["kinetic energy"].append(ke)
            trj["temperature"].append(temp)
            trj["positions"].append(pos_c)
            trj["velocities"].append(vel_c)
            trj["box"].append(current_box)
            if config.barostat:
                # Scalar mean pressure in bar for energy.log; consumers
                # (file_io._safe_float, np.asarray) handle the 0-d JAX scalar.
                trj["pressure"].append(jnp.mean(total_pressure_c * config.p_conv))

    # Run remaining steps (no trajectory output for partial chunk)
    if remainder > 0:
        step_indices = jnp.arange(
            n_chunks * chunk_size + 1, n_steps + 1
        )
        carry, _ = jax.lax.scan(_md_step_ckpt, carry, step_indices)

    # Unpack final carry
    if config.barostat:
        (positions, velocities, neigh_i, neigh_j, nlist_state, key,
         bond_forces, angle_forces, dihedral_forces, improper_forces,
         LJ_forces, elec_forces, reconstr_forces,
         bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
         any_overflow, box_state, _) = carry
        # Update config with final box for downstream code
        config = config.replace(
            box_size=box_state.box_size,
            volume=box_state.volume,
            volume_per_cell=box_state.volume_per_cell,
            k_vector=box_state.k_vector,
            k_meshgrid=box_state.k_meshgrid,
        )
    else:
        (positions, velocities, neigh_i, neigh_j, nlist_state, key,
         bond_forces, angle_forces, dihedral_forces, improper_forces,
         LJ_forces, elec_forces, reconstr_forces,
         bond_energy, angle_energy, dihedral_energy, LJ_energy, elec_energy,
         any_overflow) = carry

    # Store overflow flag in trajectory so the caller can check it as a
    # plain Python bool AFTER the function returns (outside jit/vgrad scope).
    # Do NOT use jax.debug.callback here: on some HPC environments
    # (Olivia Singularity) the XLA outfeed mechanism it relies on is not
    # supported and causes a CUDA_ERROR_LAUNCH_FAILED crash at first run.
    trj["nlist_overflow"] = any_overflow

    return trj, key, config
