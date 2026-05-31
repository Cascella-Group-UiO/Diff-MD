import datetime
import logging
import os
import warnings
from functools import partial

import jax
from jax import config as jax_config
from jax import jit, lax
import jax.numpy as jnp
import numpy as onp
from jax import random
from vesin import NeighborList

from .barostat import berendsen, c_rescale, kinetic_pressure
from .config import BoxState
from .file_io import (
    OutDataset, reconnect_for_append, store_data, store_static, truncate_time_series,
    _safe_float, format_energy_log_line, _ENERGY_LOG_HEADER,
)
from .force import (
    get_angle_energy_and_forces,
    get_bond_energy_and_forces,
    get_dihedral_energy_and_forces,
    get_impropers_energy_and_forces,
    get_protein_dipoles,
    redistribute_dipole_forces,
)
from .input_parser import System
from .integrator import integrate_position, integrate_velocity, zero_velocities, zero_forces
from .logger import Logger, format_timedelta
from .nonbonded import (
    get_coulomb_pair_energy_and_forces,
    get_coulomb_pair_energy_and_forces_npt,
    get_dipole_forces,
    get_elec_energy_potential_and_forces,
    get_elec_energy_potential_and_forces_npt,
    get_LJ_energy_and_forces,
    get_LJ_energy_and_forces_npt,
    get_reaction_field_pair_energy_and_forces,
    get_reaction_field_pair_energy_and_forces_npt,
    get_reaction_field_energy_and_forces,
    get_reaction_field_energy_and_forces_npt,
)
from .thermostat import (
    apply_thermostat,
    cancel_com_momentum,
    generate_initial_velocities,
    translational_dof,
)
from .neighbor_list import (
    apply_nlist, 
    apply_nlist_elec, 
    build_neighbor_list_cell,
    maybe_rebuild_verlet_list_jax,
    exclude_bonded_neighbors,
    apply_nlist_general,
    init_jaxmd_neighbor_list,
    maybe_update_jaxmd_nlist,
    resolve_verlet_radii,
)
from .diagnostics import print_startup_diagnostics, format_simulation_config, format_run_header


def _format_step_log(
    step,
    n_steps,
    elapsed,
    ns_per_day,
    hours_per_ns,
    steps_per_s,
    temperature,
    pressure,
    kinetic_energy,
    bond_energy,
    angle_energy,
    dihedral_energy,
    lj_energy,
    elec_energy,
):
    elapsed_str = (
        f"{elapsed.days:.0f}-{elapsed.seconds // 3600:02d}:"
        f"{(elapsed.seconds % 3600) // 60:02d}:{elapsed.seconds % 60:02d}"
    )
    perf_line = (
        f"Step {step:>8d}/{n_steps:<8d} | Elapsed {elapsed_str} | "
        f"{ns_per_day:8.3f} ns/day | {hours_per_ns:8.3f} h/ns | {steps_per_s:8.3f} step/s"
    )
    thermo_line = (
        "Thermo | "
        f"T={_safe_float(temperature):10.4f} K | "
        f"P={_safe_float(pressure):10.4f} bar | "
        f"Ekin={_safe_float(kinetic_energy):12.6f} | "
        f"Eb={_safe_float(bond_energy):12.6f} | "
        f"Ea={_safe_float(angle_energy):12.6f} | "
        f"Ed={_safe_float(dihedral_energy):12.6f} | "
        f"ELJ={_safe_float(lj_energy):12.6f} | "
        f"Eelec={_safe_float(elec_energy):12.6f}"
    )
    return perf_line, thermo_line


# _format_energy_log_line is now format_energy_log_line imported from file_io
# Keep local alias for backward compatibility with the rest of this file
_format_energy_log_line = format_energy_log_line


def _build_verlet_step_fn(
    config, topol, system, use_14_scaling, restr_atoms,
    excluded_for_elec,
):
    """Build a JIT-compiled single velocity-Verlet step function.

    All Python-level branching (topology flags, coulomb type, etc.) is resolved
    at trace time through the closure.  The returned function operates on pure
    JAX arrays and can be compiled once and reused for every MD step.
    """
    has_charges = system.charges is not None
    has_elec_excl = excluded_for_elec is not None
    has_restr = len(restr_atoms) > 0
    coulombtype = config.coulombtype
    has_14 = use_14_scaling
    has_dipoles = topol.dihedrals and config.ff_family != "amber_like"

    # Capture immutable topology arrays for the closure
    charges = system.charges
    masses = system.masses
    types = system.types
    box_size = config.box_size
    sgm_table = config.sgm_table
    epsl_table = config.epsl_table
    outer_ts = config.outer_ts

    # Optional topology arrays
    bonds_2 = topol.bonds_2 if topol.bonds else None
    bonds_3 = topol.bonds_3 if topol.angles else None
    bonds_4 = topol.bonds_4 if topol.dihedrals else None
    bonds_impr = topol.bonds_impr if topol.impropers else None
    bonds_d = topol.bonds_d if has_dipoles else None
    one_four_pairs = topol.one_four_pairs if has_14 else None
    excl_elec_i = excluded_for_elec[0] if has_elec_excl else None
    excl_elec_j = excluded_for_elec[1] if has_elec_excl else None

    @jit
    def _step(positions, velocities,
              LJ_forces, elec_forces, reconstr_forces,
              bond_forces, angle_forces, dihedral_forces, improper_forces,
              LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
              neigh_i, neigh_j, key,
              pair_params_14, excl_pair_params):
        """Execute one velocity-Verlet step.  Pure JAX, fully JIT-compiled."""
        # --- Zero restrained atoms' forces ---
        if has_restr:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- First velocity half-step ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, outer_ts)

        # --- Full position step ---
        positions = integrate_position(positions, velocities, outer_ts)
        positions = jnp.mod(positions, box_size)

        # --- Recompute all forces at new positions ---

        # Bonded forces  (NVT ⇒ no virial pressure needed)
        if topol.bonds:
            bond_energy, bond_forces, _ = get_bond_energy_and_forces(
                bond_forces, positions, box_size, *bonds_2,
                compute_pressure=False,
            )
        if topol.angles:
            angle_energy, angle_forces, _ = get_angle_energy_and_forces(
                angle_forces, positions, box_size, *bonds_3,
                only_harmonic=topol.angle_uses_only_harmonic,
                compute_pressure=False,
            )
        torsional_energy = 0.0
        improper_energy = 0.0
        if topol.dihedrals:
            (
                dih_e,
                dihedral_forces,
                _,
                _,
            ) = get_dihedral_energy_and_forces(
                dihedral_forces,
                positions,
                box_size,
                *bonds_4,
                ff_family=config.ff_family,
                compute_pressure=False,
            )
            torsional_energy = dih_e
        if topol.impropers:
            improper_energy, improper_forces, _ = get_impropers_energy_and_forces(
                improper_forces, positions, box_size, *bonds_impr,
                compute_pressure=False,
            )
        dihedral_energy = torsional_energy + improper_energy

        # Pair parameters from neighbor list
        if coulombtype and has_charges:
            pair_params = apply_nlist_elec(
                neigh_i, neigh_j, positions, charges,
                box_size, sgm_table, epsl_table, types,
            )
            if has_elec_excl:
                excl_pair_params = apply_nlist_elec(
                    excl_elec_i, excl_elec_j, positions, charges,
                    box_size, sgm_table, epsl_table, types,
                )
            if has_14:
                pair_params_14 = apply_nlist_elec(
                    one_four_pairs[0], one_four_pairs[1], positions, charges,
                    box_size, sgm_table, epsl_table, types,
                )
        else:
            pair_params = apply_nlist(
                neigh_i, neigh_j, positions,
                box_size, sgm_table, epsl_table, types,
            )
            if has_14:
                pair_params_14 = apply_nlist(
                    one_four_pairs[0], one_four_pairs[1], positions,
                    box_size, sgm_table, epsl_table, types,
                )

        # LJ forces
        LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config
        )
        if has_14:
            LJ_14_energy, LJ_14_forces = get_LJ_energy_and_forces(
                LJ_forces, pair_params_14, config
            )
            LJ_energy += config.lj14_scale * LJ_14_energy
            LJ_forces += config.lj14_scale * LJ_14_forces

        # Electrostatic forces
        if has_charges:
            if coulombtype == 1:
                (
                    elec_energy,
                    _elec_potential,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, charges, config, pair_params, excl_pair_params
                )
                if has_14 and config.coulomb14_scale != 0.0:
                    elec_14_energy, elec_14_forces = get_coulomb_pair_energy_and_forces(
                        elec_forces,
                        pair_params_14,
                        config.coulomb14_scale,
                        config.elec_conversion,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces
            elif coulombtype == 2:
                (
                    elec_energy,
                    _elec_potential,
                    elec_forces,
                ) = get_reaction_field_energy_and_forces(
                    elec_forces, pair_params, config, excl_pair_params
                )
                if has_14 and config.coulomb14_scale != 0.0:
                    elec_14_energy, _, elec_14_forces = get_reaction_field_pair_energy_and_forces(
                        elec_forces,
                        pair_params_14,
                        config,
                        config.coulomb14_scale,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces

        # Dipole forces (non-amber proteins)
        if has_dipoles:
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, box_size, *bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, _step.dip_charges, _step.dip_fog, _step.n_dip, config
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *bonds_d
            )

        # Zero restrained atoms after recompute
        if has_restr:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- Second velocity half-step ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, outer_ts)

        # Thermostat
        velocities, key = apply_thermostat(velocities, key, config, masses)

        return (positions, velocities,
                LJ_forces, elec_forces, reconstr_forces,
                bond_forces, angle_forces, dihedral_forces, improper_forces,
                LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                key, pair_params_14, excl_pair_params)

    # Attach dipole constants (set by main() if needed)
    _step.dip_charges = None
    _step.dip_fog = None
    _step.n_dip = None

    return _step


def _build_verlet_step_fn_npt(
    config, topol, system, use_14_scaling, restr_atoms,
    excluded_for_elec,
):
    """Build a JIT-compiled single velocity-Verlet step for NPT ensembles.

    Like ``_build_verlet_step_fn`` but takes ``box_size`` as an explicit
    parameter (dynamic, from the carry) and uses ``_npt`` force variants that
    return pressure contributions.
    """
    has_charges = system.charges is not None
    has_elec_excl = excluded_for_elec is not None
    has_restr = len(restr_atoms) > 0
    coulombtype = config.coulombtype
    has_14 = use_14_scaling
    has_dipoles = topol.dihedrals and config.ff_family != "amber_like"

    charges = system.charges
    masses = system.masses
    types = system.types
    sgm_table = config.sgm_table
    epsl_table = config.epsl_table
    outer_ts = config.outer_ts

    bonds_2 = topol.bonds_2 if topol.bonds else None
    bonds_3 = topol.bonds_3 if topol.angles else None
    bonds_4 = topol.bonds_4 if topol.dihedrals else None
    bonds_impr = topol.bonds_impr if topol.impropers else None
    bonds_d = topol.bonds_d if has_dipoles else None
    one_four_pairs = topol.one_four_pairs if has_14 else None
    excl_elec_i = excluded_for_elec[0] if has_elec_excl else None
    excl_elec_j = excluded_for_elec[1] if has_elec_excl else None

    @jit
    def _step(positions, velocities,
              LJ_forces, elec_forces, reconstr_forces,
              bond_forces, angle_forces, dihedral_forces, improper_forces,
              LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
              neigh_i, neigh_j, key,
              pair_params_14, excl_pair_params,
              box_state):
        """One NPT velocity-Verlet step.  Returns pressure contributions."""
        box_size = box_state.box_size

        # --- Zero restrained atoms' forces ---
        if has_restr:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- First velocity half-step ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, outer_ts)

        # --- Full position step ---
        positions = integrate_position(positions, velocities, outer_ts)
        positions = jnp.mod(positions, box_size)

        # --- Recompute all forces at new positions ---

        # Bonded forces (with pressure)
        bond_pressure = jnp.zeros(3)
        angle_pressure = jnp.zeros(3)
        dihedral_pressure = jnp.zeros(3)

        if topol.bonds:
            bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
                bond_forces, positions, box_size, *bonds_2
            )
        if topol.angles:
            angle_energy, angle_forces, angle_pressure = get_angle_energy_and_forces(
                angle_forces, positions, box_size, *bonds_3,
                only_harmonic=topol.angle_uses_only_harmonic,
            )
        torsional_energy = 0.0
        improper_energy = 0.0
        if topol.dihedrals:
            (
                dih_e,
                dihedral_forces,
                _,
                dihedral_pressure,
            ) = get_dihedral_energy_and_forces(
                dihedral_forces,
                positions,
                box_size,
                *bonds_4,
                ff_family=config.ff_family,
            )
            torsional_energy = dih_e
        improper_pressure = jnp.zeros(3)
        if topol.impropers:
            improper_energy, improper_forces, improper_pressure = get_impropers_energy_and_forces(
                improper_forces, positions, box_size, *bonds_impr
            )
        dihedral_energy = torsional_energy + improper_energy

        # Config with current box for nonbonded functions
        config_npt = config.replace(
            box_size=box_state.box_size,
            volume=box_state.volume,
            volume_per_cell=box_state.volume_per_cell,
            k_vector=box_state.k_vector,
            k_meshgrid=box_state.k_meshgrid,
        )

        # Pair parameters from neighbor list (with current box)
        if coulombtype and has_charges:
            pair_params = apply_nlist_elec(
                neigh_i, neigh_j, positions, charges,
                box_size, sgm_table, epsl_table, types,
            )
            if has_elec_excl:
                excl_pair_params = apply_nlist_elec(
                    excl_elec_i, excl_elec_j, positions, charges,
                    box_size, sgm_table, epsl_table, types,
                )
            if has_14:
                pair_params_14 = apply_nlist_elec(
                    one_four_pairs[0], one_four_pairs[1], positions, charges,
                    box_size, sgm_table, epsl_table, types,
                )
        else:
            pair_params = apply_nlist(
                neigh_i, neigh_j, positions,
                box_size, sgm_table, epsl_table, types,
            )
            if has_14:
                pair_params_14 = apply_nlist(
                    one_four_pairs[0], one_four_pairs[1], positions,
                    box_size, sgm_table, epsl_table, types,
                )

        # NPT LJ forces + pressure
        LJ_energy, LJ_forces, LJ_pressure = get_LJ_energy_and_forces_npt(
            LJ_forces, pair_params, config_npt
        )
        if has_14:
            LJ_14_energy, LJ_14_forces, LJ_14_pressure = get_LJ_energy_and_forces_npt(
                LJ_forces, pair_params_14, config_npt
            )
            LJ_energy += config.lj14_scale * LJ_14_energy
            LJ_forces += config.lj14_scale * LJ_14_forces
            LJ_pressure += config.lj14_scale * LJ_14_pressure

        # NPT electrostatic forces + pressure
        elec_pressure = jnp.zeros(3)
        if has_charges:
            if coulombtype == 1:
                (
                    elec_energy,
                    _elec_potential,
                    elec_forces,
                    pme_real_pressure,
                ) = get_elec_energy_potential_and_forces_npt(
                    positions, charges, config_npt,
                    pair_params, excl_pair_params,
                )
                elec_pressure += pme_real_pressure
                if has_14 and config.coulomb14_scale != 0.0:
                    elec_14_energy, elec_14_forces, elec_14_pressure = get_coulomb_pair_energy_and_forces_npt(
                        elec_forces,
                        pair_params_14,
                        config.coulomb14_scale,
                        config.elec_conversion,
                    )
                    elec_energy += elec_14_energy
                    elec_forces += elec_14_forces
                    elec_pressure += elec_14_pressure
            elif coulombtype == 2:
                (
                    elec_energy,
                    _elec_potential,
                    elec_forces,
                    rf_pressure,
                ) = get_reaction_field_energy_and_forces_npt(
                    elec_forces, pair_params, config_npt, excl_pair_params
                )
                elec_pressure += rf_pressure
                if has_14 and config.coulomb14_scale != 0.0:
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
        if has_dipoles:
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, box_size, *bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, _step.dip_charges, _step.dip_fog, _step.n_dip, config_npt
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *bonds_d
            )

        # Zero restrained atoms after recompute
        if has_restr:
            LJ_forces = zero_forces(LJ_forces, restr_atoms)
            elec_forces = zero_forces(elec_forces, restr_atoms)
            bond_forces = zero_forces(bond_forces, restr_atoms)
            angle_forces = zero_forces(angle_forces, restr_atoms)
            dihedral_forces = zero_forces(dihedral_forces, restr_atoms)
            improper_forces = zero_forces(improper_forces, restr_atoms)

        # --- Second velocity half-step ---
        total_accel = (
            LJ_forces + elec_forces + reconstr_forces
            + bond_forces + angle_forces
            + dihedral_forces + improper_forces
        ) / config.mass
        velocities = integrate_velocity(velocities, total_accel, outer_ts)

        # Thermostat
        velocities, key = apply_thermostat(velocities, key, config, masses)

        # Total virial pressure
        virial_pressure = (
            bond_pressure + angle_pressure + dihedral_pressure
            + improper_pressure + LJ_pressure + elec_pressure
        )

        return (positions, velocities,
                LJ_forces, elec_forces, reconstr_forces,
                bond_forces, angle_forces, dihedral_forces, improper_forces,
                LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                key, pair_params_14, excl_pair_params,
                virial_pressure)

    _step.dip_charges = None
    _step.dip_fog = None
    _step.n_dip = None

    return _step


# pyright: reportUnboundVariable=none
def main(args):
    start_time = datetime.datetime.now()

    # Print startup diagnostics (hardware, JAX config, etc.)
    print_startup_diagnostics(title="DIFF-MD MDRUN", logger=Logger.rank0)

    # ---- Early precision setup (must happen before ANY JAX array creation) ----
    # Read ensemble from TOML to auto-enable float64 for NVE.
    # `ensemble` can live in [simulation], [atomistic_ff], or at top level,
    # so we search all sections.
    import tomllib
    _config_path = os.path.join(args.destdir, args.config)
    with open(_config_path, "rb") as _f:
        _toml = tomllib.load(_f)
    _ensemble = "NVT"
    for _section in (_toml, *(_toml[k] for k in _toml if isinstance(_toml[k], dict))):
        if "ensemble" in _section:
            _ensemble = str(_section["ensemble"]).upper()
            break

    cdtype = jnp.complex64
    if args.double_precision:
        jax_config.update("jax_enable_x64", True)
        cdtype = jnp.complex128
    elif _ensemble == "NVE":
        Logger.rank0.info(
            "ensemble = 'NVE': auto-enabling double precision (jax_enable_x64) "
            "for energy conservation.  Use --double-precision to silence this."
        )
        jax_config.update("jax_enable_x64", True)
        cdtype = jnp.complex128

    key = random.PRNGKey(args.seed)

    # Load data to System class
    system = System.constructor(args)
    config = system.config
    Logger.rank0.info(f"{config}")

    # Rich run-configuration header
    Logger.rank0.info("\n" + format_run_header(
        title="DIFF-MD MDRUN",
        config=config,
        restart_path=getattr(args, "restart", None) or (args.output if getattr(args, "append", False) else None),
    ))

    # Upcast positions/velocities to float64 when running in double precision,
    # e.g. when restarting from an H5MD file that stored float32 output.
    if jax.config.jax_enable_x64:
        system.positions = system.positions.astype(jnp.float64)
        system.velocities = system.velocities.astype(jnp.float64)

    topol = system.topol
    positions = jnp.mod(system.positions, config.box_size)
    velocities = system.velocities
    restr_atoms = topol.restraints # index of atoms to exclude from integration

    # ── When appending, load positions & velocities from the output
    #    trajectory's last frame so the simulation truly continues. ────────
    _is_append = getattr(args, "append", False)
    _restart_box_size = None  # set during append for NPT box continuity
    if _is_append:
        import h5py as _h5
        _out_path = os.path.join(args.destdir, f"{args.output}.h5")
        if not os.path.exists(_out_path):
            raise FileNotFoundError(
                f"--append requested but output file '{_out_path}' not found."
            )
        with _h5.File(_out_path, "r") as _traj:
            _pos_dset = _traj["particles/all/position/value"]
            _n_frames = _pos_dset.shape[0]

            # Scan backwards to find the last valid frame.
            # A frame is invalid if all positions are zero OR any are NaN.
            _last_frame = None
            _skipped = 0
            for _fi in range(_n_frames - 1, -1, -1):
                _frame_pos = onp.array(_pos_dset[_fi])
                if onp.all(_frame_pos == 0.0):
                    _skipped += 1
                    continue
                if onp.any(onp.isnan(_frame_pos)):
                    _skipped += 1
                    continue
                _last_frame = _fi
                break

            if _last_frame is None:
                raise ValueError(
                    "--append: all frames in the output trajectory are "
                    "corrupt (zeros or NaN). Cannot restart."
                )
            if _skipped > 0:
                Logger.rank0.warning(
                    "Auto-rollback: skipped %d corrupt trailing frame(s) "
                    "in '%s'; restarting from frame %d / %d.",
                    _skipped, _out_path, _last_frame, _n_frames - 1,
                )

            positions = jnp.array(onp.array(_pos_dset[_last_frame]))

            if "velocity" in _traj["particles/all"]:
                _vel_frame = onp.array(
                    _traj["particles/all/velocity/value"][_last_frame]
                )
                if onp.any(onp.isnan(_vel_frame)):
                    Logger.rank0.warning(
                        "Velocities at frame %d contain NaN; "
                        "re-initialising from Maxwell-Boltzmann at %.1f K.",
                        _last_frame, config.target_temperature,
                    )
                    # Maxwell-Boltzmann: v_i ~ N(0, sqrt(kT/m_i))
                    # kB in kJ/(mol·K) = 8.314462e-3
                    _kT = 8.314462e-3 * config.target_temperature
                    _masses_col = onp.array(system.masses).reshape(-1, 1)
                    _sigma_v = onp.sqrt(_kT / _masses_col)
                    velocities = jnp.array(
                        (onp.random.default_rng().normal(
                            size=positions.shape
                        ) * _sigma_v).astype(onp.float32)
                    )
                else:
                    velocities = jnp.array(_vel_frame)

            # NPT: restore the last box so restart continues at the
            # correct volume, not the original options.toml box.
            if bool(config.barostat):
                _box_path = "particles/all/box/edges/value"
                if _box_path in _traj:
                    _box_raw = onp.array(_traj[_box_path][_last_frame])
                    if _box_raw.ndim == 2 and _box_raw.shape == (3, 3):
                        _restart_box_size = jnp.array(
                            onp.diagonal(_box_raw).copy()
                        )
                    elif _box_raw.ndim == 1 and _box_raw.shape[0] == 3:
                        _restart_box_size = jnp.array(_box_raw.copy())
                    if _restart_box_size is not None:
                        Logger.rank0.info(
                            "NPT restart: restoring box from frame %d: "
                            "%.4f x %.4f x %.4f nm",
                            _last_frame,
                            float(_restart_box_size[0]),
                            float(_restart_box_size[1]),
                            float(_restart_box_size[2]),
                        )

        _wrap_box = _restart_box_size if _restart_box_size is not None else config.box_size
        positions = jnp.mod(positions, _wrap_box)
        if jax.config.jax_enable_x64:
            positions = positions.astype(jnp.float64)
            velocities = velocities.astype(jnp.float64)
        Logger.rank0.info(
            "Append mode: loaded positions/velocities from output "
            "trajectory frame %d.", _last_frame,
        )

    use_14_scaling = (
        config.ff_family == "amber_like"
        and topol.one_four_pairs is not None
        and (config.lj14_scale != 1.0 or config.coulomb14_scale != 1.0)
    )

    # When nrexcl >= 3, excluded_pairs already contains 1-4 pairs (bond distance 3),
    # so we must NOT concatenate one_four_pairs again to avoid double-counting.
    need_concat_14 = use_14_scaling and config.nrexcl < 3

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

    if _is_append:
        # On append restart, keep the velocities loaded from the trajectory;
        # only remove COM drift if requested.
        if config.cancel_com_momentum:
            velocities = cancel_com_momentum(velocities, system.masses)
    elif config.start_temperature:
        key, subkey = random.split(key)
        velocities = generate_initial_velocities(velocities, subkey, config, system.masses)
    elif config.cancel_com_momentum:
        velocities = cancel_com_momentum(velocities, system.masses)

    # Initialize forces
    LJ_forces = jnp.zeros_like(positions)
    bond_forces = jnp.zeros_like(positions)
    angle_forces = jnp.zeros_like(positions)
    dihedral_forces = jnp.zeros_like(positions)
    improper_forces = jnp.zeros_like(positions)
    elec_forces = jnp.zeros_like(positions)
    reconstr_forces = jnp.zeros_like(positions)

    # Initialize energies
    bond_energy = 0.0
    angle_energy = 0.0
    dihedral_energy = 0.0
    torsional_energy = 0.0
    improper_energy = 0.0
    field_energy = 0.0
    LJ_energy = 0.0
    elec_energy = 0.0

    # Initialize pressure
    bond_pressure, angle_pressure, dihedral_pressure, LJ_pressure, elec_pressure = 0, 0, 0, 0, 0

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
        eff_search = r_cutoff_phys + jaxmd_skin
        Logger.rank0.info(
            f"Using jax-md neighbor list (OrderedSparse, "
            f"r_cut={r_cutoff_phys:.3f}, skin={jaxmd_skin:.3f}, "
            f"effective_rv={eff_search:.3f}, "
            f"capacity_multiplier={cap_mult}, "
            f"capacity={nbrs.idx.shape[1]})"
        )
    else:
        dens = config.n_particles / config.box_size.prod()
        max_neighbors = int((1/2) * config.n_particles * ( 4 * jnp.pi * rv**3 / 3 ) * dens)
        max_neighbors += 5000 # Add a buffer for safety
        neigh_i, neigh_j, max_neighbors = build_neighbor_list_cell(
            positions, config.box_size, rv, max_neighbors
        )
    ref_positions = jnp.array(positions)  # snapshot for Verlet displacement check

    if excluded_for_main is not None:
        neigh_i, neigh_j = exclude_bonded_neighbors(
            neigh_i, neigh_j, excluded_for_main[0], excluded_for_main[1]
        )

    # NOTE: This can probably be cleaned up
    if config.coulombtype and system.charges is not None:
        pair_params = apply_nlist_elec(
            neigh_i, 
            neigh_j, 
            positions, 
            system.charges, 
            config.box_size, 
            config.sgm_table, 
            config.epsl_table, 
            system.types
        )
        if excluded_for_elec is not None:
            excl_pair_params = apply_nlist_elec(
            excluded_for_elec[0],
            excluded_for_elec[1],
                positions,
                system.charges,
                config.box_size,
                config.sgm_table,
                config.epsl_table,
                system.types,
            )
        else:
            excl_pair_params = None
        if use_14_scaling:
            pair_params_14 = apply_nlist_elec(
                topol.one_four_pairs[0],
                topol.one_four_pairs[1],
                positions,
                system.charges,
                config.box_size,
                config.sgm_table,
                config.epsl_table,
                system.types,
            )
        else:
            pair_params_14 = None
    else:
        pair_params = apply_nlist(
            neigh_i, 
            neigh_j, 
            positions, 
            config.box_size, 
            config.sgm_table, 
            config.epsl_table, 
            system.types
        )
        excl_pair_params = None
        if use_14_scaling:
            pair_params_14 = apply_nlist(
                topol.one_four_pairs[0],
                topol.one_four_pairs[1],
                positions,
                config.box_size,
                config.sgm_table,
                config.epsl_table,
                system.types,
            )
        else:
            pair_params_14 = None


    if system.charges is not None:
        if config.coulombtype == 1:
            elec_energy, elec_potential, elec_forces = get_elec_energy_potential_and_forces(
                positions, system.charges, config, pair_params, excl_pair_params
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
                        
    if topol.bonds:
        bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
            bond_forces, positions, config.box_size, *topol.bonds_2
        )
    if topol.angles:
        angle_energy, angle_forces, angle_pressure = get_angle_energy_and_forces(
            angle_forces, positions, config.box_size, *topol.bonds_3,
            only_harmonic=topol.angle_uses_only_harmonic,
        )
    torsional_energy = 0.0
    improper_energy = 0.0
    if topol.dihedrals:
        (
            dihedral_energy,
            dihedral_forces,
            _,
            dihedral_pressure,
        ) = get_dihedral_energy_and_forces(
            dihedral_forces,
            positions,
            config.box_size,
            *topol.bonds_4,
            ff_family=config.ff_family,
        )
        torsional_energy = dihedral_energy
    if topol.impropers:
        improper_energy, improper_forces, _ = get_impropers_energy_and_forces(
            improper_forces, positions, config.box_size, *topol.bonds_impr,
            compute_pressure=False,
        )
    dihedral_energy = torsional_energy + improper_energy

    # Setup dipoles
    # TODO: should apply only if we have proteins
    if topol.dihedrals and config.ff_family != "amber_like":
        dip_fog = jnp.zeros((3, *config.mesh_size))
        # CHECK: should be +2? Check for missing dipole at the start of the sequence?
        n_dip = topol.dihedrals + 1
        dip_charges = jnp.hstack((jnp.full(n_dip, 0.25), jnp.full(n_dip, -0.25)))
        dip_charges = dip_charges.reshape((2 * n_dip, 1))

        # Step 0
        transfer_matrices, dip_positions = get_protein_dipoles(
            positions, config.box_size, *topol.bonds_d
        )
        dip_forces = get_dipole_forces(
            dip_positions, dip_charges, dip_fog, n_dip, config
        )
        reconstr_forces = redistribute_dipole_forces(
            reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
        )

    LJ_energy, LJ_forces = get_LJ_energy_and_forces(
        LJ_forces, pair_params, config
    )
    if use_14_scaling and pair_params_14 is not None and config.lj14_scale != 0.0:
        LJ_14_energy, LJ_14_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params_14, config
        )
        LJ_energy += config.lj14_scale * LJ_14_energy
        LJ_forces += config.lj14_scale * LJ_14_forces


    kinetic_energy = 0.5 * jnp.sum(system.masses * jnp.linalg.norm(velocities, axis=1)**2)

    # Build JIT-compiled velocity-Verlet step function
    if config.barostat:
        verlet_step = _build_verlet_step_fn_npt(
            config, topol, system, use_14_scaling, restr_atoms, excluded_for_elec,
        )
    else:
        verlet_step = _build_verlet_step_fn(
            config, topol, system, use_14_scaling, restr_atoms, excluded_for_elec,
        )
    if topol.dihedrals and config.ff_family != "amber_like":
        verlet_step.dip_charges = dip_charges
        verlet_step.dip_fog = dip_fog
        verlet_step.n_dip = n_dip

    out_dataset = OutDataset(
        args.destdir,
        args.output,
        double_out=False,
        append=_is_append,
    )

    # Gate the H5 charge-energy writer on whether the system carries charges.
    # OutDataset only creates the `field_q_energy_*` datasets when
    # `charges is not None` (file_io.py:700); passing charge_out=True
    # unconditionally would raise AttributeError on no-charge systems.
    _charge_out = system.charges is not None

    # ── Append vs fresh-start bookkeeping ────────────────────────────────
    step_offset = 0
    frame_offset = 0

    if out_dataset._is_append:
        frame_offset, step_offset = reconnect_for_append(
            out_dataset, config,
            system.molecules, topol.bonds_2[0], topol.bonds_2[1],
            velocity_out=True,
            force_out=True,
            charges_present=system.charges is not None,
        )

    from mpi4py import MPI
    write_energy_log = not MPI.Is_initialized() or MPI.COMM_WORLD.Get_rank() == 0
    energy_log_file = None
    if write_energy_log:
        energy_log_path = os.path.join(args.destdir, "energy.log")
        if out_dataset._is_append and os.path.exists(energy_log_path):
            energy_log_file = open(energy_log_path, "a", encoding="utf-8")
        else:
            energy_log_file = open(energy_log_path, "w", encoding="utf-8")
            energy_log_file.write(_ENERGY_LOG_HEADER)

    if not out_dataset._is_append:
        store_static(
            out_dataset,
            system.names,
            onp.asarray(system.types),
            system.indices,
            config,
            topol.bonds_2[0],
            topol.bonds_2[1],
            topol.molecules,
            molecules=system.molecules,
            velocity_out=True,
            force_out=True,
            charges=onp.asarray(system.charges) if system.charges is not None else None,
            resnames=system.resnames,
        )

    _has_excl_main = excluded_for_main is not None
    _masses = system.masses
    _is_npt = bool(config.barostat)

    last_written_frame = frame_offset - 1 if out_dataset._is_append else 0
    if config.n_print is not None and config.n_print > 0 and not out_dataset._is_append:
        step = 0
        frame = 0
        temperature = 2 * kinetic_energy / (config.R * translational_dof(config.n_particles))

        # Compute initial pressure for NPT, zero for NVT/NVE
        if _is_npt:
            _init_kin_p = kinetic_pressure(velocities, _masses, config.volume)
            # virial_pressure is not available before the first step;
            # approximate with kinetic contribution only.
            pressure = onp.asarray(_init_kin_p)
            pressure_scalar = float(jnp.mean(pressure) * config.p_conv)
        else:
            pressure = 0.0
            pressure_scalar = 0.0

        store_data(
            out_dataset,
            step,
            frame,
            system.indices,
            onp.asarray(positions),
            onp.asarray(velocities),
            onp.asarray(LJ_forces),
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            dihedral_energy,
            LJ_energy,
            elec_energy,
            # elec_ener_real,
            # elec_ener_fourrier,
            config,
            velocity_out=True,
            force_out=True,
            charge_out=_charge_out,
            dump_per_particle=False,
        )

        if energy_log_file is not None:
            sim_time_fs = step * _safe_float(config.outer_ts) * 1000.0
            potential_energy = (
                _safe_float(LJ_energy)
                + _safe_float(elec_energy)
                + _safe_float(bond_energy)
                + _safe_float(angle_energy)
                + _safe_float(torsional_energy)
                + _safe_float(improper_energy)
            )
            total_energy = potential_energy + _safe_float(kinetic_energy)
            energy_log_file.write(
                _format_energy_log_line(
                    step,
                    sim_time_fs,
                    temperature,
                    total_energy,
                    potential_energy,
                    kinetic_energy,
                    LJ_energy,
                    elec_energy,
                    bond_energy,
                    angle_energy,
                    torsional_energy,
                    improper_energy,
                    pressure=pressure_scalar,
                )
                + "\n"
            )

    loop_start_time = datetime.datetime.now()

    # ── lax.scan body: nlist update + Verlet step + COM cancel ───────────

    if _is_npt:
        def _scan_body(carry, step):
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow, box_state, _prev_pressure) = carry

            box_sz = box_state.box_size

            # Neighbor list update (uses dynamic box)
            if nlist_method == "jaxmd":
                neigh_i, neigh_j, nlist_overflow, nlist_state = maybe_update_jaxmd_nlist(
                    step, ns_nlist, positions, nlist_state, n_atoms,
                    neigh_i, neigh_j,
                    excluded_nlist_i, excluded_nlist_j,
                    _has_excl_main,
                    box_sz,
                )
            else:
                neigh_i, neigh_j, nlist_overflow, nlist_state = maybe_rebuild_verlet_list_jax(
                    step, ns_nlist, positions, nlist_state,
                    box_sz, rv, skin,
                    neigh_i, neigh_j,
                    excluded_nlist_i, excluded_nlist_j,
                    _has_excl_main,
                )
            any_overflow = any_overflow | nlist_overflow

            # NPT velocity Verlet step
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             virial_pressure) = verlet_step(
                positions, velocities,
                LJ_forces, elec_forces, reconstr_forces,
                bond_forces, angle_forces, dihedral_forces, improper_forces,
                LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                neigh_i, neigh_j, key,
                pair_params_14, excl_pair_params,
                box_state,
            )

            # Compute total instantaneous pressure
            volume = box_state.volume
            kin_pressure = kinetic_pressure(velocities, _masses, volume)
            total_pressure = kin_pressure + virial_pressure / volume

            # Apply barostat
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
                velocities = lax.cond(
                    should_cancel,
                    lambda v: cancel_com_momentum(v, _masses),
                    lambda v: v,
                    velocities,
                )

            new_carry = (positions, velocities,
                         LJ_forces, elec_forces, reconstr_forces,
                         bond_forces, angle_forces, dihedral_forces, improper_forces,
                         LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                         key, pair_params_14, excl_pair_params,
                         neigh_i, neigh_j, nlist_state, any_overflow, box_state, total_pressure)
            return new_carry, None
    else:
        def _scan_body(carry, step):
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow) = carry

            # Neighbor list update
            if nlist_method == "jaxmd":
                neigh_i, neigh_j, nlist_overflow, nlist_state = maybe_update_jaxmd_nlist(
                    step, ns_nlist, positions, nlist_state, n_atoms,
                    neigh_i, neigh_j,
                    excluded_nlist_i, excluded_nlist_j,
                    _has_excl_main,
                    config.box_size,
                )
            else:
                neigh_i, neigh_j, nlist_overflow, nlist_state = maybe_rebuild_verlet_list_jax(
                    step, ns_nlist, positions, nlist_state,
                    config.box_size, rv, skin,
                    neigh_i, neigh_j,
                    excluded_nlist_i, excluded_nlist_j,
                    _has_excl_main,
                )
            any_overflow = any_overflow | nlist_overflow

            # JIT-compiled velocity Verlet step
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params) = verlet_step(
                positions, velocities,
                LJ_forces, elec_forces, reconstr_forces,
                bond_forces, angle_forces, dihedral_forces, improper_forces,
                LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                neigh_i, neigh_j, key,
                pair_params_14, excl_pair_params,
            )

            # Cancel COM momentum
            if config.cancel_com_momentum:
                should_cancel = jnp.mod(step, config.cancel_com_momentum) == 0
                velocities = lax.cond(
                    should_cancel,
                    lambda v: cancel_com_momentum(v, _masses),
                    lambda v: v,
                    velocities,
                )

            new_carry = (positions, velocities,
                         LJ_forces, elec_forces, reconstr_forces,
                         bond_forces, angle_forces, dihedral_forces, improper_forces,
                         LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                         key, pair_params_14, excl_pair_params,
                         neigh_i, neigh_j, nlist_state, any_overflow)
            return new_carry, None

    # ── Determine nlist_state and chunk structure ────────────────────────
    if nlist_method == "jaxmd":
        nlist_state = nbrs
    else:
        nlist_state = ref_positions

    carry = (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, jnp.array(False))

    if _is_npt:
        if _is_append and _restart_box_size is not None:
            # Restart: scale box_state to match the last recorded box volume
            _box_alpha = _restart_box_size / config.box_size
            box_state = BoxState.from_config(config).rescale(
                _box_alpha, config.empty_mesh.shape, config.fft_shape
            )
        else:
            box_state = BoxState.from_config(config)
        carry = carry + (box_state, jnp.zeros(3))

    if config.n_print is not None and config.n_print > 0:
        chunk_size = config.n_print
        n_chunks = config.n_steps // chunk_size
        remainder = config.n_steps % chunk_size
    else:
        chunk_size = config.n_steps
        n_chunks = 1
        remainder = 0

    ###################
    # # # MD LOOP # # #
    ###################
    Logger.rank0.info(
        f"Starting MD loop: {config.n_steps} steps in {n_chunks} chunk(s) "
        f"of {chunk_size} steps (first chunk triggers JIT compilation)."
    )
    for chunk in range(n_chunks):
        step_indices = jnp.arange(
            step_offset + chunk * chunk_size + 1,
            step_offset + (chunk + 1) * chunk_size + 1,
        )
        carry, _ = lax.scan(_scan_body, carry, step_indices)

        # Unpack carry for I/O and overflow check
        if _is_npt:
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow, box_state, inst_pressure) = carry
        else:
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow) = carry

        step = step_offset + (chunk + 1) * chunk_size  # last step of this chunk

        # jaxmd overflow re-allocation (Python-level, between scan chunks)
        if nlist_method == "jaxmd" and jax.device_get(jnp.bool_(any_overflow)):
            Logger.rank0.warning(
                "jax-md neighbor list overflow — re-allocating with larger capacity."
            )
            current_box = box_state.box_size if _is_npt else config.box_size
            pos_frac = positions / current_box
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="scatter inputs have incompatible types",
                    category=FutureWarning,
                )
                nlist_state = neighbor_fn.allocate(pos_frac, box=current_box)
            neigh_i = jnp.where(
                nlist_state.idx[0] < n_atoms, nlist_state.idx[0], jnp.int32(-1)
            ).astype(jnp.int32)
            neigh_j = jnp.where(
                nlist_state.idx[1] < n_atoms, nlist_state.idx[1], jnp.int32(-1)
            ).astype(jnp.int32)
            if _has_excl_main:
                neigh_i, neigh_j = exclude_bonded_neighbors(
                    neigh_i, neigh_j,
                    excluded_for_main[0], excluded_for_main[1],
                )
            any_overflow = jnp.array(False)
            # Repack carry with updated nlist
            carry = (positions, velocities,
                     LJ_forces, elec_forces, reconstr_forces,
                     bond_forces, angle_forces, dihedral_forces, improper_forces,
                     LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
                     key, pair_params_14, excl_pair_params,
                     neigh_i, neigh_j, nlist_state, any_overflow)
            if _is_npt:
                carry = carry + (box_state, inst_pressure)

        # I/O at end of each chunk (= every n_print steps)
        if config.n_print > 0:
            frame = step // config.n_print
            last_written_frame = frame

            kinetic_energy = 0.5 * jnp.sum(
                _masses * jnp.linalg.norm(velocities, axis=1) ** 2
            )
            temperature = 2 * kinetic_energy / (config.R * translational_dof(config.n_particles))

            # Use instantaneous pressure from scan carry for NPT
            if _is_npt:
                pressure = onp.asarray(inst_pressure * config.p_conv)
                pressure_scalar = float(jnp.mean(inst_pressure * config.p_conv))
            else:
                pressure = 0.0
                pressure_scalar = 0.0

            # For NPT, use the dynamic box from box_state for I/O
            config_io = config
            if _is_npt:
                config_io = config.replace(
                    box_size=box_state.box_size,
                    volume=box_state.volume,
                )

            store_data(
                out_dataset,
                step,
                frame,
                system.indices,
                onp.asarray(positions),
                onp.asarray(velocities),
                onp.asarray(LJ_forces),
                temperature,
                pressure,
                kinetic_energy,
                bond_energy,
                angle_energy,
                dihedral_energy,
                LJ_energy,
                elec_energy,
                config_io,
                velocity_out=True,
                force_out=True,
                charge_out=_charge_out,
                dump_per_particle=False,
            )

            tot_t = datetime.datetime.now() - loop_start_time
            seconds_per_day = 24 * 60 * 60
            seconds_elapsed = (
                tot_t.days * seconds_per_day
                + tot_t.seconds
                + 1e-6 * tot_t.microseconds
            )
            seconds_elapsed = max(seconds_elapsed, 1e-12)

            ns_sim = (step - step_offset + 1) * _safe_float(config.outer_ts) / 1000.0
            ns_sim = max(ns_sim, 1e-12)

            days_elapsed = seconds_elapsed / seconds_per_day
            hours_elapsed = seconds_elapsed / 3600.0

            ns_per_day = ns_sim / days_elapsed
            hours_per_ns = hours_elapsed / ns_sim
            steps_per_s = (step - step_offset + 1) / seconds_elapsed

            perf_line, thermo_line = _format_step_log(
                step,
                step_offset + config.n_steps,
                tot_t,
                ns_per_day,
                hours_per_ns,
                steps_per_s,
                temperature,
                pressure_scalar,
                kinetic_energy,
                bond_energy,
                angle_energy,
                dihedral_energy,
                LJ_energy,
                elec_energy,
            )
            Logger.rank0.log(logging.INFO, perf_line)
            Logger.rank0.log(logging.INFO, thermo_line)

            if energy_log_file is not None:
                # dihedral_energy (from carry) = torsional + improper.
                # Re-derive the split cheaply.
                log_improper_energy = 0.0
                if topol.impropers:
                    log_improper_energy, _, _ = get_impropers_energy_and_forces(
                        jnp.zeros_like(positions),
                        positions,
                        config.box_size,
                        *topol.bonds_impr,
                        compute_pressure=False,
                    )
                log_torsional_energy = dihedral_energy - log_improper_energy

                sim_time_fs = step * _safe_float(config.outer_ts) * 1000.0
                potential_energy = (
                    _safe_float(LJ_energy)
                    + _safe_float(elec_energy)
                    + _safe_float(bond_energy)
                    + _safe_float(angle_energy)
                    + _safe_float(log_torsional_energy)
                    + _safe_float(log_improper_energy)
                )
                total_energy = potential_energy + _safe_float(kinetic_energy)
                energy_log_file.write(
                    _format_energy_log_line(
                        step,
                        sim_time_fs,
                        temperature,
                        total_energy,
                        potential_energy,
                        kinetic_energy,
                        LJ_energy,
                        elec_energy,
                        bond_energy,
                        angle_energy,
                        log_torsional_energy,
                        log_improper_energy,
                        pressure=pressure_scalar,
                    )
                    + "\n"
                )

            if config.n_flush and config.n_flush > 0 and onp.mod(step, config.n_print * config.n_flush) == 0:
                out_dataset.flush()
                if energy_log_file is not None:
                    energy_log_file.flush()

    # Run remaining steps (if n_steps not divisible by chunk_size)
    if remainder > 0:
        step_indices = jnp.arange(
            step_offset + n_chunks * chunk_size + 1,
            step_offset + config.n_steps + 1,
        )
        carry, _ = lax.scan(_scan_body, carry, step_indices)
        if _is_npt:
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow, box_state, inst_pressure) = carry
        else:
            (positions, velocities,
             LJ_forces, elec_forces, reconstr_forces,
             bond_forces, angle_forces, dihedral_forces, improper_forces,
             LJ_energy, elec_energy, bond_energy, angle_energy, dihedral_energy,
             key, pair_params_14, excl_pair_params,
             neigh_i, neigh_j, nlist_state, any_overflow) = carry
        step = step_offset + config.n_steps

    # Post-loop overflow safety check (single host sync).
    if jax.device_get(any_overflow):
        Logger.rank0.log(
            logging.WARNING,
            "Neighbor list overflow detected during simulation. "
            "Increase initial neighbor list capacity (rv or buffer) "
            "to avoid silently dropping pairs.",
        )

    # Update config with final box for downstream code (NPT)
    if _is_npt:
        config = config.replace(
            box_size=box_state.box_size,
            volume=box_state.volume,
            volume_per_cell=box_state.volume_per_cell,
            k_vector=box_state.k_vector,
            k_meshgrid=box_state.k_meshgrid,
        )

    # End simulation
    end_time = datetime.datetime.now()
    sim_time = end_time - start_time
    setup_time = loop_start_time - start_time
    loop_time = end_time - loop_start_time
    Logger.rank0.log(
        logging.INFO,
        (
            f"Elapsed time: {format_timedelta(sim_time)}   "
            f"Setup time: {format_timedelta(setup_time)}   "
            f"MD loop time: {format_timedelta(loop_time)}"
        ),
    )

    if config.n_print is not None and config.n_print > 0 and jnp.mod(config.n_steps - 1, config.n_print) != 0:
        # Recompute pair_params from final positions for the trailing frame
        if config.coulombtype and system.charges is not None:
            pair_params = apply_nlist_elec(
                neigh_i, neigh_j, positions, system.charges,
                config.box_size, config.sgm_table, config.epsl_table, system.types,
            )
        else:
            pair_params = apply_nlist(
                neigh_i, neigh_j, positions,
                config.box_size, config.sgm_table, config.epsl_table, system.types,
            )

        LJ_energy, LJ_forces = get_LJ_energy_and_forces(
            LJ_forces, pair_params, config 
        )
        if use_14_scaling and pair_params_14 is not None and config.lj14_scale != 0.0:
            LJ_14_energy, LJ_14_forces = get_LJ_energy_and_forces(
                LJ_forces, pair_params_14, config
            )
            LJ_energy += config.lj14_scale * LJ_14_energy
            LJ_forces += config.lj14_scale * LJ_14_forces

        if system.charges is not None:
            if config.coulombtype == 1:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, system.charges, config, pair_params, excl_pair_params
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


        kinetic_energy = 0.5 * jnp.sum(system.masses * jnp.linalg.norm(velocities, axis=1)**2)
        # kinetic_energy = 0.5 * config.mass * jnp.sum(velocities * velocities)

        frame = (step + 1) // config.n_print
        last_written_frame = frame
        temperature = 2 * kinetic_energy / (config.R * translational_dof(config.n_particles))

        # Use instantaneous pressure from scan carry for NPT
        if _is_npt:
            pressure = onp.asarray(inst_pressure * config.p_conv)
            pressure_scalar = float(jnp.mean(inst_pressure * config.p_conv))
        else:
            pressure = 0.0
            pressure_scalar = 0.0

        config_io = config
        if _is_npt:
            config_io = config.replace(
                box_size=box_state.box_size,
                volume=box_state.volume,
            )

        store_data(
            out_dataset,
            step,
            frame,
            system.indices,
            onp.asarray(positions),
            onp.asarray(velocities),
            onp.asarray(LJ_forces),
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            dihedral_energy,
            LJ_energy,
            elec_energy,
            # elec_ener_real,
            # elec_ener_fourrier,
            config_io,
            velocity_out=True,
            force_out=True,
            charge_out=_charge_out,
            dump_per_particle=False,
        )

        if energy_log_file is not None:
            # Re-derive torsional/improper split from dihedral_energy
            # (the carry only tracks the combined value).
            trail_improper_energy = 0.0
            if topol.impropers:
                trail_improper_energy, _, _ = get_impropers_energy_and_forces(
                    jnp.zeros_like(positions),
                    positions,
                    config.box_size,
                    *topol.bonds_impr,
                    compute_pressure=False,
                )
            trail_torsional_energy = dihedral_energy - trail_improper_energy

            sim_time_fs = step * _safe_float(config.outer_ts) * 1000.0
            potential_energy = (
                _safe_float(LJ_energy)
                + _safe_float(elec_energy)
                + _safe_float(bond_energy)
                + _safe_float(angle_energy)
                + _safe_float(trail_torsional_energy)
                + _safe_float(trail_improper_energy)
            )
            total_energy = potential_energy + _safe_float(kinetic_energy)
            energy_log_file.write(
                _format_energy_log_line(
                    step,
                    sim_time_fs,
                    temperature,
                    total_energy,
                    potential_energy,
                    kinetic_energy,
                    LJ_energy,
                    elec_energy,
                    bond_energy,
                    angle_energy,
                    trail_torsional_energy,
                    trail_improper_energy,
                    pressure=pressure_scalar,
                )
                + "\n"
            )
            energy_log_file.flush()
    if energy_log_file is not None:
        energy_log_file.close()
    if config.n_print > 0:
        truncate_time_series(out_dataset, int(last_written_frame) + 1)
    out_dataset.close_file()
