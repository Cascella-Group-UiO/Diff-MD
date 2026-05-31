import jax.numpy as jnp
import jax


def translational_dof(n_particles):
    """Number of thermal translational degrees of freedom for ``n_particles``.

    The global center-of-mass velocity is removed at initialization
    (``generate_initial_velocities``) and kept removed by ``cancel_com_momentum``,
    so for ``N > 1`` only ``3N - 3`` independent momenta carry thermal energy.
    This matches GROMACS' default linear center-of-mass-motion removal and the
    per-group convention already used by ``csvr_thermostat``. Initialization,
    temperature reporting, and the thermostat must all use this same count or
    they disagree (initialization runs hot by ``N/(N-1)`` while a ``3N``
    denominator silently reports the target).

    ``n_particles`` is static (a plain Python int from ``Config``/``ThermostatGroup``
    metadata), so this returns a Python int usable as a constant inside jit.
    """
    return 3 * n_particles - 3 if n_particles > 1 else 3


@jax.jit
def cancel_com_momentum(velocities, masses):
    masses = jnp.reshape(masses, (-1, 1))
    total_mass = jnp.sum(masses)
    com_velocity = jnp.sum(masses * velocities, axis=0) / total_mass
    return velocities - com_velocity


def generate_initial_velocities(velocities, key, config, masses):
    kT_start = config.R * config.start_temperature
    n_particles_ = velocities.shape[0]

    # μ = 0
    masses = jnp.reshape(masses, (-1, 1))
    sigma = jnp.sqrt(kT_start / masses)
    velocities = sigma * jax.random.normal(key, shape=(n_particles_, 3))

    total_mass = jnp.sum(masses)
    com_velocity = jnp.sum(masses * velocities, axis=0) / total_mass
    velocities = velocities - com_velocity

    # kinetic_energy = 0.5 * jnp.sum(masses * jnp.linalg.norm(velocities, axis=1)**2)
    speed2 = jnp.sum(velocities**2, axis=1, keepdims=True)
    kinetic_energy = 0.5 * jnp.sum(masses * speed2)

    # Rescale to the equipartition target for the physical DOF count (3N-3 after
    # the COM removal just applied), not 3N — otherwise the velocities come out
    # hot by N/(N-1). 0.5 * dof * kT == target kinetic energy.
    target_kinetic = 0.5 * translational_dof(config.n_particles) * kT_start
    factor = jnp.sqrt(target_kinetic / kinetic_energy)
    return velocities * factor


@jax.jit
def csvr_thermostat(velocity, key, config, masses):
    """Canonical sampling through velocity rescaling thermostat

    References
    ----------
    G. Bussi, D. Donadio, and M. Parrinello, J. Chem. Phys. 126, 014101 (2007).
    G. Bussi and M. Parrinello, Comput. Phys. Commun. 179, 26-29, (2008).
    """
    new_velocity = velocity
    masses = jnp.reshape(masses, (-1, 1))
    for group in config.thermostat_coupling_groups:
        if group.n_particles == 0:
            continue

        key, subkey_1, subkey_2 = jax.random.split(key, 3)
        group_mask = group.mask.astype(velocity.dtype)
        group_masses = masses * group_mask
        total_group_mass = jnp.sum(group_masses)
        com_velocity = jnp.sum(group_masses * velocity, axis=0) / total_group_mass
        group_velocity = (velocity - com_velocity) * group_mask

        dof = translational_dof(group.n_particles)

        speed2 = jnp.sum(group_velocity**2, axis=1, keepdims=True)
        kinetic_energy = 0.5 * jnp.sum(masses * speed2)
        # Guard against zero/tiny kinetic energy to avoid NaNs in the CSVR ratio terms.
        kinetic_energy = jnp.maximum(kinetic_energy, jnp.array(1e-12, dtype=velocity.dtype))

        target_kinetic = 0.5 * dof * config.R * config.target_temperature
        c = jnp.exp(-(config.outer_ts) / config.tau_t)

        # Draw random numbers
        gauss = jax.random.normal(subkey_1)
        gamma = 2 * jax.random.gamma(subkey_2, 0.5 * (dof - 1))
        # Equal to gamma = jax.random.chisquare(subkey_2, dof - 1)

        alpha2 = (
            # fmt: off
            + (1 - c) * (gamma + gauss * gauss) * target_kinetic / (dof * kinetic_energy)
            + 2 * gauss * jnp.sqrt(c * (1 - c) * target_kinetic / (dof * kinetic_energy))
        )
        alpha = jnp.sqrt(c + alpha2)

        group_velocity *= alpha
        updated_group_velocity = group_velocity + com_velocity * group_mask
        new_velocity = jnp.where(group_mask.astype(bool), updated_group_velocity, new_velocity)
    return new_velocity, key


def apply_thermostat(velocity, key, config, masses):
    """Apply the thermostat selected in the TOML config."""
    if config.thermostat == "no":
        return velocity, key
    if config.thermostat == "v-rescale":
        return csvr_thermostat(velocity, key, config, masses)

    raise ValueError(f"Unknown thermostat '{config.thermostat}'.")
