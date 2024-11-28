import jax.numpy as jnp
import jax


@jax.jit
def cancel_com_momentum(velocities, config):
    com_velocity = jnp.sum(velocities, axis=0)
    velocities -= com_velocity / config.n_particles
    return velocities


def generate_initial_velocities(velocities, key, config):
    kT_start = config.R * config.start_temperature
    n_particles_ = velocities.shape[0]

    # μ = 0
    sigma = kT_start / config.mass
    velocities = sigma * jax.random.normal(key, shape=(n_particles_, 3))

    com_velocity = jnp.sum(velocities, axis=0)
    velocities = velocities - com_velocity / config.n_particles
    kinetic_energy = 0.5 * config.mass * jnp.sum(velocities * velocities)

    factor = jnp.sqrt(1.5 * config.n_particles * kT_start / kinetic_energy)
    return velocities * factor


@jax.jit
def csvr_thermostat(velocity, key, config):
    """Canonical sampling through velocity rescaling thermostat

    References
    ----------
    G. Bussi, D. Donadio, and M. Parrinello, J. Chem. Phys. 126, 014101 (2007).
    G. Bussi and M. Parrinello, Comput. Phys. Commun. 179, 26-29, (2008).
    """
    new_velocity = jnp.zeros_like(velocity)
    for group_n_particles, group in config.thermostat_coupling_groups:
        key, subkey_1, subkey_2 = jax.random.split(key, 3)
        group_velocity = velocity * group
        com_velocity = jnp.sum(group_velocity, axis=0) / group_n_particles * group
        group_velocity -= com_velocity

        dof = 3 * group_n_particles
        kinetic_energy = 0.5 * config.mass * jnp.sum(group_velocity**2)
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
        new_velocity += group_velocity + com_velocity
    return new_velocity, key
