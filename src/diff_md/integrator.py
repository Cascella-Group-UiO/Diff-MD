from jax import jit
import jax.numpy as jnp


@jit
def integrate_velocity(velocities, accelerations, time_step):
    return velocities + 0.5 * time_step * accelerations


@jit
def integrate_position(positions, velocities, time_step):
    return positions + time_step * velocities


@jit
def lp_integrate_position(positions, velocities, time_step):
    return positions + time_step * velocities


@jit
def lp_integrate_velocity(velocities, accelerations, time_step):
    return velocities + time_step * accelerations


@jit
def zero_velocities(velocities, restr_atoms):
    mask = jnp.zeros(velocities.shape[0], dtype=bool).at[restr_atoms].set(True)
    velocities = jnp.where(mask[:, None], jnp.array([0.0, 0.0, 0.0]), velocities)
    return velocities


@jit
def zero_forces(forces, restr_atoms):
    mask = jnp.zeros(forces.shape[0], dtype=bool).at[restr_atoms].set(True)
    forces = jnp.where(mask[:, None], jnp.array([0.0, 0.0, 0.0]), forces)
    return forces