from jax import jit


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
