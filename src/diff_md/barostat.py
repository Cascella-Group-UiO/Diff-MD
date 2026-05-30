import jax.numpy as jnp
from jax import random


def kinetic_pressure(velocities, masses, volume):
    """Kinetic contribution to the pressure tensor (diagonal).

    Returns shape-(3,) array [Pxx_kin, Pyy_kin, Pzz_kin].
    """
    return jnp.sum(masses[:, None] * velocities ** 2, axis=0) / volume


def scaling_factor(target, pressure, config):
    return (
        -config.n_b
        * config.beta
        * config.outer_ts
        / config.tau_p
        * (target - pressure * config.p_conv)
    )


def noise(factor, gauss, config, volume):
    return (
        jnp.sqrt(
            factor
            * config.n_b
            * config.beta
            * config.outer_ts
            * config.R
            * config.target_temperature
            * config.p_conv
            / (volume * config.tau_p)
        )
        * gauss
    )


def berendsen(pressure, positions, box_state, config):
    """Berendsen barostat — rescales positions and box.

    Parameters
    ----------
    pressure : (3,) array — instantaneous diagonal pressure tensor
    positions : (N, 3) array
    box_state : BoxState (dynamic box)
    config : Config (static parameters)

    Returns (positions, new_box_state).
    """
    if config.barostat_type == 1:
        # isotropic
        pressure = jnp.mean(pressure) * jnp.ones(3)
    elif config.barostat_type == 2:
        # Semi-isotropic
        pressure = pressure.at[0:2].set((pressure[0] + pressure[1]) / 2)

    alpha = jnp.cbrt(1.0 + scaling_factor(config.target_pressure, pressure, config))

    # Keep the barostat path side-effect free: NPT optimization
    # differentiates through this function, and host callbacks here break
    # JAX transforms used by reverse-mode training.
    positions = positions * alpha

    mesh_shape = config.empty_mesh.shape
    fft_shape = config.fft_shape
    new_box_state = box_state.rescale(alpha, mesh_shape, fft_shape)
    return positions, new_box_state


def c_rescale(pressure, positions, velocities, box_state, config, key):
    """Stochastic cell rescaling (SCR) barostat.

    Parameters
    ----------
    pressure : (3,) array — instantaneous diagonal pressure tensor
    positions : (N, 3) array
    velocities : (N, 3) array
    box_state : BoxState (dynamic box)
    config : Config (static parameters)
    key : PRNGKey

    Returns (positions, velocities, new_box_state, key).
    """
    key, subkey = random.split(key)
    volume = box_state.volume

    if config.barostat_type == 1:
        # Isotropic
        gauss = random.normal(subkey)
        pressure = jnp.mean(pressure) * jnp.ones(3)
        alpha = jnp.exp(
            (
                scaling_factor(config.target_pressure, pressure, config)
                + noise(2.0, gauss, config, volume)
            )
            / 3
        )
    else:
        if config.barostat_type == 2:
            # Semi-isotropic
            pressure = pressure.at[0:2].set((pressure[0] + pressure[1]) / 2)
            p_scale = scaling_factor(config.target_pressure, pressure, config) / 3
        elif config.barostat_type == 3:
            # Surface tension
            pressure = pressure.at[0:2].set(
                (pressure[0] + pressure[1]) / 2
                + config.target_pressure[0] / box_state.box_size[2]
            )
            p_scale = scaling_factor(config.target_pressure[-1], pressure, config) / 3

        gauss_xy, gauss_z = random.normal(subkey, shape=(2,))
        alpha_xy = jnp.exp(p_scale[0] + noise(4.0 / 3.0, gauss_xy, config, volume) / 2.0)
        alpha_z = jnp.exp(p_scale[-1] + noise(2.0 / 3.0, gauss_z, config, volume))
        alpha = jnp.array((alpha_xy, alpha_xy, alpha_z))

    # Keep the barostat path side-effect free: NPT optimization
    # differentiates through this function, and host callbacks here break
    # JAX transforms used by reverse-mode training.
    positions = positions * alpha
    velocities = velocities / alpha

    mesh_shape = config.empty_mesh.shape
    fft_shape = config.fft_shape
    new_box_state = box_state.rescale(alpha, mesh_shape, fft_shape)
    return positions, velocities, new_box_state, key
