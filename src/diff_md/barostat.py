import jax.numpy as jnp
from jax import jit, random


def scaling_factor(target, pressure, config):
    return (
        -config.n_b
        * config.beta
        * config.outer_ts
        / config.tau_p
        * (target - pressure * config.p_conv)
    )


def noise(factor, gauss, config):
    return (
        jnp.sqrt(
            factor
            * config.n_b
            * config.beta
            * config.outer_ts
            * config.R
            * config.target_temperature
            * config.p_conv
            / (config.volume * config.tau_p)
        )
        * gauss
    )


@jit
def berendsen(pressure, positions, config):
    if config.barostat_type == 1:
        # isotropic
        pressure = jnp.mean(pressure)
    elif config.barostat_type == 2:
        # Semi-isotropic
        pressure = pressure.at[0:2].set((pressure[0] + pressure[1]) / 2)
    # else:
    #     # Surface tension
    #     return

    alpha = jnp.cbrt(1.0 + scaling_factor(config.target_pressure, pressure, config))
    positions *= alpha
    return positions, config.update_box(alpha)


@jit
def c_rescale(pressure, positions, velocities, config, key):
    key, subkey = random.split(key)
    if config.barostat_type == 1:
        # Isotropic
        gauss = random.normal(subkey)
        pressure = jnp.mean(pressure)
        alpha = jnp.exp(
            (
                scaling_factor(config.target_pressure, pressure, config)
                + noise(2.0, gauss, config)
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
                + config.target_pressure[0] / config.box_size[2]
            )
            p_scale = scaling_factor(config.target_pressure[-1], pressure, config) / 3

        gauss_xy, gauss_z = random.normal(subkey, shape=(2,))
        alpha_xy = jnp.exp(p_scale[0] + noise(4.0 / 3.0, gauss_xy, config) / 2.0)
        alpha_z = jnp.exp(p_scale[-1] + noise(2.0 / 3.0, gauss_z, config))
        alpha = jnp.array((alpha_xy, alpha_xy, alpha_z))

    positions *= alpha
    velocities /= alpha
    return positions, velocities, config.update_box(alpha), key
