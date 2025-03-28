import os
import random
from typing import Any, Tuple

import jax.random
import mpi4jax
import numpy as onp
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from jax import value_and_grad
from mpi4py import MPI

from .file_io import OutDataset, save_params, store_static, write_full_trajectory
from .input_parser import System
from .logger import Logger
from .losses import get_chi, get_LJ_param
from .nn_options import get_training_parameters
from .simulate import simulator

# NOTE: double precision helps mitigate gradient explosion, but it's expensive
# jax_conf.update("jax_enable_x64", True)


def save_state(dirname: str, data: dict[str, Any]) -> None:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(data)
    # TODO: create new directory if 'dirname' exists, so we don't overwrite with force
    orbax_checkpointer.save(dirname, data, save_args=save_args, force=True)


def load_state(dirname: str, target: dict[str, Any]) -> Tuple[Any, ...]:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(dirname, item=target)
    return (
        restored["epoch"],
        restored["params"],
        restored["state"],
    )


def main(args, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize PRNG keys
    key = None
    for i in range(size):
        if rank == i:
            key = jax.random.PRNGKey(args.seed + i)

    def step(params, opt_state, key):
        (loss_value, (output, trj, key, config)), grads = value_and_grad(
            nn_options.loss, has_aux=True
        )(
            params,
            system,
            key,
            start_temperature,
            comm,
            **nn_options.loss_args,
            **nn_options.system_args[system.name],
        )

        # Save stuff for plotting
        if rank == 0:
            for k, v in output.items():
                if k == "density":
                    onp.save(f"{destdir}/{system.name}_density.npy", v)
                else:
                    Logger.rank0.debug(f"{system.name} {k} = {v}")

        # Get gradients from all ranks
        # Gradients are already normalized by autodiff
        total_chi_grad, _ = mpi4jax.allreduce(grads.chi, op=MPI.SUM, comm=comm)
        grads = grads.replace(chi=total_chi_grad)

        # Update parameters
        updates, opt_state = nn_options.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        Logger.rank0.debug(
            f"System {system.name}, current_loss: {loss_value}\n{50*'*'}\n"
            f"Gradients\n{grads}{50*'-'}\n"
            f"Updated parameters\n{params}{50*'-'}",
        )

        return params, opt_state, loss_value, trj, key, config

    nn_options, params, toml_input = get_training_parameters(args.model)

    n_systems = len(nn_options.systems)
    loop_range = list(range(n_systems))

    dataset = []
    for dir in nn_options.systems:
        dataset.append(System.constructor(args, nn_options.name_to_type, dir, params))

    # Save starting configurations for equilibration
    start_pos, start_vel, start_config = [], [], []
    if nn_options.equilibration:
        for system in dataset:
            start_pos.append(system.positions)
            start_vel.append(system.velocities)
            start_config.append(system.config)
    init_temps = [system.config.start_temperature for system in dataset]

    # Debug single simulation
    if args.debug:
        Logger.rank0.debug("Executing in debug mode...")
        for i, system in enumerate(dataset):
            Logger.rank0.debug(f"Simulating system: {system.name}")

            loss_value, (output, trj, _, config) = nn_options.loss(
                # fmt: off
                params, system, key, init_temps[i], comm,
                **nn_options.loss_args, **nn_options.system_args[system.name]
            )
            Logger.rank0.debug(f"Loss = {loss_value}")

            for k, v in output.items():
                if k == "density":
                    Logger.rank0.debug("Saving density to 'debug_density.npy'")
                    onp.save(f"{args.destdir}/{system.name}_debug_density.npy", v)
                else:
                    Logger.rank0.debug(f"{k} = {v}")

            # Write debug simulation to h5md
            out_dataset = OutDataset(
                f"{args.destdir}/{system.name}",
                "debug",
                double_out=False,
            )
            store_static(
                # fmt: off
                out_dataset, system.names, system.types, system.indices, config, system.topol.bonds_2[0], system.topol.bonds_2[1],
                system.topol.molecules, molecules=system.molecules, velocity_out=False, force_out=False, charges=False,
            )
            write_full_trajectory(out_dataset, trj, system.indices, config)
            out_dataset.file.close()
        exit()

    start_epoch = 0
    opt_state = nn_options.optimizer.init(params)
    out_loss = f"{args.destdir}/loss.dat"

    if args.restart:
        start_epoch, params, opt_state = load_state(
            args.restart,
            {
                "epoch": 0,
                "params": params,
                "state": opt_state,
            },
        )

    Logger.rank0.info(f"\n\tInitial parameters:\n" f"\t\t{params}")

    # Run training loop
    for epoch in range(start_epoch, start_epoch + nn_options.n_epochs):
        epoch_loss = 0
        
        destdir = f"{args.destdir}/step_{epoch}"
        params_file = f"{destdir}/training.toml"
        if rank == 0:
            os.makedirs(destdir, exist_ok=True)
            save_params(params_file, toml_input, params)

        Logger.rank0.debug(f"Starting epoch {epoch}\n{50*'='}\n")

        if nn_options.shuffle and n_systems > 1:
            if rank == 0:
                random.shuffle(loop_range)
            loop_range = comm.bcast(loop_range, root=0)

        for i in loop_range:
            system = dataset[i]
            start_temperature = init_temps[i]

            if nn_options.equilibration:
                # Restarts from initial positions
                # chi, _, type_mask = get_chi(params, system.types, system.config)
                epsl, _ = get_LJ_param(params, system.config)
                trj, key, config = simulator(
                    # fmt: off
                    params, start_pos[i], start_vel[i], system.types, system.charges,
                    epsl, key, system.topol, start_config[i], start_temperature, nn_options.equilibration
                )
                system.positions, system.velocities = (
                    trj["positions"][-1],
                    trj["velocities"][-1],
                )
                system.config = config
                start_temperature = False

            if nn_options.teacher_forcing:
                # Teacher forcing == continuous simulation, restarting from the last step
                params, opt_state, loss_value, trj, key, config = step(
                    params, opt_state, key
                )

                if nn_options.equilibration:
                    start_pos[i], start_vel[i] = (
                        trj["positions"][-1],
                        trj["velocities"][-1],
                    )
                    start_config[i] = config
                else:
                    system.positions, system.velocities = (
                        trj["positions"][-1],
                        trj["velocities"][-1],
                    )
                    system.config = config
                init_temps[i] = False
            else:
                params, opt_state, loss_value, trj, _, _ = step(params, opt_state, key)
                epoch_loss += loss_value

        Logger.rank0.info(f"Epoch {epoch}, mean loss = {epoch_loss / n_systems}\n\n")
        if rank == 0:
            # Write intermediate loss values to file
            with open(out_loss, "a") as outfile:
                print(f"{epoch}\t{epoch_loss / n_systems}", file=outfile)

            # save optimizer state and parameters for restart
            save_state(
                f"{destdir}/cpt",
                {
                    "epoch": epoch + 1,
                    "params": params,
                    "state": opt_state,
                },
            )

    if rank == 0:
        # save toml with final parameters
        save_params(f"{args.destdir}/final.toml", toml_input, params)
