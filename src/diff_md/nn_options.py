import copy
import dataclasses
import os
from typing import Any, Callable, Tuple

import jax.numpy as jnp
import numpy as np
import optax

from . import losses
from .config import get_type_to_LJ, read_toml
from .logger import Logger
from .models import GeneralModel


@dataclasses.dataclass
class NNoptions:
    optimizer: optax.GradientTransformation
    n_epochs: int
    name_to_type: dict[str, int]
    systems: list[str]
    loss: Callable
    loss_args: dict
    system_args: dict
    teacher_forcing: bool = False
    chain: bool = False
    equilibration: int = 0
    shuffle: bool = False

@dataclasses.dataclass
class System_options:


def str_from_dict(input: dict, output: str = "", depth: int = 1) -> str:
    """Recursively parse dict of dicts"""
    for k, v in input.items():
        if isinstance(v, dict):
            output += depth * "\t" + f"{k}:\n"
            output = str_from_dict(v, output, depth + 1)
        else:
            output += depth * "\t" + f"{k}: {v}\n"
    return output


def check_missing_section(section, toml_config, file_path):
    if section not in toml_config:
        Logger.rank0.error(f"Missing required [{section}] section in '{file_path}'.")
        exit()


def toml_key_error_exit(key, section, file_path):
    if key not in section:
        Logger.rank0.error(
            f"Missing required {key} inside [{section}] section in '{file_path}'."
        )
        exit()


def get_training_parameters(
    file_path: str,
) -> Tuple[NNoptions, GeneralModel, dict[str, Any]]:
    """Parse training options toml file"""
    toml_config = read_toml(file_path)

    # save copy for output parameters
    toml_copy = copy.deepcopy(toml_config)
    check_missing_section("nn", toml_config, file_path)

    name_to_type = {}
    model_dict = toml_config["nn"].pop("model")
    if "LJ_param" in model_dict:
        names = []
        for row in model_dict["LJ_param"]:
            names += row[:2]

        names = np.array(names)
        _, name_idx = np.unique(names, return_index=True)
        unique_names = names[np.sort(name_idx)]

        n_types = len(unique_names)
        name_to_type = {name: type for type, name in enumerate(unique_names)}

        model_dict["n_types"] = n_types
        model_dict["type_to_LJ"] = (ttlj := get_type_to_LJ(n_types))

        start_value = {}
        epsl_constraints = {}
        epsl = np.zeros((n_types, n_types))
        for pair in model_dict["LJ_param"]:
            type_0, type_1 = sorted((name_to_type[pair[0]], name_to_type[pair[1]]))
            val = pair[3]
            epsl[type_0, type_1] = val

            if len(pair) > 4:
                pair_idx = ttlj[type_0, type_1]

                # train only these pairs
                if "on" in pair[4:]:
                    start_value[pair_idx] = val

                # constrain these pairs
                if "cs" in pair[4:]:
                    epsl_constraints[pair_idx] = val

        if start_value:
            model_dict["type_to_LJ"] = jnp.array(list(start_value.keys()))
            model_dict["LJ_param"] = jnp.array(list(start_value.values()), dtype=jnp.float_)
        else:
            model_dict["LJ_param"] = jnp.array(epsl[np.triu_indices(n_types)])

        if epsl_constraints:
            model_dict["epsl_constraints"] = epsl_constraints

    if "bonds" in model_dict:
        Logger.rank0.warning(
            f"Bond information was provided in {file_path}, but bond optimization is not implemented yet."
        )
        pass

    args = toml_config.pop("nn")
    ret_str = str_from_dict(args)

    # Optimizer
    if "chain" in args and args["chain"]:
        optim_list = []
        for optim in args["optimizer"]:
            fun = getattr(optax, optim.pop("name"))
            optim_list.append(fun(**optim))
        opt = optax.chain(*optim_list)
    else:
        fun = getattr(optax, args["optimizer"].pop("name"))

        # Check if we have learning rate scheduling
        if isinstance(args["optimizer"]["learning_rate"], dict):
            learning_rate = args["optimizer"].pop("learning_rate")
            scheduler = getattr(optax, learning_rate.pop("schedule"))
            scheduler = scheduler(**learning_rate)
            opt = fun(learning_rate=scheduler, **args["optimizer"])
        else:
            opt = fun(**args["optimizer"])

    batch_size = 1
    if "batch_size" in args and args["batch_size"] > 0:
        batch_size = args.pop("batch_size")

    args["optimizer"] = optax.MultiSteps(opt, every_k_schedule=batch_size)

    # Loss function
    # TODO: check that all required arguments are provided for a given loss name
    loss_function = getattr(losses, args["loss"].pop("name"))
    metric = getattr(losses, args["loss"]["metric"])

    # TODO: Move to system args
    # Read in reference all atom densities
    for dir in args["systems"]:
        filename, ext = os.path.splitext(args["loss"]["target_density"])
        file_path = f"{dir}/{filename}{ext}"
        if ext == ".npy":
            reference = jnp.array(np.load(file_path))
        elif ext == ".xvg":
            reference = jnp.array(
                # transpose so we can work with rows
                np.loadtxt(file_path, comments=["#", "@"]).T
            )
        else:
            Logger.rank0.error(
                f"Target density filename '{file_path}' has the wrong extension."
                f"Valid extensions are '.npy' and '.xvg'."
            )
            exit()
        args["system_args"][dir]["z_range"] = reference[0]
        args["system_args"][dir]["target_density"] = reference[1:]

        # Expand with other system specific options that might need to be converted to type (ie RDF selections)
        if "com_type" in args["system_args"][dir]:
            args["system_args"][dir]["com_type"] = name_to_type[
                args["system_args"][dir]["com_type"]
            ]
    del args["loss"]["target_density"]

    # TODO: needs improvements, we do not parse it correctly when using multiple systems
    # therefore easily leads to errors in case a list is passed
    if density_weight := args["loss"].get("density_weight"):
        if isinstance(density_weight, (float, int)):
            weight = density_weight
        elif isinstance(density_weight, list):
            if len(density_weight) != len(name_to_type):
                Logger.rank0.error(
                    "Length of 'density_weight' should be the same as the number of bead types."
                )
                exit()
            weight = jnp.array(args["loss"]["density_weight"])
        else:
            Logger.rank0.error(
                f"Invalid 'density_weight' in '{file_path}': '{density_weight}'"
            )
            exit()
        args["loss"]["density_weight"] = weight

    if constr := args["loss"].get("constraint"):
        if constr == "cubic":
            f_constr = getattr(losses, "cubic_constraint")
        elif constr == "harmonic":
            f_constr = getattr(losses, "harmonic_constraint")
        else:
            Logger.rank0.error(
                f"Invalid 'constraint' in '{file_path}': '{constr}'\n",
                "Only 'cubic' and 'harmonic' are accepted.'",
            )
            exit()
        args["loss"]["constraint"] = f_constr

    args["loss"]["metric"] = metric
    args["loss_args"] = args["loss"]
    args["loss"] = loss_function
    args["name_to_type"] = name_to_type

    nn_options = NNoptions(**args)
    model = GeneralModel(**model_dict)

    Logger.rank0.info(f"Training file '{file_path}' parsed successfully.")
    Logger.rank0.info(f"\n\n\tOptimization parameters:\n\t{50 * '-'}\n" f"{ret_str}")
    return nn_options, model, toml_copy

def get_system_options(
    toml_config, 
    systems: list, 
    name_to_type: dict
) -> System_options:
    """Parse training options toml file"""
    args = toml_config.pop("nn")

    for system in systems:
        dir = system.name
        # Get data for density profile
        filename, ext = os.path.splitext(args["loss"]["target_density"])
        file_path = f"{dir}/{filename}{ext}"
        if ext == ".npy":
            reference = jnp.array(np.load(file_path))
        elif ext == ".xvg":
            reference = jnp.array(
                # transpose so we can work with rows
                np.loadtxt(file_path, comments=["#", "@"]).T
            )
        else:
            Logger.rank0.error(
                f"Target density filename '{file_path}' has the wrong extension."
                f"Valid extensions are '.npy' and '.xvg'."
            )
            exit()
        args["system_args"][dir]["z_range"] = reference[0]
        args["system_args"][dir]["target_density"] = reference[1:]

        # Expand with other system specific options that might need to be converted to type (ie RDF selections)
        if "com_type" in args["system_args"][dir]:
            args["system_args"][dir]["com_type"] = name_to_type[
                args["system_args"][dir]["com_type"]
            ]
        
        # Get data for Radius of gyration
        

        # Get data for thickness
    del args["loss"]["target_density"]

    # TODO: needs improvements, we do not parse it correctly when using multiple systems
    # therefore easily leads to errors in case a list is passed
    if density_weight := args["loss"].get("density_weight"):
        if isinstance(density_weight, (float, int)):
            weight = density_weight
        elif isinstance(density_weight, list):
            if len(density_weight) != len(name_to_type):
                Logger.rank0.error(
                    "Length of 'density_weight' should be the same as the number of bead types."
                )
                exit()
            weight = jnp.array(args["loss"]["density_weight"])
        else:
            Logger.rank0.error(
                f"Invalid 'density_weight' in '{file_path}': '{density_weight}'"
            )
            exit()
        args["loss"]["density_weight"] = weight

    if constr := args["loss"].get("constraint"):
        if constr == "cubic":
            f_constr = getattr(losses, "cubic_constraint")
        elif constr == "harmonic":
            f_constr = getattr(losses, "harmonic_constraint")
        else:
            Logger.rank0.error(
                f"Invalid 'constraint' in '{file_path}': '{constr}'\n",
                "Only 'cubic' and 'harmonic' are accepted.'",
            )
            exit()
        args["loss"]["constraint"] = f_constr

    system_options = System_options(**args)
    return system_options