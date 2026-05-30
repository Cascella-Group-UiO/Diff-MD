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
    teacher_forcing: bool = False
    chain: bool = False
    equilibration: int = 0
    shuffle: bool = False
    n_epochs_longer: int = 0
    n_steps_longer: int = 0
    grad_method: str = "reverse"
    fd_epsilon: float = 1e-4
    double_precision: bool = False
    # XLA cache eviction interval (0 = off, 1 = every epoch, N = every N epochs)
    clear_xla_cache: int = 0
    # Hard clipping bounds for optimizer projection (per parameter type)
    clip_sigma_min: float = 0.05      # nm (0.5 Å minimum)
    clip_sigma_max: float = 2.0       # nm (20 Å maximum)
    clip_epsilon_min: float = 0.001   # kJ/mol
    clip_epsilon_max: float = 100.0   # kJ/mol
    # Gradient norm clipping (applied before optimizer; None = disabled)
    max_grad_norm: float = None

@dataclasses.dataclass
class System_options:
    system_args: dict
    # dir_name -> logical-system name. Populated by get_system_options.
    # Defaults to {dir_name: dir_name} when no `replica_of` field is used.
    replica_map: dict = dataclasses.field(default_factory=dict)


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


# ── Training flag helpers (shared by LJ_param and LJ_type_param) ─────────

_SIGMA_FLAGS = {"on_sigma", "on_sgm"}
_EPSILON_FLAGS = {"on_eps", "on_epsilon"}
_ALL_TRAIN_FLAGS = _SIGMA_FLAGS | _EPSILON_FLAGS
_SIGMA_CS_FLAGS = {"cs_sigma", "cs_sgm"}
_EPSILON_CS_FLAGS = {"cs", "cs_eps", "cs_epsilon"}


def _validate_no_bare_on(flags_dict: dict, section: str):
    """Error out if any entry uses the bare 'on' flag."""
    for _key, flags in flags_dict.items():
        if "on" in flags:
            Logger.rank0.error(
                f"Bare 'on' flag in {section} is not supported. "
                "Use 'on_eps' / 'on_epsilon' to train epsilon and/or "
                "'on_sigma' / 'on_sgm' to train sigma."
            )
            exit()


def _has_explicit_flags(flags_dict: dict) -> bool:
    """True if ANY entry has an explicit on_eps/on_sigma flag."""
    return any(
        flags & _ALL_TRAIN_FLAGS
        for flags in flags_dict.values()
    )


def _should_train(flags: set, param_type: str, has_explicit: bool, default: bool = True) -> bool:
    """Decide whether a parameter should be trained.

    param_type: "sigma" or "epsilon".
    """
    if not has_explicit:
        return default
    target_flags = _SIGMA_FLAGS if param_type == "sigma" else _EPSILON_FLAGS
    return bool(flags & target_flags)


def _has_constraint(flags: set, param_type: str) -> bool:
    """True if the constraint flag is set for this parameter type."""
    target = _SIGMA_CS_FLAGS if param_type == "sigma" else _EPSILON_CS_FLAGS
    return bool(flags & target)


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
        model_dict["type_to_LJ"] = (ttlj := jnp.asarray(get_type_to_LJ(n_types), dtype=jnp.int32))
        model_dict["lj_mode"] = "pair"
        model_dict["lj_name_to_type"] = name_to_type

        # Collect sigma/epsilon values and per-pair flags
        sgm = np.zeros((n_types, n_types))
        epsl = np.zeros((n_types, n_types))
        pair_flags = {}  # {pair_idx: set(flags)}
        for pair in model_dict["LJ_param"]:
            type_0, type_1 = sorted((name_to_type[pair[0]], name_to_type[pair[1]]))
            pair_idx = int(ttlj[type_0, type_1])
            sgm[type_0, type_1] = pair[2]
            epsl[type_0, type_1] = pair[3]
            flags = set()
            if len(pair) > 4:
                flags = {str(f).lower() for f in pair[4:]}
            pair_flags[pair_idx] = flags

        _validate_no_bare_on(pair_flags, "LJ_param")
        has_explicit_train_flags = _has_explicit_flags(pair_flags)

        # First pass: collect trainable sigma pairs
        sigma_train = []  # [(pair_idx, value), ...]
        constraints = {}
        seen = set()
        for pair in model_dict["LJ_param"]:
            type_0, type_1 = sorted((name_to_type[pair[0]], name_to_type[pair[1]]))
            pair_idx = int(ttlj[type_0, type_1])
            if pair_idx in seen:
                continue
            seen.add(pair_idx)
            flags = pair_flags[pair_idx]
            if _should_train(flags, "sigma", has_explicit_train_flags, default=False):
                sigma_train.append((pair_idx, float(pair[2])))
                if _has_constraint(flags, "sigma"):
                    constraints[len(sigma_train) - 1] = float(pair[2])

        n_sigma = len(sigma_train)

        # Second pass: collect trainable epsilon pairs
        epsilon_train = []  # [(pair_idx, value), ...]
        seen = set()
        for pair in model_dict["LJ_param"]:
            type_0, type_1 = sorted((name_to_type[pair[0]], name_to_type[pair[1]]))
            pair_idx = int(ttlj[type_0, type_1])
            if pair_idx in seen:
                continue
            seen.add(pair_idx)
            flags = pair_flags[pair_idx]
            if _should_train(flags, "epsilon", has_explicit_train_flags):
                epsilon_train.append((pair_idx, float(pair[3])))
                k = n_sigma + len(epsilon_train) - 1
                if _has_constraint(flags, "epsilon"):
                    constraints[k] = float(pair[3])

        if not sigma_train and not epsilon_train:
            Logger.rank0.error(
                "No trainable LJ parameters in 'LJ_param'. "
                "Add 'on_eps' or 'on_sigma' flags to at least one pair."
            )
            exit()

        train_values = [v for _, v in sigma_train] + [v for _, v in epsilon_train]
        model_dict["n_sigma_train"] = n_sigma
        model_dict["n_epsilon_train"] = len(epsilon_train)
        model_dict["lj_sigma_idx"] = (
            jnp.array([idx for idx, _ in sigma_train], dtype=jnp.int32)
            if sigma_train else None
        )
        model_dict["lj_epsilon_idx"] = jnp.array(
            [idx for idx, _ in epsilon_train], dtype=jnp.int32
        )
        model_dict["LJ_param"] = jnp.array(train_values)
        if constraints:
            model_dict["epsl_constraints"] = constraints

    elif "LJ_type_param" in model_dict:
        entries = model_dict.pop("LJ_type_param")
        names = np.array([row[0] for row in entries])
        _, name_idx = np.unique(names, return_index=True)
        unique_names = names[np.sort(name_idx)]

        n_types = len(unique_names)
        name_to_type = {name: type for type, name in enumerate(unique_names)}

        model_dict["n_types"] = n_types
        model_dict["type_to_LJ"] = jnp.asarray(get_type_to_LJ(n_types), dtype=jnp.int32)
        model_dict["lj_mode"] = "type"
        model_dict["lj_name_to_type"] = name_to_type

        train_sigma_default = bool(model_dict.pop("train_sigma", False))

        sigma_ref = np.zeros(n_types, dtype=float)
        epsilon_ref = np.zeros(n_types, dtype=float)
        type_flags: dict[int, set[str]] = {}

        for row in entries:
            if len(row) < 3:
                Logger.rank0.error(
                    "Invalid 'LJ_type_param' row. Expected [type, sigma, epsilon, ...flags]. "
                    f"Got: {row}."
                )
                exit()

            t = name_to_type[row[0]]
            sigma_ref[t] = float(row[1])
            epsilon_ref[t] = float(row[2])
            type_flags[t] = {str(flag).lower() for flag in row[3:]}

        _validate_no_bare_on(type_flags, "LJ_type_param")
        has_explicit_train_flags = _has_explicit_flags(type_flags)

        train_values = []
        sigma_idx = []
        epsilon_idx = []
        constraints = {}

        for t in range(n_types):
            flags = type_flags.get(t, set())
            if _should_train(flags, "sigma", has_explicit_train_flags, default=train_sigma_default):
                sigma_idx.append(t)
                train_values.append(sigma_ref[t])
                if _has_constraint(flags, "sigma"):
                    constraints[len(train_values) - 1] = sigma_ref[t]

        for t in range(n_types):
            flags = type_flags.get(t, set())
            if _should_train(flags, "epsilon", has_explicit_train_flags):
                epsilon_idx.append(t)
                train_values.append(epsilon_ref[t])
                if _has_constraint(flags, "epsilon"):
                    constraints[len(train_values) - 1] = epsilon_ref[t]

        if len(train_values) == 0:
            Logger.rank0.error(
                "No trainable LJ parameters selected from 'LJ_type_param'. "
                "Add train flags (e.g. 'on_eps', 'on_sigma') or remove explicit train flags."
            )
            exit()

        model_dict["lj_sigma_idx"] = jnp.array(sigma_idx, dtype=int)
        model_dict["lj_epsilon_idx"] = jnp.array(epsilon_idx, dtype=int)
        model_dict["lj_sigma_ref"] = jnp.array(sigma_ref)
        model_dict["lj_epsilon_ref"] = jnp.array(epsilon_ref)
        model_dict["n_sigma_train"] = len(sigma_idx)
        model_dict["n_epsilon_train"] = len(epsilon_idx)
        model_dict["LJ_param"] = jnp.array(train_values)

        if constraints:
            model_dict["epsl_constraints"] = constraints

    if "bonds" in model_dict:
        Logger.rank0.warning(
            f"Bond information was provided in {file_path}, but bond optimization is not implemented yet."
        )
        pass

    args = toml_config.pop("nn")
    del args["system_args"] # Easier to parse it in get_system_options 
    ret_str = str_from_dict(args)

    # Optimizer
    if "chain" in args and args["chain"]:
        optim_list = []
        for optim in args["optimizer"]:
            fun = getattr(optax, optim.pop("name"))
            optim_list.append(fun(**optim))
        opt = optax.chain(*optim_list) 
    else:
        # fun = optax.chain(args["optimizer"].pop("name"), optax.keep_params_nonnegative())
        fun = getattr(optax, args["optimizer"].pop("name"))

        # Check if we have learning rate scheduling
        if isinstance(args["optimizer"]["learning_rate"], dict):
            learning_rate = args["optimizer"].pop("learning_rate")
            scheduler = getattr(optax, learning_rate.pop("schedule"))
            
            # TOML parses all keys as strings. TODO: Check wheter there is a better fix for this
            if learning_rate['boundaries_and_scales']:
                aux = {}
                for k, v in learning_rate['boundaries_and_scales'].items():
                    aux[int(k)] = v
                learning_rate['boundaries_and_scales'] = aux
                
            scheduler = scheduler(**learning_rate)
            opt = fun(learning_rate=scheduler, **args["optimizer"])
            
        else:
            opt = fun(**args["optimizer"])
            # opt = optax.chain(fun(**args["optimizer"]), optax.keep_params_nonnegative())

    batch_size = 1
    if "batch_size" in args and args["batch_size"] > 0:
        batch_size = args.pop("batch_size")

    # Optional gradient norm clipping (applied before the base optimizer)
    _max_grad_norm = args.pop("max_grad_norm", None)
    if _max_grad_norm is not None:
        _max_grad_norm = float(_max_grad_norm)
        opt = optax.chain(optax.clip_by_global_norm(_max_grad_norm), opt)
        Logger.rank0.info(
            f"Gradient clipping enabled: max_grad_norm={_max_grad_norm}"
        )
    args["max_grad_norm"] = _max_grad_norm

    args["optimizer"] = optax.MultiSteps(opt, every_k_schedule=batch_size)

    # Loss function
    # TODO: check that all required arguments are provided for a given loss name
    loss_function = getattr(losses, args["loss"].pop("name"))
    metric = getattr(losses, args["loss"]["metric"])

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

    # Handle boundary parameters (backward compatibility: 'boundary' → 'upper_boundary')
    if "boundary" in args["loss"] and "upper_boundary" not in args["loss"]:
        args["loss"]["upper_boundary"] = args["loss"]["boundary"]
        Logger.rank0.warning(
            f"'boundary' in [nn.loss] is deprecated. Use 'upper_boundary' instead."
        )

    args["loss"]["metric"] = metric
    args["loss_args"] = args["loss"]
    args["loss"] = loss_function
    args["name_to_type"] = name_to_type

    # Strip optimizer/clip keys that users may accidentally place inside
    # [nn.loss] instead of [nn].  These are not loss-function parameters
    # and would cause a TypeError if forwarded via **loss_args.
    _non_loss_keys = {
        "clip_sigma_min", "clip_sigma_max",
        "clip_epsilon_min", "clip_epsilon_max",
        "grad_method", "fd_epsilon", "double_precision",
        "clear_xla_cache", "max_grad_norm",
    }
    for _k in _non_loss_keys:
        if _k in args["loss_args"]:
            # Move the value to the top-level args so it's still honoured
            args.setdefault(_k, args["loss_args"].pop(_k))
            Logger.rank0.warning(
                f"'{_k}' found inside [nn.loss] — it belongs in [nn]. Moving it."
            )

    # Gradient method: "reverse" (default), "jvp", or "finite_diff"
    grad_method = args.pop("grad_method", "reverse").lower()
    if grad_method not in ("reverse", "jvp", "finite_diff"):
        Logger.rank0.error(
            f"Invalid grad_method='{grad_method}' in '{file_path}'. "
            f"Valid options: 'reverse', 'jvp', 'finite_diff'."
        )
        exit()
    args["grad_method"] = grad_method
    args["fd_epsilon"] = float(args.pop("fd_epsilon", 1e-4))
    args["double_precision"] = bool(args.pop("double_precision", False))

    # Hard clipping bounds for optimizer projection
    args["clip_sigma_min"] = float(args.pop("clip_sigma_min", 0.05))
    args["clip_sigma_max"] = float(args.pop("clip_sigma_max", 2.0))
    args["clip_epsilon_min"] = float(args.pop("clip_epsilon_min", 0.001))
    args["clip_epsilon_max"] = float(args.pop("clip_epsilon_max", 100.0))

    # XLA cache eviction interval (VMA cleanup for low vm.max_map_count hosts)
    _clear_xla_raw = args.pop("clear_xla_cache", 0)
    if isinstance(_clear_xla_raw, bool):
        clear_xla_cache = 1 if _clear_xla_raw else 0
    elif isinstance(_clear_xla_raw, int):
        clear_xla_cache = _clear_xla_raw
    elif isinstance(_clear_xla_raw, float) and _clear_xla_raw.is_integer():
        clear_xla_cache = int(_clear_xla_raw)
    else:
        Logger.rank0.error(
            f"Invalid clear_xla_cache='{_clear_xla_raw}' in '{file_path}'. "
            "Use false/0 to disable, true/1 to clear every epoch, or a "
            "positive integer N to clear every N epochs."
        )
        exit()
    if clear_xla_cache < 0:
        Logger.rank0.error(
            f"Invalid clear_xla_cache='{clear_xla_cache}' in '{file_path}'. "
            "Value must be >= 0."
        )
        exit()
    args["clear_xla_cache"] = clear_xla_cache

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
    # save copy for output parameters
    toml_copy = copy.deepcopy(toml_config)    
    args = toml_config.pop("nn")
    

    for system in systems:
        dir = system.name
        # Get data for density profile
        if "target_density" in args["system_args"][dir]:
            filename, ext = os.path.splitext(args["system_args"][dir]["target_density"])
            file_path = f"{dir}/{filename}{ext}"
            print('Reading ref density from', file_path)
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
        if "radius_of_gyration" in args["loss"]["name"]:
            n_chains = args["system_args"][dir]["n_chains"]
            chain = np.bytes_(args["system_args"][dir].pop("resname"))
            chain_indices = jnp.where(system.resnames == chain)[0]
            n_atoms_per_chain = int(len(chain_indices) / n_chains)
            
            chain_indices = jnp.reshape(chain_indices, (n_chains, n_atoms_per_chain))
            chain_masses = jnp.take(system.masses, chain_indices)
            
            args["system_args"][dir]["n_atoms_per_chain"] = n_atoms_per_chain
            args["system_args"][dir]["chain_indices"] = chain_indices
            args["system_args"][dir]["chain_masses"] = chain_masses

        # Get data for Rg Probability density function
        if "target_dist" in args["system_args"][dir]:
            filename, ext = os.path.splitext(args["system_args"][dir]["target_dist"])
            file_path = f"{dir}/{filename}{ext}"
            print('Reading ref distribution from', file_path)
            if ext == ".npy":
                reference = jnp.array(np.load(file_path))
            elif ext == ".xvg":
                reference = jnp.array(
                    # transpose so we can work with rows
                    np.loadtxt(file_path, comments=["#", "@"]).T
                )
            else:
                Logger.rank0.error(
                    f"Target distribution filename '{file_path}' has the wrong extension."
                    f"Valid extensions are '.npy' and '.xvg'."
                )
                exit()
            args["system_args"][dir]["data_range"] = reference[0]
            args["system_args"][dir]["target_dist"] = reference[1:]       

        # Get data for coordination distance losses
        if "coordination_distance" in args["loss"]["name"]:
            coord_pairs_raw = args["system_args"][dir].pop("coord_pairs")
            coord_pair_indices = []
            coord_pair_names = []
            for pair in coord_pairs_raw:
                name_a, name_b = pair[0], pair[1]
                idx_a = jnp.where(system.names == np.bytes_(name_a))[0]
                idx_b = jnp.where(system.names == np.bytes_(name_b))[0]
                if idx_a.size == 0 or idx_b.size == 0:
                    Logger.rank0.error(
                        f"Coordination pair ({name_a}, {name_b}): "
                        f"found {idx_a.size} atoms for '{name_a}' and "
                        f"{idx_b.size} atoms for '{name_b}'. "
                        f"Check atom names in H5 file."
                    )
                    exit()
                coord_pair_indices.append((idx_a, idx_b))
                coord_pair_names.append((name_a, name_b))
                Logger.rank0.info(
                    f"Coordination pair {name_a}-{name_b}: "
                    f"{idx_a.size} x {idx_b.size} atoms"
                )
            args["system_args"][dir]["coord_pair_indices"] = coord_pair_indices
            args["system_args"][dir]["coord_pairs"] = coord_pair_names

            # Load reference coordination distance distributions (.xvg / .npy)
            if "target_coord_dist" in args["system_args"][dir]:
                filename, ext = os.path.splitext(
                    args["system_args"][dir].pop("target_coord_dist")
                )
                file_path = f"{dir}/{filename}{ext}"
                Logger.rank0.info(f"Reading ref coordination dist from {file_path}")
                if ext == ".npy":
                    reference = jnp.array(np.load(file_path))
                elif ext == ".xvg":
                    reference = jnp.array(
                        np.loadtxt(file_path, comments=["#", "@"]).T
                    )
                else:
                    Logger.rank0.error(
                        f"Target coord dist filename '{file_path}' has the wrong extension. "
                        f"Valid extensions are '.npy' and '.xvg'."
                    )
                    exit()
                args["system_args"][dir]["data_range"] = reference[0]
                args["system_args"][dir]["target_dist"] = reference[1:]

            # Convert target mean distances to jnp array
            if "target_distances" in args["system_args"][dir]:
                args["system_args"][dir]["target_distances"] = jnp.array(
                    args["system_args"][dir]["target_distances"]
                )

        # Get data for tetrahedral coordination losses
        if "coordination_tetrahedral" in args["loss"]["name"]:
            coord_sites_raw = args["system_args"][dir].pop("coord_sites")
            site_metal_indices = []
            site_ligand_indices = []
            site_group_labels = []
            site_group_slices = []  # (global_start, global_end) into the flat ligand array

            global_offset = 0
            for s, site in enumerate(coord_sites_raw):
                if len(site) != 5:
                    Logger.rank0.error(
                        f"coord_sites[{s}] must have exactly 5 entries "
                        f"[metal_idx, lig1, lig2, lig3, lig4], got {len(site)}."
                    )
                    exit()
                metal_idx = int(site[0])
                lig_indices = [int(x) for x in site[1:5]]

                site_metal_indices.append(metal_idx)
                site_ligand_indices.append(jnp.array(lig_indices, dtype=jnp.int32))

                # Group ligands by atom name within this site
                lig_names = [system.names[i].decode("utf-8") for i in lig_indices]
                metal_name = system.names[metal_idx].decode("utf-8")

                # Stable sort: preserve order of first occurrence
                seen_types = []
                for n in lig_names:
                    if n not in seen_types:
                        seen_types.append(n)

                # Re-order ligand indices within the site so same-type atoms
                # are contiguous. Build groups.
                reordered = []
                for lig_type in seen_types:
                    group_start = global_offset + len(reordered)
                    count = 0
                    for idx, n in zip(lig_indices, lig_names):
                        if n == lig_type:
                            reordered.append(idx)
                            count += 1
                    group_end = group_start + count
                    label = f"site{s} {lig_type}-{metal_name}"
                    site_group_labels.append(label)
                    site_group_slices.append((group_start, group_end))

                # Replace with the reordered ligand array
                site_ligand_indices[-1] = jnp.array(reordered, dtype=jnp.int32)
                global_offset += 4

                Logger.rank0.info(
                    f"Tetrahedral site {s}: metal {metal_name} (idx {metal_idx}), "
                    f"ligands {list(zip(lig_names, lig_indices))}"
                )

            # Log the column ↔ group mapping (use site-local slot indices
            # so the user can cross-reference with target_site_distances)
            for g, label in enumerate(site_group_labels):
                gstart, gend = site_group_slices[g]
                s_owner = gstart // 4
                ls = gstart - 4 * s_owner
                le = gend   - 4 * s_owner
                Logger.rank0.info(
                    f"  Distribution group {g}: {label} "
                    f"(site-local ligand slots {ls}-{le - 1}, "
                    f"target_site_distances[{g}])"
                )

            args["system_args"][dir]["site_metal_indices"] = site_metal_indices
            args["system_args"][dir]["site_ligand_indices"] = site_ligand_indices
            args["system_args"][dir]["site_group_labels"] = site_group_labels
            args["system_args"][dir]["site_group_slices"] = site_group_slices

            # target_q: scalar → broadcast to all sites, or per-site list
            raw_q = args["system_args"][dir].pop("target_q", 1.0)
            n_sites = len(site_metal_indices)
            if isinstance(raw_q, (int, float)):
                target_q = jnp.full(n_sites, float(raw_q))
            else:
                if len(raw_q) != n_sites:
                    Logger.rank0.error(
                        f"target_q has {len(raw_q)} entries but there are "
                        f"{n_sites} coord_sites."
                    )
                    exit()
                target_q = jnp.array(raw_q)
            args["system_args"][dir]["target_q"] = target_q

            # target_site_distances (mean variant)
            if "target_site_distances" in args["system_args"][dir]:
                raw_d = args["system_args"][dir]["target_site_distances"]
                if len(raw_d) != len(site_group_labels):
                    Logger.rank0.error(
                        f"target_site_distances has {len(raw_d)} entries but "
                        f"there are {len(site_group_labels)} ligand groups."
                    )
                    exit()
                args["system_args"][dir]["target_site_distances"] = jnp.array(raw_d)

            # target_coord_dist (distribution variant)
            if "target_coord_dist" in args["system_args"][dir]:
                filename, ext = os.path.splitext(
                    args["system_args"][dir].pop("target_coord_dist")
                )
                file_path = f"{dir}/{filename}{ext}"
                Logger.rank0.info(
                    f"Reading ref tetrahedral coordination dist from {file_path}"
                )
                if ext == ".npy":
                    reference = jnp.array(np.load(file_path))
                elif ext == ".xvg":
                    reference = jnp.array(
                        np.loadtxt(file_path, comments=["#", "@"]).T
                    )
                else:
                    Logger.rank0.error(
                        f"Unsupported extension '{ext}' for target_coord_dist. "
                        f"Use '.npy' or '.xvg'."
                    )
                    exit()
                args["system_args"][dir]["data_range"] = reference[0]
                args["system_args"][dir]["target_dist"] = reference[1:]
                n_dist_cols = reference.shape[0] - 1
                if n_dist_cols != len(site_group_labels):
                    Logger.rank0.error(
                        f"target_coord_dist has {n_dist_cols} distribution columns "
                        f"but there are {len(site_group_labels)} ligand groups."
                    )
                    exit()

            # Optional: target_q_dist (q distribution XVG)
            if "target_q_dist" in args["system_args"][dir]:
                filename, ext = os.path.splitext(
                    args["system_args"][dir].pop("target_q_dist")
                )
                file_path = f"{dir}/{filename}{ext}"
                Logger.rank0.info(
                    f"Reading ref q distribution from {file_path}"
                )
                if ext == ".npy":
                    q_ref = jnp.array(np.load(file_path))
                elif ext == ".xvg":
                    q_ref = jnp.array(
                        np.loadtxt(file_path, comments=["#", "@"]).T
                    )
                else:
                    Logger.rank0.error(
                        f"Unsupported extension '{ext}' for target_q_dist. "
                        f"Use '.npy' or '.xvg'."
                    )
                    exit()
                args["system_args"][dir]["q_data_range"] = q_ref[0]
                args["system_args"][dir]["target_q_dist"] = q_ref[1:]

    # Heterogeneous multidir: parse optional `replica_of` per system_args
    # entry.  Default = the dir's own name (standalone replica group).
    # Two dirs with the same `replica_of` value are treated as replicas of
    # one logical system: they must have identical molecular topology.
    replica_map: dict[str, str] = {}
    for system in systems:
        dir = system.name
        replica_of = args["system_args"][dir].pop("replica_of", None)
        replica_map[dir] = replica_of if replica_of is not None else dir

    groups: dict[str, list] = {}
    for system in systems:
        groups.setdefault(replica_map[system.name], []).append(system)
    for group_name, members in groups.items():
        if len(members) <= 1:
            continue
        ref = members[0]
        for m in members[1:]:
            if not np.array_equal(np.asarray(m.molecules), np.asarray(ref.molecules)):
                Logger.rank0.error(
                    f"Replica group '{group_name}': system '{m.name}' has a "
                    f"different molecule layout than reference '{ref.name}'. "
                    f"Replicas must share identical molecular topology."
                )
                exit()
            if not np.array_equal(np.asarray(m.names), np.asarray(ref.names)):
                Logger.rank0.error(
                    f"Replica group '{group_name}': system '{m.name}' has a "
                    f"different atom-name list than reference '{ref.name}'. "
                    f"Replicas must share identical molecular topology."
                )
                exit()

    system_options = System_options(args["system_args"], replica_map=replica_map)


    return system_options, toml_copy
