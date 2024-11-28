import dataclasses
from argparse import Namespace
from typing import Optional, Self

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from .config import Config, get_config
from .logger import Logger
from .models import GeneralModel
from .topology import Topology, get_topol


@dataclasses.dataclass
class System:
    positions: Array
    velocities: Array
    indices: np.ndarray
    types: Array
    names: np.ndarray
    molecules: np.ndarray
    charges: Optional[Array]
    config: Config
    topol: Topology
    name: str

    @classmethod
    def constructor(
        cls,
        args: Namespace,
        name_to_type: Optional[dict[str, int]] = None,
        dir: str = ".",
        model: Optional[GeneralModel] = None,
    ) -> Self:
        coord_path = f"{dir}/{args.coord}"
        try:
            with h5py.File(f"{coord_path}", "r", driver=None) as in_file:
                positions = jnp.array(np.array(in_file["coordinates"])[-1, :, :])
                velocities = (
                    jnp.array(np.array(in_file["velocities"])[-1, :, :])
                    if "velocities" in in_file
                    else jnp.zeros_like(positions)
                )
                indices = np.array(in_file["indices"])
                types = np.array(in_file["types"])
                names = np.array(in_file["names"])
                molecules = np.array(in_file["molecules"])

                # Added in the h5py to test dynamics with different masses
                # masses = in_file["masses"][:]

                charges = (
                    None
                    if args.no_charges
                    else jnp.reshape(np.array(in_file["charge"]), (-1, 1))
                )
                box = jnp.array(in_file.attrs["box"])
        except Exception as e:
            Logger.rank0.error(
                f"Unable to parse coordinate file '{coord_path}'.", exc_info=e
            )
            exit()

        Logger.rank0.info(
            f"Coordinate file '{coord_path}' parsed successfully.",
        )
        # Set fixed mass
        masses = 72.0

        topol = get_topol(f"{dir}/{args.topol}", molecules, model)
        config, types = get_config(
            # fmt: off
            f"{dir}/{args.config}", names, types, masses, box, 
            charges, name_to_type, args.database,
        )

        return cls(
            positions,
            velocities,
            indices,
            types,
            names,
            molecules,
            charges,
            config,
            topol,
            name=dir,
        )
