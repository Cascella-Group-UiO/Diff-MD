import dataclasses
import os
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
    masses: np.ndarray
    charges: Optional[Array]
    config: Config
    topol: Topology
    resnames: np.ndarray # Implement
    name: str

    @classmethod
    def constructor(
        cls,
        args: Namespace,
        name_to_type: Optional[dict[str, int]] = None,
        dir: str = ".",
        model: Optional[GeneralModel] = None,
        rank: Optional[int] = None,
    ) -> Self:
        # When rank is given, check for a rank-specific coordinate file
        # (e.g. input_0003.h5).  Fall back to the shared file otherwise.
        coord_base = args.coord
        if rank is not None:
            stem, ext = os.path.splitext(coord_base)
            rank_coord = f"{stem}_{rank:04d}{ext}"
            rank_path = f"{dir}/{rank_coord}"
            if os.path.exists(rank_path):
                coord_base = rank_coord
                Logger.rank0.debug(
                    "Rank %d using per-rank coordinate file '%s'.",
                    rank, rank_path,
                )
        coord_path = f"{dir}/{coord_base}"
        is_restart = False
        try:
            with h5py.File(f"{coord_path}", "r", driver=None) as in_file:
                vel_dset = None
                box_dset = None
                box_attr = None
                has_velocity_dataset = False

                # Support both the original flat layout (``/coordinates``) and
                # H5MD output files (``/particles/all/position/value``).
                # For interrupted trajectories, trailing frames may be
                # pre-allocated zeros; scan backwards to find the last
                # usable frame.
                if "coordinates" in in_file:
                    pos_dset = in_file["coordinates"]
                elif "particles" in in_file:
                    pos_dset = in_file["particles/all/position/value"]
                    is_restart = True
                else:
                    raise KeyError("No 'coordinates' or 'particles/all/position/value' in h5 file.")

                if "velocities" in in_file:
                    vel_dset = in_file["velocities"]
                    has_velocity_dataset = True
                elif "particles" in in_file and "velocity" in in_file["particles/all"]:
                    vel_dset = in_file["particles/all/velocity/value"]
                    has_velocity_dataset = True

                if "box" in in_file.attrs:
                    box_attr = np.asarray(in_file.attrs["box"], dtype=np.float64)
                    if box_attr.shape != (3,) or not np.all(np.isfinite(box_attr)) or not np.all(box_attr > 0.0):
                        raise ValueError(
                            "Coordinate file stores an invalid static box attribute; "
                            "expected 3 finite positive box lengths."
                        )
                elif "particles/all/box/edges/value" in in_file:
                    box_dset = in_file["particles/all/box/edges/value"]

                # Walk backwards from the last frame to find a valid one.
                # A frame is invalid if the coordinates are all zero or if any
                # position / velocity / box value is non-finite.
                n_frames = pos_dset.shape[0]
                last_frame = None
                _skipped = 0
                for _fi in range(n_frames - 1, -1, -1):
                    _frame_pos = np.array(pos_dset[_fi])
                    if np.all(_frame_pos == 0.0):
                        _skipped += 1
                        continue
                    if not np.all(np.isfinite(_frame_pos)):
                        _skipped += 1
                        continue

                    if vel_dset is not None:
                        _frame_vel = np.array(vel_dset[_fi])
                        if not np.all(np.isfinite(_frame_vel)):
                            _skipped += 1
                            continue

                    if box_dset is not None:
                        _frame_box = np.array(box_dset[_fi], dtype=np.float64)
                        _frame_diag = np.diag(_frame_box)
                        if not np.all(np.isfinite(_frame_diag)) or not np.all(_frame_diag > 0.0):
                            _skipped += 1
                            continue

                    last_frame = _fi
                    break

                if last_frame is None:
                    raise ValueError(
                        "All position frames in the H5 file are "
                        "corrupt (zeros or non-finite position/velocity/box state)."
                    )

                if _skipped > 0:
                    Logger.rank0.warning(
                        "Skipped %d trailing corrupt frame(s) (zeros or non-finite state) "
                        "in '%s'; using frame %d / %d.",
                        _skipped,
                        coord_path,
                        last_frame,
                        n_frames - 1,
                    )

                positions = jnp.array(np.array(pos_dset[last_frame]))

                if vel_dset is not None:
                    velocities = jnp.array(np.array(vel_dset[last_frame]))
                else:
                    velocities = jnp.zeros_like(positions)

                # ---- Flat restart datasets (present in recent outputs) ----
                if "indices" in in_file:
                    indices = np.array(in_file["indices"])
                    # Normalize to 0-based if stored as 1-based atom IDs
                    if len(indices) > 0 and indices.min() == 1:
                        indices = indices - 1
                    types = np.array(in_file["types"])
                    names = np.array(in_file["names"])
                    molecules = np.array(in_file["molecules"])
                    masses = np.array(in_file["masses"])
                # ---- Fallback: reconstruct from H5MD structure ----
                elif "particles" in in_file:
                    n_atoms = positions.shape[0]
                    indices = np.arange(n_atoms, dtype=np.int32)
                    types = np.array(in_file["particles/all/species"])

                    # Try per-atom names first (finish.h5 written before
                    # the flat-layout fix), then fall back to the
                    # per-type ``parameters/vmd_structure/name`` array.
                    if "particles/all/names" in in_file:
                        raw_names = in_file["particles/all/names"][:]
                        names = np.array(
                            [n if isinstance(n, bytes) else n.encode("utf-8")
                             for n in raw_names],
                            dtype="S16",
                        )
                    elif "parameters/vmd_structure/name" in in_file:
                        names_per_type = np.array(
                            in_file["parameters/vmd_structure/name"]
                        )
                        names = np.array(
                            [names_per_type[t] for t in types],
                            dtype=names_per_type.dtype,
                        )
                    else:
                        raise KeyError(
                            "H5MD file has no atom name data "
                            "('particles/all/names' or "
                            "'parameters/vmd_structure/name')."
                        )

                    if "particles/all/indices" in in_file:
                        indices = np.array(
                            in_file["particles/all/indices"]
                        ).astype(np.int32)

                    mass_arr = np.array(in_file["particles/all/mass"])
                    masses = mass_arr.ravel()
                    # Trim trailing dimension if shape was (N,1)
                    if masses.shape[0] != n_atoms:
                        masses = mass_arr[:n_atoms].ravel()

                    if "parameters/vmd_structure/resid" in in_file:
                        molecules = np.array(
                            in_file["parameters/vmd_structure/resid"]
                        )
                    else:
                        molecules = np.zeros(n_atoms, dtype=np.int32)
                else:
                    raise KeyError(
                        "H5 file has neither flat restart datasets "
                        "('indices', 'types', …) nor H5MD structure "
                        "('particles/all/…')."
                    )

                if "resnames" in in_file:
                    resnames = np.array(in_file["resnames"])
                elif "parameters/vmd_structure/resname" in in_file:
                    # Per-unique-molecule → expand to per-atom via resid
                    resname_per_mol = np.array(
                        in_file["parameters/vmd_structure/resname"]
                    )
                    if len(resname_per_mol) == 1:
                        resnames = np.full(
                            len(indices), resname_per_mol[0],
                            dtype=resname_per_mol.dtype,
                        )
                    else:
                        unique_mols = np.unique(molecules)
                        resnames = np.empty(len(indices), dtype=resname_per_mol.dtype)
                        for i, mol_id in enumerate(unique_mols):
                            mask = molecules == mol_id
                            idx = min(i, len(resname_per_mol) - 1)
                            resnames[mask] = resname_per_mol[idx]
                else:
                    resnames = np.full(len(indices), b"UNK", dtype="S5")

                if "charge" in in_file:
                    charge_arr = np.array(in_file["charge"])
                elif "particles" in in_file and "charge" in in_file["particles/all"]:
                    charge_arr = np.array(in_file["particles/all/charge"])
                else:
                    charge_arr = None

                # Detect placeholder charges stored in old H5MD files
                # (e.g. all 1.0).  Real partial charges vary per atom.
                if (
                    charge_arr is not None
                    and charge_arr.size > 1
                    and np.all(charge_arr == charge_arr[0])
                ):
                    Logger.rank0.warning(
                        "Charges in H5 file are all identical (%.4g); "
                        "treating as placeholder — electrostatics disabled. "
                        "Re-run from a coordinate file with correct charges "
                        "to enable electrostatics.",
                        charge_arr[0],
                    )
                    charge_arr = None

                charges = (
                    None
                    if args.no_charges or charge_arr is None or np.all(charge_arr == 0.)
                    else jnp.reshape(charge_arr, (-1, 1))
                )

                # Box: prefer flat attribute, fall back to the validated box
                # frame matching the selected restart frame.
                if box_attr is not None:
                    box = jnp.array(box_attr)
                elif box_dset is not None:
                    box = jnp.array(np.diag(np.array(box_dset[last_frame], dtype=np.float64)))
                else:
                    raise KeyError(
                        "No box information in H5 file (neither 'box' "
                        "attribute nor 'particles/all/box/edges/value')."
                    )
        except Exception as e:
            Logger.rank0.error(
                f"Unable to parse coordinate file '{coord_path}'.", exc_info=e
            )
            exit()

        if is_restart:
            Logger.rank0.info(
                f"Restarting from trajectory file '{coord_path}' "
                f"(frame {last_frame}, {positions.shape[0]} atoms).",
            )
        Logger.rank0.info(
            f"Coordinate file '{coord_path}' parsed successfully.",
        )

        if name_to_type is not None and len(name_to_type) > 0:
            unique_atom_names = np.unique(names)
            missing = []
            for _n in unique_atom_names:
                _n_str = _n.decode("utf-8") if isinstance(_n, (bytes, np.bytes_)) else str(_n)
                if _n_str not in name_to_type:
                    missing.append(_n_str)
            if missing:
                Logger.rank0.error(
                    f"Atom name(s) {missing} in system '{dir}' are not declared "
                    f"in the training TOML's [nn.model.LJ_param] table. "
                    f"Add row(s) covering these atom names (any pair containing them) "
                    f"and re-run. Known names: {sorted(name_to_type.keys())}"
                )
                exit()

        # Set fixed mass
        # masses = 72.0

        # topol = get_topol(f"{dir}/{args.topol}", molecules, model)
        config, types = get_config(
            # fmt: off
            f"{dir}/{args.config}", names, types, masses, box, 
            charges, name_to_type, args.database,
        )

        if config.ff_family == "amber_like":
            Logger.rank0.info(
                "Amber-like atomistic mode enabled "
                f"(ensemble={config.ensemble}, combining_rule={config.combining_rule}, "
                f"coulomb14_scale={config.coulomb14_scale}, lj14_scale={config.lj14_scale}, "
                f"constrain_xh_bonds={config.constrain_xh_bonds})."
            )
            if charges is None:
                Logger.rank0.warning(
                    "Amber-like mode selected but charges are missing/disabled. "
                    "Electrostatics will be inactive."
                )

        if not config.start_temperature and not has_velocity_dataset:
            Logger.rank0.warning(
                "start_temperature is disabled, but the coordinate file carries no velocities. "
                "The run will start from zero velocities. Provide restart velocities or set "
                "start_temperature to initialize them explicitly."
            )

        topol = get_topol(f"{dir}/{args.topol}", molecules, config, model)

        return cls(
            positions,
            velocities,
            indices,
            types,
            names,
            molecules,
            masses,
            charges,
            config,
            topol,
            resnames,
            name=dir,
        )
