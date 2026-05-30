import os
import sys
from argparse import Namespace

import h5py
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from diff_md.force import get_dihedral_energy_and_forces, get_impropers_energy_and_forces
from diff_md.input_parser import System


def generate_input_h5(example_dir: str) -> str:
    h5_path = os.path.join(example_dir, "input.h5")

    coordinates = np.array(
        [
            [
                [0.10, 0.10, 0.10],
                [0.24, 0.10, 0.10],
                [0.36, 0.18, 0.10],
                [0.45, 0.24, 0.12],
            ]
        ],
        dtype=np.float64,
    )
    velocities = np.zeros_like(coordinates)
    indices = np.array([1, 2, 3, 4], dtype=np.int32)
    types = np.array([0, 1, 2, 3], dtype=np.int32)
    names = np.array([b"C", b"N", b"O", b"H"])
    molecules = np.array([0, 0, 0, 0], dtype=np.int32)
    masses = np.array([12.01, 14.01, 16.00, 1.01], dtype=np.float64)
    resnames = np.array([b"TEST", b"TEST", b"TEST", b"TEST"])
    charges = np.array([0.30, -0.30, -0.20, 0.20], dtype=np.float64)
    box = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    with h5py.File(h5_path, "w") as out:
        out.create_dataset("coordinates", data=coordinates)
        out.create_dataset("velocities", data=velocities)
        out.create_dataset("indices", data=indices)
        out.create_dataset("types", data=types)
        out.create_dataset("names", data=names)
        out.create_dataset("molecules", data=molecules)
        out.create_dataset("masses", data=masses)
        out.create_dataset("resnames", data=resnames)
        out.create_dataset("charge", data=charges)
        out.attrs["box"] = box

    return h5_path


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo_root)
    example_dir = os.path.join("examples", "atomistic")
    os.makedirs(example_dir, exist_ok=True)

    h5_path = generate_input_h5(example_dir)

    args = Namespace(
        coord="input.h5",
        topol="topol.toml",
        config="options.toml",
        no_charges=False,
        database=None,
    )

    system = System.constructor(args=args, dir=example_dir)
    positions = jnp.asarray(system.positions)
    box = jnp.asarray(system.config.box_size)

    dihedral_energy, _, _, _ = get_dihedral_energy_and_forces(
        jnp.zeros_like(positions),
        positions,
        box,
        *system.topol.bonds_4,
        ff_family=system.config.ff_family,
    )

    improper_energy, _, _ = get_impropers_energy_and_forces(
        jnp.zeros_like(positions),
        positions,
        box,
        *system.topol.bonds_impr,
        compute_pressure=False,
    )

    print(f"OK input: {h5_path}")
    print(f"ff_family: {system.config.ff_family}")
    print(f"n dihedrals: {system.topol.dihedrals}")
    print(f"n impropers: {system.topol.impropers}")
    print(f"dihedral_energy: {float(dihedral_energy):.6f}")
    print(f"improper_energy: {float(improper_energy):.6f}")


if __name__ == "__main__":
    main()
