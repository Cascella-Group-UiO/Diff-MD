import getpass
import logging
import os
from typing import Any

import h5py
import numpy as np
import tomlkit

from .config import read_toml
from .logger import Logger
from .models import GeneralModel


class OutDataset:
    def __init__(
        self,
        destdir,
        filename,
        double_out=False,
    ):
        if double_out:
            self.float_dtype = "float64"
        else:
            self.float_dtype = "float32"

        os.makedirs(destdir, exist_ok=True)
        self.file = h5py.File(
            os.path.join(destdir, f"{filename}.h5"),
            "w",
        )

    def close_file(self):
        self.file.close()

    def flush(self):
        self.file.flush()


def save_params(filename: str, toml: dict[str, Any], params: GeneralModel) -> None:
    assert params.chi is not None

    with open(filename, "w") as outfile:
        num_pairs = len(toml["nn"]["model"]["chi"])
        num_params = len(params.chi)
        if num_pairs >= num_params:
            idx = []
            for i, pair in enumerate(toml["nn"]["model"]["chi"]):
                if len(pair) > 3 and isinstance(pair[3], str):
                    idx.append(i)
            assert len(idx) == num_params
            for i, chi in zip(idx, params.chi):
                toml["nn"]["model"]["chi"][i][2] = float(chi)
        else:
            idx = np.triu_indices(params.n_types, k=1)
            idx = params.type_to_chi[idx]
            assert len(params.chi[idx]) == num_pairs
            for i, chi in enumerate(params.chi[idx]):
                toml["nn"]["model"]["chi"][i][2] = float(chi)
        tomlkit.dump(toml, outfile)


def setup_time_dependent_element(
    name, parent_group, n_frames, shape, dtype, units=None
):
    group = parent_group.create_group(name)
    step = group.create_dataset("step", (n_frames,), "int32")
    time = group.create_dataset("time", (n_frames,), "float32")
    value = group.create_dataset("value", (n_frames, *shape), dtype)
    if units is not None:
        value.attrs["unit"] = units
        time.attrs["unit"] = "ps"
    return group, step, time, value


def store_static(
    h5md,
    names,
    types,
    indices,
    config,
    bonds_2_atom1,
    bonds_2_atom2,
    topol_mols,
    molecules=None,
    velocity_out=False,
    force_out=False,
    charges=False,  # Provide charge array here
):
    dtype = h5md.float_dtype

    h5md_group = h5md.file.create_group("/h5md")
    h5md.h5md_group = h5md_group
    h5md.observables = h5md.file.create_group("/observables")
    h5md.connectivity = h5md.file.create_group("/connectivity")
    h5md.parameters = h5md.file.create_group("/parameters")

    h5md_group.attrs["version"] = np.array([1, 1], dtype=int)
    author_group = h5md_group.create_group("author")
    author_group.attrs["name"] = np.string_(getpass.getuser())
    creator_group = h5md_group.create_group("creator")
    creator_group.attrs["name"] = np.string_("Diff-MD")

    creator_group.attrs["version"] = np.string_("0.0")

    h5md.particles_group = h5md.file.create_group("/particles")
    h5md.all_particles = h5md.particles_group.create_group("all")
    mass = h5md.all_particles.create_dataset("mass", (config.n_particles, 1), dtype)
    mass[...] = np.asarray(config.mass)

    if charges is not False:
        charge = h5md.all_particles.create_dataset(
            "charge", (config.n_particles,), dtype="float32"
        )
        charge[indices] = charges

    box = h5md.all_particles.create_group("box")
    box.attrs["dimension"] = 3
    box.attrs["boundary"] = np.array(
        [np.string_(s) for s in 3 * ["periodic"]], dtype="S8"
    )

    n_frames = config.n_steps // config.n_print
    if np.mod(config.n_steps - 1, config.n_print) != 0:
        n_frames += 1
    if np.mod(config.n_steps, config.n_print) == 1:
        n_frames += 1
    if n_frames == config.n_steps:
        n_frames += 1

    species = h5md.all_particles.create_dataset(
        "species", (config.n_particles,), dtype="i"
    )
    (
        _,
        h5md.positions_step,
        h5md.positions_time,
        h5md.positions,
    ) = setup_time_dependent_element(
        "position",
        h5md.all_particles,
        n_frames,
        (config.n_particles, 3),
        dtype,
        units="nm",
    )
    if velocity_out:
        (
            _,
            h5md.velocities_step,
            h5md.velocities_time,
            h5md.velocities,
        ) = setup_time_dependent_element(
            "velocity",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="nm ps-1",
        )
    if force_out:
        (
            _,
            h5md.forces_step,
            h5md.forces_time,
            h5md.forces,
        ) = setup_time_dependent_element(
            "force",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="kJ mol-1 nm-1",
        )
    (
        _,
        h5md.total_energy_step,
        h5md.total_energy_time,
        h5md.total_energy,
    ) = setup_time_dependent_element(
        "total_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.kinetc_energy_step,
        h5md.kinetc_energy_time,
        h5md.kinetc_energy,
    ) = setup_time_dependent_element(
        "kinetic_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.potential_energy_step,
        h5md.potential_energy_time,
        h5md.potential_energy,
    ) = setup_time_dependent_element(  # noqa: E501
        "potential_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.bond_energy_step,
        h5md.bond_energy_time,
        h5md.bond_energy,
    ) = setup_time_dependent_element(
        "bond_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.angle_energy_step,
        h5md.angle_energy_time,
        h5md.angle_energy,
    ) = setup_time_dependent_element(
        "angle_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.dihedral_energy_step,
        h5md.dihedral_energy_time,
        h5md.dihedral_energy,
    ) = setup_time_dependent_element(
        "dihedral_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    (
        _,
        h5md.field_energy_step,
        h5md.field_energy_time,
        h5md.field_energy,
    ) = setup_time_dependent_element(
        "field_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
    )
    if charges is not False:
        (
            _,
            h5md.field_q_energy_step,
            h5md.field_q_energy_time,
            h5md.field_q_energy,
        ) = setup_time_dependent_element(
            "field_q_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"
        )
    (
        _,
        h5md.total_momentum_step,
        h5md.total_momentum_time,
        h5md.total_momentum,
    ) = setup_time_dependent_element(  # noqa: E501
        "total_momentum",
        h5md.observables,
        n_frames,
        (3,),
        dtype,
        units="nm g ps-1 mol-1",
    )
    # (
    #     _,
    #     h5md.angular_momentum_step,
    #     h5md.angular_momentum_time,
    #     h5md.angular_momentum,
    # ) = setup_time_dependent_element(  # noqa: E501
    #     "angular_momentum",
    #     h5md.observables,
    #     n_frames,
    #     (3,),
    #     dtype,
    #     units="nm+2 g ps-1 mol-1",
    # )
    # (
    #     _,
    #     h5md.torque_step,
    #     h5md.torque_time,
    #     h5md.torque,
    # ) = setup_time_dependent_element(  # noqa: E501
    #     "torque",
    #     h5md.observables,
    #     n_frames,
    #     (3,),
    #     dtype,
    #     units="kJ nm+2 mol-1",
    # )
    (
        _,
        h5md.temperature_step,
        h5md.temperature_time,
        h5md.temperature,
    ) = setup_time_dependent_element(
        "temperature", h5md.observables, n_frames, (3,), dtype, units="K"
    )

    (
        _,
        h5md.pressure_step,
        h5md.pressure_time,
        h5md.pressure,
    ) = setup_time_dependent_element(
        "pressure", h5md.observables, n_frames, (3,), dtype, units="bar"
    )
    (
        _,
        h5md.box_step,
        h5md.box_time,
        h5md.box_value,
    ) = setup_time_dependent_element(
        "edges", box, n_frames, (3, 3), "float32", units="nm"
    )

    for i in indices:
        species[i] = types[i]

    h5md.parameters.attrs["config.toml"] = np.string_(str(config))
    vmd_group = h5md.parameters.create_group("vmd_structure")
    index_of_species = vmd_group.create_dataset(
        "indexOfSpecies", (config.n_types,), "i"
    )
    index_of_species[:] = np.array(list(range(config.n_types)))

    # VMD-h5mdplugin maximum name/type name length is 16 characters (for
    # whatever reason [VMD internals?]).
    name_dataset = vmd_group.create_dataset("name", (config.n_types,), "S16")

    if molecules is not None:
        resid_dataset = vmd_group.create_dataset("resid", (config.n_particles,), "i")
        resid_dataset[indices] = molecules

        unique_mols = np.unique(molecules)
        resname_dataset = vmd_group.create_dataset("resname", (len(unique_mols),), "S8")

        prev = 0
        for resname, n in topol_mols:
            resname_dataset[(unique_mols >= prev) & (unique_mols < prev + n)] = (
                np.string_(resname)
            )
            prev += n

    _, name_idx = np.unique(names, return_index=True)
    unique_names = names[np.sort(name_idx)]

    for i, n in enumerate(unique_names):
        name_dataset[i] = np.string_(n.decode("utf-8")[:16])

    total_bonds = len(bonds_2_atom1)
    bonds_from = vmd_group.create_dataset("bond_from", (total_bonds,), "i")
    bonds_to = vmd_group.create_dataset("bond_to", (total_bonds,), "i")
    for i in range(total_bonds):
        a = bonds_2_atom1[i]
        b = bonds_2_atom2[i]
        bonds_from[i] = indices[a] + 1
        bonds_to[i] = indices[b] + 1


def store_data(
    h5md,
    step,
    frame,
    indices,
    positions,
    velocities,
    forces,
    temperature,
    pressure,
    kinetic_energy,
    bond2_energy,
    bond3_energy,
    bond4_energy,
    field_energy,
    field_q_energy,
    config,
    velocity_out=False,
    force_out=False,
    charge_out=False,
    dump_per_particle=False,
):
    for dset in (
        h5md.positions_step,
        h5md.total_energy_step,
        h5md.potential_energy,
        h5md.kinetc_energy_step,
        h5md.bond_energy_step,
        h5md.angle_energy_step,
        h5md.dihedral_energy_step,
        h5md.field_energy_step,
        h5md.total_momentum_step,
        # h5md.angular_momentum_step,
        # h5md.torque_step,
        h5md.temperature_step,
        h5md.pressure_step,
        h5md.box_step,
    ):
        dset[frame] = step

    for dset in (
        h5md.positions_time,
        h5md.total_energy_time,
        h5md.potential_energy_time,
        h5md.kinetc_energy_time,
        h5md.bond_energy_time,
        h5md.angle_energy_time,
        h5md.dihedral_energy_time,
        h5md.field_energy_time,
        h5md.total_momentum_time,
        # h5md.angular_momentum_time,
        # h5md.torque_time,
        h5md.temperature_time,
        h5md.pressure_time,
        h5md.box_time,
    ):
        dset[frame] = step * config.outer_ts

    if velocity_out:
        h5md.velocities_step[frame] = step
        h5md.velocities_time[frame] = step * config.outer_ts
    if force_out:
        h5md.forces_step[frame] = step
        h5md.forces_time[frame] = step * config.outer_ts
    if charge_out:
        h5md.field_q_energy_step[frame] = step
        h5md.field_q_energy_time[frame] = step * config.outer_ts

    ind_sort = np.argsort(indices)
    # positions, velocities and forces are already np.ndarryas
    h5md.positions[frame, indices[ind_sort]] = positions[ind_sort]

    if velocity_out:
        h5md.velocities[frame, indices[ind_sort]] = velocities[ind_sort]
    if force_out:
        h5md.forces[frame, indices[ind_sort]] = forces[ind_sort]
    if charge_out:
        h5md.field_q_energy[frame] = field_q_energy

    potential_energy = (
        bond2_energy + bond3_energy + bond4_energy + field_energy + field_q_energy
    )

    total_momentum = np.sum(config.mass * velocities, axis=0)
    # angular_momentum = config.mass * np.sum(np.cross(positions, velocities), axis=0)
    # torque = config.mass * np.sum(np.cross(positions, forces), axis=0)

    h5md.total_energy[frame] = kinetic_energy + potential_energy
    h5md.potential_energy[frame] = potential_energy
    h5md.kinetc_energy[frame] = kinetic_energy
    h5md.bond_energy[frame] = bond2_energy
    h5md.angle_energy[frame] = bond3_energy
    h5md.dihedral_energy[frame] = bond4_energy
    h5md.field_energy[frame] = field_energy
    h5md.total_momentum[frame, :] = total_momentum
    # h5md.angular_momentum[frame, :] = angular_momentum
    # h5md.torque[frame, :] = torque
    h5md.temperature[frame] = temperature
    h5md.pressure[frame] = pressure
    for d in range(3):
        h5md.box_value[frame, d, d] = config.box_size[d]

    fmt_ = [
        "step",
        "time",
        "temp",
        "tot E",
        "kin E",
        "pot E",
        "Field E",
        "Elec E",
        "bond E",
        "ang E",
        "dih E",
        "Px",
        "Py",
        "Pz",
    ]
    fmt_ = np.array(fmt_)

    # create mask to show only energies != 0
    en_array = np.array(
        [
            field_energy,
            field_q_energy,
            bond2_energy,
            bond3_energy,
            bond4_energy,
        ]
    )
    mask = np.full_like(fmt_, True, dtype=bool)
    mask[range(6, 11)] = en_array != 0.0

    divide_by = 1.0
    if dump_per_particle:
        for i in range(3, 9):
            fmt_[i] = fmt_[i][:-2] + "E/N"
        fmt_[-1] += "/N"
        divide_by = config.n_particles
    total_energy = kinetic_energy + potential_energy

    header_ = fmt_[mask].shape[0] * "{:>13}"
    header = header_.format(*fmt_[mask])

    data_fmt = f'{"{:13}"}{(fmt_[mask].shape[0] -1 ) * "{:13.5g}" }'
    all_data = (
        step,
        config.outer_ts * step,
        temperature,
        total_energy / divide_by,
        kinetic_energy / divide_by,
        potential_energy / divide_by,
        field_energy / divide_by,
        field_q_energy / divide_by,
        bond2_energy / divide_by,
        bond3_energy / divide_by,
        bond4_energy / divide_by,
        total_momentum[0] / divide_by,
        total_momentum[1] / divide_by,
        total_momentum[2] / divide_by,
    )
    data = data_fmt.format(*[val for i, val in enumerate(all_data) if mask[i]])
    Logger.rank0.log(logging.INFO, ("\n" + header + "\n" + data))


def write_full_trajectory(
    h5md,
    trj_dict,
    indices,
    config,
    velocity_out=False,
    force_out=False,
    charge_out=False,
):
    for frame, (  # type: ignore
        angle_energy,
        bond_energy,
        box_sizes,
        dih_energy,
        elec_energy,
        field_energy,
        forces,
        kinetic_energy,
        positions,
        temperature,
        velocities,
    ) in enumerate(zip(*trj_dict.values())):
        for dset in (
            h5md.positions_step,
            h5md.total_energy_step,
            h5md.potential_energy,
            h5md.kinetc_energy_step,
            h5md.bond_energy_step,
            h5md.angle_energy_step,
            h5md.dihedral_energy_step,
            h5md.field_energy_step,
            h5md.total_momentum_step,
            h5md.temperature_step,
            h5md.box_step,
        ):
            dset[frame] = frame

        for dset in (
            h5md.positions_time,
            h5md.total_energy_time,
            h5md.potential_energy_time,
            h5md.kinetc_energy_time,
            h5md.bond_energy_time,
            h5md.angle_energy_time,
            h5md.dihedral_energy_time,
            h5md.field_energy_time,
            h5md.total_momentum_time,
            h5md.temperature_time,
            h5md.box_time,
        ):
            dset[frame] = frame * config.outer_ts

        if velocity_out:
            h5md.velocities_step[frame] = frame
            h5md.velocities_time[frame] = frame * config.outer_ts
        if force_out:
            h5md.forces_step[frame] = frame
            h5md.forces_time[frame] = frame * config.outer_ts
        if charge_out:
            h5md.field_q_energy_step[frame] = frame
            h5md.field_q_energy_time[frame] = frame * config.outer_ts

        ind_sort = np.argsort(indices)
        h5md.positions[frame, indices[ind_sort]] = np.asarray(positions)[ind_sort]

        if velocity_out:
            h5md.velocities[frame, indices[ind_sort]] = np.asarray(velocities)[ind_sort]
        if force_out:
            h5md.forces[frame, indices[ind_sort]] = np.asarray(forces)[ind_sort]
        if charge_out:
            h5md.field_q_energy[frame] = elec_energy

        potential_energy = (
            bond_energy + angle_energy + dih_energy + field_energy + elec_energy
        )

        total_momentum = np.sum(config.mass * np.asarray(velocities), axis=0)

        h5md.total_energy[frame] = kinetic_energy + potential_energy
        h5md.potential_energy[frame] = potential_energy
        h5md.kinetc_energy[frame] = kinetic_energy
        h5md.bond_energy[frame] = bond_energy
        h5md.angle_energy[frame] = angle_energy
        h5md.dihedral_energy[frame] = dih_energy
        h5md.field_energy[frame] = field_energy
        h5md.total_momentum[frame, :] = total_momentum
        h5md.temperature[frame] = temperature
        for d in range(3):
            h5md.box_value[frame, d, d] = box_sizes[d]
