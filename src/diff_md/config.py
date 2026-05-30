from dataclasses import dataclass
from typing import Any, Optional, Self, Tuple, Union
import sys

import jax.numpy as jnp
import numpy as np
import tomli
from flax import struct
from jax import Array, jit
from jax.typing import ArrayLike

from .cmap import CmapGridBank, parse_cmap_section
from .logger import Logger


@struct.dataclass
class BoxState:
    """Dynamic box state carried through lax.scan for NPT simulations.

    All fields are JAX-traced pytree leaves so they can change every MD step.
    For NVT the box_state passes through the scan carry unchanged.
    """
    box_size: Array              # (3,) edge lengths
    volume: Array                # scalar
    volume_per_cell: Array       # scalar (volume / n_mesh_cells)
    k_vector: tuple              # (kx, ky, kz) broadcast-shaped for FFT
    k_meshgrid: list[Array]      # [kx_3d, ky_3d, kz_3d] full meshgrid

    @classmethod
    def from_config(cls, config: "Config") -> "BoxState":
        return cls(
            box_size=config.box_size,
            volume=config.volume,
            volume_per_cell=config.volume_per_cell,
            k_vector=config.k_vector,
            k_meshgrid=config.k_meshgrid,
        )

    def rescale(self, scaling: Array, mesh_shape: tuple, fft_shape: tuple) -> "BoxState":
        """Apply barostat box scaling — mirrors Config.update_box logic."""
        new_box = scaling * self.box_size
        new_volume = jnp.prod(new_box)

        step = new_box / (2 * jnp.pi * jnp.array(mesh_shape, dtype=new_box.dtype))
        kx = jnp.fft.fftfreq(mesh_shape[0], step[0])
        ky = jnp.fft.fftfreq(mesh_shape[1], step[1])
        kz = jnp.fft.rfftfreq(mesh_shape[2], step[2])
        m_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")
        k_vector = (
            kx.reshape(fft_shape[0], 1, 1),
            ky.reshape(1, fft_shape[1], 1),
            kz.reshape(1, 1, fft_shape[2]),
        )

        n_mesh_cells = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
        return self.replace(
            box_size=new_box,
            volume=new_volume,
            volume_per_cell=new_volume / n_mesh_cells,
            k_vector=k_vector,
            k_meshgrid=m_grid,
        )


@dataclass(frozen=True)
class ZeroMeshTemplate:
    shape: tuple[int, int, int]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype).name)

    @property
    def at(self):
        return jnp.zeros(self.shape, dtype=jnp.dtype(self.dtype)).at

    def __array__(self, dtype=None):
        arr = np.zeros(self.shape, dtype=self.dtype)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __jax_array__(self):
        return jnp.zeros(self.shape, dtype=jnp.dtype(self.dtype))


@struct.dataclass
class ThermostatGroup:
    n_particles: int = struct.field(pytree_node=False)
    mask: Array


@struct.dataclass
class Config:
    box_size: ArrayLike
    volume: ArrayLike
    volume_per_cell: ArrayLike
    k_vector: tuple
    k_meshgrid: list[Array]

    n_steps: int = struct.field(pytree_node=False)
    n_particles: int = struct.field(pytree_node=False)
    n_types: int = struct.field(pytree_node=False)
    dielectric_const: float = struct.field(pytree_node=False)
    # By defining "struct.field(pytree_node=False)" The field will not
    # participate in JAX's transformations. This is used for quantities that are not differentiable
    # - More efficience.

    # Derived quantities
    particle_per_type: dict = struct.field(pytree_node=False)
    range_types: Optional[tuple[int, ...]] = struct.field(pytree_node=False)
    unique_types: tuple = struct.field(pytree_node=False)
    type_to_charge_map: dict = struct.field(pytree_node=False)
    inner_ts: float = struct.field(pytree_node=False)
    outer_ts: float = struct.field(pytree_node=False)
    empty_mesh: ZeroMeshTemplate = struct.field(pytree_node=False)
    n_mesh_cells: int = struct.field(pytree_node=False)
    fft_shape: tuple = struct.field(pytree_node=False)
    # window: ArrayLike = struct.field(pytree_node=False)
    elec_const: float = struct.field(pytree_node=False)
    self_energy: ArrayLike

    # Field options
    mesh_size: Array
    rho0: float = struct.field(pytree_node=False)
    kappa: float = struct.field(pytree_node=False)
    sigma: float = struct.field(pytree_node=False) 

    # LJ options
    LJ_param: Array
    sgm_dict: dict = struct.field(pytree_node=False)
    epsl_dict: dict = struct.field(pytree_node=False)
    type_to_LJ: Array
    sgm_table: Array
    epsl_table: Array

    rv: float = struct.field(pytree_node=False)
    rc: float = struct.field(pytree_node=False)
    rlj: float = struct.field(pytree_node=False)
    ns_nlist: int = struct.field(pytree_node=False)
    nrexcl: int = struct.field(pytree_node=False)
    epsilon_rf: float = struct.field(pytree_node=False)
    skin: float = struct.field(pytree_node=False, default=0.0)
    nlist_method: str = struct.field(pytree_node=False, default="cell")
    nlist_capacity_multiplier: float = struct.field(pytree_node=False, default=1.25)

    #add default none for the thermostat to debug
    thermostat_coupling_groups: Optional[tuple[ThermostatGroup, ...]] = None
    # Force-field options
    ff_family: str = struct.field(pytree_node=False, default="martini")
    combining_rule: str = struct.field(pytree_node=False, default="pairtable")
    lj_input_source: str = struct.field(pytree_node=False, default="auto")
    lj_force_shift: bool = struct.field(pytree_node=False, default=False)
    coulomb14_scale: float = struct.field(pytree_node=False, default=1.0)
    lj14_scale: float = struct.field(pytree_node=False, default=1.0)
    constrain_xh_bonds: bool = struct.field(pytree_node=False, default=False)
    unwrap_output: bool = struct.field(pytree_node=False, default=True)
    ensemble: str = struct.field(pytree_node=False, default="NVT")
    thermostat: str = struct.field(pytree_node=False, default="v-rescale")
    respa_inner: int = struct.field(pytree_node=False, default=1)
    pme_order: int = struct.field(pytree_node=False, default=2)
    n_print: int = struct.field(pytree_node=False, default=0)
    n_flush: int = struct.field(pytree_node=False, default=1)

    # Optional CMAP backbone-correction bank (only used when
    # ``ff_family == "amber19sb"``). Stored as static metadata so two
    # different banks do not collide in the JIT trace cache.
    cmap_grid_bank: Optional[Any] = struct.field(pytree_node=False, default=None)

    # Gas constant (in kJ/mol K)
    R: float = struct.field(pytree_node=False, default=0.00831446261815324)
    mass: Union[float, ArrayLike] = 72.0
    cancel_com_momentum: Union[int, bool] = struct.field(
        pytree_node=False, default=False
    )
    coulombtype: int = struct.field(pytree_node=False, default=0)
    elec_conversion: float = struct.field(
        pytree_node=False, default=(138.935458)
    )
    ewald_rtol: float = struct.field(pytree_node=False, default=1e-5)

    # NVT options
    tau_t: float = struct.field(pytree_node=False, default=None)
    start_temperature: Union[float, bool] = struct.field(
        pytree_node=False, default=None
    )
    target_temperature: Union[float, bool] = struct.field(
        pytree_node=False, default=None
    )

    # NPT options
    a: float = struct.field(pytree_node=False, default=None)
    pressure: bool = struct.field(pytree_node=False, default=False)
    # Possible barostat values: 0 = "no", 1 = "berendsen", 2 = "scr"
    barostat: int = struct.field(pytree_node=False, default=0)
    barostat_type: int = struct.field(pytree_node=False, default=None)
    tau_p: float = struct.field(pytree_node=False, default=None)
    n_b: int = struct.field(pytree_node=False, default=1)

    # Isothermal compressibility (in bar^(-1), defaults to water)
    beta: float = struct.field(pytree_node=False, default=3.6e-5)
    target_pressure: ArrayLike = None

    # Pressure conversion constant
    p_conv: float = struct.field(pytree_node=False, default=16.605390666)

    def __post_init__(self) -> None:
        if self.range_types is not None and not isinstance(self.range_types, tuple):
            range_values = np.asarray(self.range_types).reshape(-1)
            object.__setattr__(self, "range_types", tuple(int(v) for v in range_values.tolist()))

        if not isinstance(self.n_mesh_cells, int):
            object.__setattr__(self, "n_mesh_cells", int(np.asarray(self.n_mesh_cells)))

        if not isinstance(self.empty_mesh, ZeroMeshTemplate):
            if self.empty_mesh is None:
                mesh_shape = tuple(int(v) for v in np.asarray(self.mesh_size).tolist())
                mesh_dtype = self.box_size.dtype
            else:
                mesh_shape = tuple(int(v) for v in np.shape(self.empty_mesh))
                mesh_dtype = getattr(self.empty_mesh, "dtype", self.box_size.dtype)
            object.__setattr__(self, "empty_mesh", ZeroMeshTemplate(mesh_shape, str(mesh_dtype)))

        groups = self.thermostat_coupling_groups
        if groups is not None:
            if len(groups) == 2 and isinstance(groups[0], (int, np.integer)):
                groups = (groups,)
            elif isinstance(groups, list):
                groups = tuple(groups)

            normalized_groups = []
            for group in groups:
                if isinstance(group, ThermostatGroup):
                    normalized_groups.append(group)
                    continue

                normalized_groups.append(
                    ThermostatGroup(
                        n_particles=int(group[0]),
                        mask=jnp.asarray(group[1], dtype=jnp.int32),
                    )
                )

            object.__setattr__(self, "thermostat_coupling_groups", tuple(normalized_groups))

    @jit
    def window(self) -> Array:
        """Fourier transform of the window filter H"""
        kx, ky, kz = self.k_meshgrid
        return jnp.exp(-0.5 * self.sigma * self.sigma * (kx * kx + ky * ky + kz * kz))

    @jit
    def knorm(self) -> Array:
        """|k|² with the first frequency set to 1 (to avoid discontinuities).
        Used for the calculation of the electrostatic forces"""
        kx, ky, kz = self.k_meshgrid
        # Prevents nan
        kx = kx.at[0, 0, 0].set(1)
        ky = ky.at[0, 0, 0].set(1)
        kz = kz.at[0, 0, 0].set(1)
        return kx * kx + ky * ky + kz * kz

    @jit
    def update_box(self, scaling: Array) -> Self:
        new_box = scaling * self.box_size
        new_volume = jnp.prod(new_box)

        step = new_box / (2 * jnp.pi * self.mesh_size)
        kx = jnp.fft.fftfreq(self.empty_mesh.shape[0], step[0])
        ky = jnp.fft.fftfreq(self.empty_mesh.shape[1], step[1])
        kz = jnp.fft.rfftfreq(self.empty_mesh.shape[2], step[2])
        m_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")
        k_vector = (
            kx.reshape(self.fft_shape[0], 1, 1),
            ky.reshape(1, self.fft_shape[1], 1),
            kz.reshape(1, 1, self.fft_shape[2]),
        )

        return self.replace(
            box_size=new_box,
            volume=new_volume,
            volume_per_cell=new_volume / self.n_mesh_cells,
            k_vector=k_vector,
            k_meshgrid=m_grid,
        )
      

    @classmethod
    def constructor(
        cls,
        types: np.ndarray,
        charges: Optional[Array],
        **args: Any,
    ) -> Self:
        def elec_self_energy(charges, elec_conversion, coulombtype, sigma, erf, er, rc):
            if coulombtype == 1: # PME
                prefac = elec_conversion * jnp.sqrt(1.0 / (2.0 * jnp.pi * sigma * sigma))
                self_energy = prefac * jnp.sum(charges * charges)
            elif coulombtype == 2: # Reaction field
                # prefac = elec_conversion * (3 * erf / (2 * erf + er)) / (2 * rc)
                prefac = elec_conversion * (3 * erf / (2 * erf + er)) / rc 
                self_energy = prefac * 0.5 * (jnp.sum(charges * charges) - (1/erf) * (jnp.sum(charges) ** 2))
            else:
                self_energy = None
            return self_energy   

        mesh_shape = tuple(int(v) for v in np.asarray(args["mesh_size"]).tolist())

        step = args["box_size"] / (2 * jnp.pi * args["mesh_size"])
        total_volume = jnp.prod(args["box_size"])
        n_mesh_cells = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
        n_particles = len(types)

        kx = jnp.fft.fftfreq(mesh_shape[0], step[0])
        ky = jnp.fft.fftfreq(mesh_shape[1], step[1])
        kz = jnp.fft.rfftfreq(mesh_shape[2], step[2])
        fft_shape = (len(kx), len(ky), len(kz))
        k_vector = (
            kx.reshape(fft_shape[0], 1, 1),
            ky.reshape(1, fft_shape[1], 1),
            kz.reshape(1, 1, fft_shape[2]),
        )
        volume_per_cell = total_volume / n_mesh_cells
        m_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")
        # window = fourier_window(m_grid, args["sigma"])
        elec_conversion = 138.935458 / args["dielectric_const"]
        args["elec_conversion"] = elec_conversion
        
        elec_const = 4.0 * jnp.pi * elec_conversion
        # elec_const = elec_transfer_constant(m_grid, elec_conversion)

        if charges is not None:
            # NOTE: charges could be not type dependant
            self_energy = elec_self_energy(
                charges, 
                elec_conversion,
                args["coulombtype"], 
                args["sigma"],
                args["epsilon_rf"],
                args["dielectric_const"],
                args["rc"]
            )
            type_to_charge_map = {}
            for i in range(types.shape[0]):
                if types[i] not in type_to_charge_map.keys():
                    type_to_charge_map[types[i]] = charges[i]
        else:
            self_energy = 0.0
            type_to_charge_map = {}
            for i in range(types.shape[0]):
                if types[i] not in type_to_charge_map.keys():
                    type_to_charge_map[types[i]] = 0.0

        return cls(
            volume=total_volume,
            n_particles=n_particles,
            k_vector=k_vector,
            k_meshgrid=m_grid,
            fft_shape=fft_shape,
            volume_per_cell=volume_per_cell,
            empty_mesh=ZeroMeshTemplate(mesh_shape, str(args["box_size"].dtype)),
            elec_const=elec_const,
            self_energy=self_energy,
            type_to_charge_map=type_to_charge_map,
            n_mesh_cells=n_mesh_cells,
            **args,
        )
    
    def __str__(self) -> str:
        ret_str = f'\n\n\tSimulation parameters: \n\t{50 * "-"}\n'
        for k, v in self.__dict__.items():
            if k not in (
                "empty_mesh",
                "k_vector",
                "k_meshgrid",
                "LJ_param",
                "sgm_dict",
                "epsl_dict",
                "type_to_LJ",
                "sgm_table",
                "epsl_table",
                "window",
                "elec_const",
                "thermostat_coupling_groups",
                "type_to_charge_map",
                "cmap_grid_bank",
            ):
                ret_str += f"\t{k}: {v}\n"
        return ret_str


def read_toml(file_path: str) -> dict[str, Any]:
    with open(file_path, "rb") as in_file:
        toml_content = tomli.load(in_file)
    return toml_content


def get_type_to_LJ(n: int) -> np.ndarray:
    """
    This function builds a matrix that maps types to χ values, including the self interaction.
    n is the number of unique types in the system.
    """
    idx_triu = np.triu_indices(n)
    arr = np.arange(len(idx_triu[0]))
    square_matrix = np.zeros((n, n), dtype=int)
    square_matrix[idx_triu] = arr
    square_matrix += np.triu(square_matrix, k=1).T
    return square_matrix


def get_config(
    file_path: str,
    names: np.ndarray,
    types: np.ndarray,
    masses: np.ndarray | float,
    box_size: ArrayLike,
    charges: Optional[Array],
    ext_name_to_type: dict[str, int] | None,
    database: str | None,
) -> Tuple[Config, Array]:
    try:
        toml_config = read_toml(file_path)
    except Exception as e:
        Logger.rank0.error(f"Unable to parse config file '{file_path}'.", exc_info=e)
        exit()
    
    if database is not None:
        try:
            db_data = read_toml(database)
        except Exception as e:
            Logger.rank0.error(
                f"Unable to parse model database file '{database}'.", exc_info=e
            )
            exit()

        Logger.rank0.info(
            f"Model database file '{database}' parse successfully.",
        )
        db_chi = db_data["nn"]["model"].pop("LJ_param")

    config_dict = {}


    _, name_idx = np.unique(names, return_index=True)
    unique_names = names[np.sort(name_idx)]
    unique_types = types[np.sort(name_idx)]
    particle_per_type = {t: len(types[types == t]) for t in unique_types}
    n_types = len(unique_names)
    n_particles = len(types)
    total_volume = jnp.prod(box_size) 

    name_to_type = {}
    for n, t in zip(unique_names, unique_types):
        n = n.decode("utf-8")
        if n not in name_to_type:
            name_to_type[n] = t
    
    # name_to_type = {name.decode("utf-8"): type for type, name in enumerate(unique_names)}

    config_dict["box_size"] = box_size
    config_dict["range_types"] = tuple(range(n_types))
    config_dict["n_types"] = n_types
    config_dict["particle_per_type"] = particle_per_type
    config_dict["sgm_dict"] = {}
    config_dict["epsl_dict"] = {}
    config_dict["sgm_table"] = {}
    config_dict["epsl_table"] = {}
    config_dict["type_to_LJ"] = (ttlj := jnp.asarray(get_type_to_LJ(n_types), dtype=jnp.int32))

    config_dict["dielectric_const"] = 1.0  # default

    # if len(jnp.unique(masses)) > 1:
    #     config_dict["mass"] = jnp.reshape(masses, (-1, 1))
        # config_dict["mass"] = masses

    config_dict["mass"] = jnp.reshape(masses, (-1, 1))

    for k, v in toml_config.items():
        if isinstance(v, dict):
            if k in ("simulation", "field", "atomistic_ff"):
                for nested_k, nested_v in v.items():
                    config_dict[nested_k] = nested_v
        else:
            config_dict[k] = v

    def _normalize_type_name(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    name_to_type_lower = {k.lower(): v for k, v in name_to_type.items()}

    def _resolve_type_name(name: str) -> int | None:
        if name in name_to_type:
            return name_to_type[name]

        lname = name.lower()
        if lname in name_to_type_lower:
            return name_to_type_lower[lname]

        aliases = []
        if name.endswith("_spc"):
            aliases.append(name[:-1])
        elif name.endswith("_sp"):
            aliases.append(name + "c")

        for alias in aliases:
            if alias in name_to_type:
                return name_to_type[alias]
            lalias = alias.lower()
            if lalias in name_to_type_lower:
                return name_to_type_lower[lalias]

        return None

    def _set_lj_tables(
        sgm: np.ndarray,
        epsl: np.ndarray,
        sgm_dict: dict,
        epsl_dict: dict,
    ) -> None:
        config_dict["sgm_dict"] = sgm_dict
        config_dict["epsl_dict"] = epsl_dict

        config_dict["epsl_table"] = jnp.array(epsl + epsl.T - np.diag(np.diag(epsl)))
        config_dict["sgm_table"] = jnp.array(sgm + sgm.T - np.diag(np.diag(sgm)))

        lj_len = int(np.max(ttlj)) + 1
        lj_param = jnp.zeros(lj_len)
        lj_param = lj_param.at[ttlj].set(config_dict["epsl_table"])
        config_dict["LJ_param"] = lj_param

    def _parse_pair_lj(entries: list) -> None:
        sgm_dict = {}
        epsl_dict = {}
        skipped_pairs = []
        provided_pairs = set()
        seen_types = np.zeros(n_types, dtype=bool)

        sgm = np.zeros((n_types, n_types))
        epsl = np.zeros((n_types, n_types))

        for c in entries:
            n0 = _normalize_type_name(c[0])
            n1 = _normalize_type_name(c[1])
            t0 = _resolve_type_name(n0)
            t1 = _resolve_type_name(n1)
            if t0 is None or t1 is None:
                skipped_pairs.append((n0, n1))
                continue

            # Canonicalize pair order so (A, B) and (B, A) map to the same entry.
            type_0, type_1 = (t0, t1) if t0 <= t1 else (t1, t0)

            sgm_dict[(type_0, type_1)] = c[2]
            epsl_dict[(type_0, type_1)] = c[3]
            provided_pairs.add((type_0, type_1))
            seen_types[type_0] = True
            seen_types[type_1] = True

            sgm[type_0, type_1] = c[2]
            epsl[type_0, type_1] = c[3]

        if skipped_pairs:
            preview = ", ".join([f"({a}, {b})" for a, b in skipped_pairs[:6]])
            if len(skipped_pairs) > 6:
                preview += ", ..."
            Logger.rank0.warning(
                "Ignoring LJ_param entries with unknown particle names not present in the current coordinates: "
                f"{preview}"
            )

        # Strict coverage check for atom types present in the input system.
        # Unknown/extraneous pairs are allowed (ignored above), but missing
        # in-system pairs must raise an error to prevent incorrect forces.
        required_pairs = {(i, j) for i in range(n_types) for j in range(i, n_types)}
        missing_pairs = sorted(required_pairs - provided_pairs)

        missing_types = [
            n for n, t in sorted(name_to_type.items(), key=lambda x: x[1]) if not seen_types[t]
        ]

        if missing_types or missing_pairs:
            err_lines = []
            if missing_types:
                err_lines.append(
                    "Missing LJ pair-table coverage for atom types present in the input coordinates: "
                    + ", ".join(missing_types)
                )

            if missing_pairs:
                type_to_name = {
                    t: n for n, t in sorted(name_to_type.items(), key=lambda x: x[1])
                }
                missing_named = [
                    (type_to_name[i], type_to_name[j]) for (i, j) in missing_pairs
                ]
                preview = ", ".join([f"({a}, {b})" for a, b in missing_named[:10]])
                if len(missing_named) > 10:
                    preview += ", ..."
                err_lines.append(
                    "Missing LJ pair entries for required in-system couples: " + preview
                )

            err_msg = "\n".join(err_lines)
            Logger.rank0.error(err_msg)
            print(err_msg, file=sys.stderr)
            exit(1)

        _set_lj_tables(sgm, epsl, sgm_dict, epsl_dict)

    def _parse_type_lj(entries: list, mixing_rule: str) -> None:
        sgm_type = np.zeros(n_types)
        epsl_type = np.zeros(n_types)
        seen_type = np.zeros(n_types, dtype=bool)
        skipped_types = []

        for row in entries:
            if len(row) < 3:
                Logger.rank0.error(
                    "LJ_type_param entries must be [type_name, sigma, epsilon]. "
                    f"Got: {row}."
                )
                exit()

            name = _normalize_type_name(row[0])
            t = _resolve_type_name(name)
            if t is None:
                skipped_types.append(name)
                continue

            sgm_type[t] = float(row[1])
            epsl_type[t] = float(row[2])
            seen_type[t] = True

        if skipped_types:
            preview = ", ".join(skipped_types[:8])
            if len(skipped_types) > 8:
                preview += ", ..."
            Logger.rank0.warning(
                "Ignoring LJ_type_param entries with unknown particle names not present in the current coordinates: "
                f"{preview}"
            )

        missing = [
            n
            for n, t in sorted(name_to_type.items(), key=lambda x: x[1])
            if not seen_type[t]
        ]
        if missing:
            Logger.rank0.error(
                "Missing LJ_type_param entries for particle types present in coordinates: "
                + ", ".join(missing)
            )
            exit()

        if mixing_rule not in ("lorentz-berthelot", "geometric"):
            Logger.rank0.error(
                "Type-based LJ input requires 'combining_rule' to be "
                "'lorentz-berthelot' or 'geometric'."
            )
            exit()

        sgm_dict = {}
        epsl_dict = {}
        sgm = np.zeros((n_types, n_types))
        epsl = np.zeros((n_types, n_types))

        for i in range(n_types):
            for j in range(i, n_types):
                if mixing_rule == "lorentz-berthelot":
                    sij = 0.5 * (sgm_type[i] + sgm_type[j])
                else:
                    sij = np.sqrt(sgm_type[i] * sgm_type[j])

                eij = np.sqrt(epsl_type[i] * epsl_type[j])

                sgm[i, j] = sij
                epsl[i, j] = eij
                sgm_dict[(i, j)] = sij
                epsl_dict[(i, j)] = eij

        _set_lj_tables(sgm, epsl, sgm_dict, epsl_dict)

    combining_rule_raw = str(config_dict.get("combining_rule", "pairtable")).lower()
    lj_input_source_raw = str(config_dict.get("lj_input_source", "auto")).lower()
    lj_input_alias = {
        "auto": "auto",
        "mixing": "mixing",
        "mixed": "mixing",
        "input": "input",
        "pair": "input",
        "pairtable": "input",
    }
    if lj_input_source_raw not in lj_input_alias:
        Logger.rank0.error(
            "Valid values for 'lj_input_source' are: ('auto', 'mixing', 'input'). "
            f"Got '{lj_input_source_raw}'."
        )
        exit()
    lj_input_source = lj_input_alias[lj_input_source_raw]
    config_dict["lj_input_source"] = lj_input_source

    config_dict["lj_force_shift"] = bool(config_dict.get("lj_force_shift", False))

    has_pair_lj_legacy = "LJ_param" in config_dict
    has_pair_lj_new = "LJ_pair_param" in config_dict
    has_type_lj = "LJ_type_param" in config_dict

    pair_entries = None
    if has_pair_lj_new and has_pair_lj_legacy:
        Logger.rank0.warning(
            "Both 'LJ_pair_param' and legacy 'LJ_param' are present. "
            "Using 'LJ_pair_param'."
        )
        pair_entries = config_dict["LJ_pair_param"]
    elif has_pair_lj_new:
        pair_entries = config_dict["LJ_pair_param"]
    elif has_pair_lj_legacy:
        pair_entries = config_dict["LJ_param"]

    has_pair_lj = pair_entries is not None

    if lj_input_source == "input":
        if not has_pair_lj:
            Logger.rank0.error(
                "'lj_input_source = input' requires 'LJ_pair_param' "
                "(or legacy 'LJ_param') in the options TOML."
            )
            exit()
        if has_type_lj:
            Logger.rank0.warning(
                "'lj_input_source = input' selected: ignoring 'LJ_type_param'."
            )
        _parse_pair_lj(pair_entries)
    elif lj_input_source == "mixing":
        if not has_type_lj:
            Logger.rank0.error(
                "'lj_input_source = mixing' requires 'LJ_type_param' in the options TOML."
            )
            exit()
        if has_pair_lj:
            Logger.rank0.warning(
                "'lj_input_source = mixing' selected: ignoring pair LJ table inputs "
                "('LJ_pair_param'/'LJ_param')."
            )
        if combining_rule_raw == "pairtable":
            Logger.rank0.warning(
                "'LJ_type_param' provided with combining_rule='pairtable'. "
                "Switching to 'lorentz-berthelot' for internal LJ mixing."
            )
            config_dict["combining_rule"] = "lorentz-berthelot"
            combining_rule_raw = "lorentz-berthelot"
        _parse_type_lj(config_dict["LJ_type_param"], combining_rule_raw)
    else:
        # auto mode: keep backward-compatible behavior.
        if has_pair_lj and has_type_lj:
            if combining_rule_raw == "pairtable":
                Logger.rank0.warning(
                    "Both pair LJ entries and 'LJ_type_param' are present. "
                    "Using pair entries because combining_rule='pairtable'."
                )
                _parse_pair_lj(pair_entries)
            else:
                Logger.rank0.warning(
                    "Both pair LJ entries and 'LJ_type_param' are present. "
                    "Using 'LJ_type_param' because combining_rule requests internal mixing."
                )
                _parse_type_lj(config_dict["LJ_type_param"], combining_rule_raw)
        elif has_type_lj:
            if combining_rule_raw == "pairtable":
                Logger.rank0.warning(
                    "'LJ_type_param' was provided with combining_rule='pairtable'. "
                    "Switching to 'lorentz-berthelot' for internal LJ mixing."
                )
                config_dict["combining_rule"] = "lorentz-berthelot"
                combining_rule_raw = "lorentz-berthelot"
            _parse_type_lj(config_dict["LJ_type_param"], combining_rule_raw)
        elif has_pair_lj:
            _parse_pair_lj(pair_entries)

    if "LJ_type_param" in config_dict:
        del config_dict["LJ_type_param"]
    if "LJ_pair_param" in config_dict:
        del config_dict["LJ_pair_param"]

    # --- Log LJ parameter table ---
    type_to_name = {t: n for n, t in sorted(name_to_type.items(), key=lambda x: x[1])}
    sgm_tbl = np.asarray(config_dict["sgm_table"])
    eps_tbl = np.asarray(config_dict["epsl_table"])
    lj_lines = [
        f"LJ parameters | combining_rule={combining_rule_raw} | "
        f"lj_input_source={lj_input_source} | n_types={n_types}",
        f"  {'type_i':>8s}  {'type_j':>8s}  {'id_i':>4s}  {'id_j':>4s}  "
        f"{'sigma_ij':>10s}  {'epsilon_ij':>10s}",
    ]
    for i in range(n_types):
        for j in range(i, n_types):
            ni = type_to_name.get(i, f"?{i}")
            nj = type_to_name.get(j, f"?{j}")
            lj_lines.append(
                f"  {ni:>8s}  {nj:>8s}  {i:4d}  {j:4d}  "
                f"{sgm_tbl[i, j]:10.6f}  {eps_tbl[i, j]:10.6f}"
            )
    Logger.rank0.info("\n".join(lj_lines))

    for k, v in config_dict.items():
        if k == "thermostat_coupling_groups":
            group_coverage = np.zeros(len(names), dtype=np.int32)
            group_masks = []
            for i, name_list in enumerate(v):
                name_list = np.array([n.encode("UTF-8") for n in name_list])
                group = np.where(np.isin(names, name_list), 1, 0)
                group_n_particles = int(np.sum(group))
                group_coverage += group

                if group_n_particles == 0:
                    err_str = (
                        "A thermostat coupling group matched zero atoms. "
                        f"Group index: {i}, names: {list(v[i])}."
                    )
                    Logger.rank0.error(err_str)
                    exit()

                group_masks.append(
                    ThermostatGroup(
                        group_n_particles,
                        jnp.asarray(group, dtype=jnp.int32).reshape(-1, 1),
                    )
                )

            if np.any(group_coverage == 0):
                n_missing = int(np.sum(group_coverage == 0))
                err_str = (
                    "Thermostat coupling groups do not cover all atoms. "
                    f"Missing atoms: {n_missing}."
                )
                Logger.rank0.error(err_str)
                exit()

            config_dict["thermostat_coupling_groups"] = tuple(group_masks)

            if np.any(group_coverage > 1):
                n_overlap = int(np.sum(group_coverage > 1))
                err_str = (
                    "Thermostat coupling groups overlap. "
                    f"Overlapping atoms: {n_overlap}."
                )
                Logger.rank0.error(err_str)
                exit()

        if k == "target_pressure":
            if isinstance(v, list):
                if len(v) == 3:
                    config_dict["target_pressure"] = jnp.array(v)
                elif len(v) == 2:
                    config_dict["target_pressure"] = jnp.array([v[0], v[0], v[1]])
                elif len(v) == 1:
                    config_dict["target_pressure"] = jnp.array(3 * [v[0]])
            elif isinstance(v, int) or isinstance(v, float):
                config_dict["target_pressure"] = jnp.array(3 * [v])

    for n in ("n_steps", "time_step", "mesh_size", "sigma", "kappa"):
        if n not in config_dict:
            err_str = (
                f"No '{n}' specified in config file '{file_path}'."
                f"Unable to start simulation."
            )
            Logger.rank0.error(err_str)
            exit()

    # set some defaults
    if "thermostat_coupling_groups" not in config_dict:
        # If no thermostat_coupling_groups are specified in the input,
        # the whole system is coupled together
        config_dict["thermostat_coupling_groups"] = (
            ThermostatGroup(
                n_particles,
                jnp.ones((n_particles, 1), dtype=jnp.int32),
            ),
        )

    # TODO: remove this backward compatibility when it's time
    if "tau" in config_dict:
        config_dict["tau_t"] = config_dict.pop("tau")

    if "rho0" not in config_dict:
        config_dict["rho0"] = n_particles / total_volume

    if "a" not in config_dict:
        config_dict["a"] = n_particles / total_volume

    # Force-field defaults
    if "ff_family" not in config_dict:
        config_dict["ff_family"] = config_dict.get("family", "martini")
    if "combining_rule" not in config_dict:
        config_dict["combining_rule"] = "pairtable"
    if "coulomb14_scale" not in config_dict:
        config_dict["coulomb14_scale"] = 1.0
    if "lj14_scale" not in config_dict:
        config_dict["lj14_scale"] = 1.0
    if "constrain_xh_bonds" not in config_dict:
        config_dict["constrain_xh_bonds"] = False
    if "unwrap_output" not in config_dict:
        config_dict["unwrap_output"] = True
    if "ensemble" not in config_dict:
        config_dict["ensemble"] = "NVT"
    if "thermostat" not in config_dict:
        # Keep previous behavior by defaulting to CSVR when a target temperature is set.
        if config_dict.get("target_temperature"):
            config_dict["thermostat"] = "v-rescale"
        else:
            config_dict["thermostat"] = "no"

    # PME/Ewald defaults
    if "ewald_rtol" not in config_dict:
        config_dict["ewald_rtol"] = 1e-5
    if "pme_order" not in config_dict:
        config_dict["pme_order"] = 2  # default CIC

    # Resolve sigma: numeric or "auto"
    try:
        config_dict["rc"] = float(config_dict["rc"])
    except Exception as e:
        Logger.rank0.error("'rc' must be a numeric value.", exc_info=e)
        exit()

    if config_dict["rc"] <= 0:
        Logger.rank0.error("'rc' must be > 0.")
        exit()

    try:
        config_dict["ewald_rtol"] = float(config_dict["ewald_rtol"])
    except Exception as e:
        Logger.rank0.error("'ewald_rtol' must be a numeric value in (0, 1).", exc_info=e)
        exit()

    if not (0.0 < config_dict["ewald_rtol"] < 1.0):
        Logger.rank0.error("'ewald_rtol' must be in (0, 1).")
        exit()

    sigma_raw = config_dict["sigma"]
    sigma_auto = isinstance(sigma_raw, str) and sigma_raw.strip().lower() == "auto"
    if sigma_auto:
        from scipy.special import erfcinv as _erfcinv

        x_target = float(_erfcinv(config_dict["ewald_rtol"]))
        sigma_opt = config_dict["rc"] / (x_target * np.sqrt(2.0))
        config_dict["sigma"] = float(sigma_opt)
        Logger.rank0.info(
            f"PME sigma auto-tuning enabled: rc={config_dict['rc']:.6g}, "
            f"ewald_rtol={config_dict['ewald_rtol']:.2e} -> "
            f"sigma={config_dict['sigma']:.6g}"
        )
    else:
        try:
            config_dict["sigma"] = float(sigma_raw)
        except Exception as e:
            Logger.rank0.error(
                "'sigma' must be a numeric value or the string 'auto'.",
                exc_info=e,
            )
            exit()

    if config_dict["sigma"] <= 0:
        Logger.rank0.error("'sigma' must be > 0.")
        exit()

    ff_family = str(config_dict["ff_family"]).lower()
    valid_ff = ("martini", "amber_like", "amber19sb")
    if ff_family not in valid_ff:
        err_str = f"Valid force-field families are: {valid_ff}. Got '{ff_family}'."
        Logger.rank0.error(err_str)
        exit()
    config_dict["ff_family"] = ff_family

    combining_rule = str(config_dict["combining_rule"]).lower()
    valid_combining = ("pairtable", "lorentz-berthelot", "geometric")
    if combining_rule not in valid_combining:
        err_str = f"Valid combining rules are: {valid_combining}. Got '{combining_rule}'."
        Logger.rank0.error(err_str)
        exit()
    config_dict["combining_rule"] = combining_rule

    config_dict["coulomb14_scale"] = float(config_dict["coulomb14_scale"])
    config_dict["lj14_scale"] = float(config_dict["lj14_scale"])
    if not (0.0 <= config_dict["coulomb14_scale"] <= 1.0):
        err_str = "'coulomb14_scale' must be in [0, 1]."
        Logger.rank0.error(err_str)
        exit()
    if not (0.0 <= config_dict["lj14_scale"] <= 1.0):
        err_str = "'lj14_scale' must be in [0, 1]."
        Logger.rank0.error(err_str)
        exit()

    if isinstance(config_dict["constrain_xh_bonds"], str):
        config_dict["constrain_xh_bonds"] = (
            config_dict["constrain_xh_bonds"].lower() in ("1", "true", "yes", "on")
        )
    else:
        config_dict["constrain_xh_bonds"] = bool(config_dict["constrain_xh_bonds"])

    if isinstance(config_dict["unwrap_output"], str):
        config_dict["unwrap_output"] = (
            config_dict["unwrap_output"].lower() in ("1", "true", "yes", "on")
        )
    else:
        config_dict["unwrap_output"] = bool(config_dict["unwrap_output"])

    config_dict["ensemble"] = str(config_dict["ensemble"]).upper()
    if config_dict["ensemble"] not in ("NVT", "NVE", "NPT"):
        err_str = "Valid ensembles are: 'NVT', 'NVE', 'NPT'."
        Logger.rank0.error(err_str)
        exit()

    # ---- NVE enforcement ----
    if config_dict["ensemble"] == "NVE":
        # Thermostat must be off
        thermo_raw = str(config_dict.get("thermostat", "no")).lower()
        if thermo_raw not in ("no", "none", "off"):
            Logger.rank0.warning(
                f"ensemble = 'NVE' but thermostat = '{config_dict['thermostat']}'. "
                "Forcing thermostat = 'no'."
            )
        config_dict["thermostat"] = "no"

        # Barostat must be off
        baro_raw = str(config_dict.get("barostat", "no")).lower()
        if baro_raw not in ("no", "0"):
            Logger.rank0.warning(
                f"ensemble = 'NVE' but barostat = '{config_dict['barostat']}'. "
                "Forcing barostat = 'no'."
            )
        config_dict["barostat"] = 0
        config_dict["barostat_type"] = None
        config_dict["pressure"] = False


    thermostat_alias = {
        "v-rescale": "v-rescale",
        "vrescale": "v-rescale",
        "csvr": "v-rescale",
        "bussi": "v-rescale",
        "no": "no",
        "none": "no",
        "off": "no",
    }
    thermostat_raw = str(config_dict["thermostat"]).lower()
    if thermostat_raw not in thermostat_alias:
        err_str = (
            "Valid thermostats are: 'v-rescale' (alias: 'csvr') or 'no'. "
            f"Got '{config_dict['thermostat']}'."
        )
        Logger.rank0.error(err_str)
        exit()
    config_dict["thermostat"] = thermostat_alias[thermostat_raw]

    if config_dict["thermostat"] != "no":
        if not config_dict.get("target_temperature"):
            err_str = (
                "A thermostat is enabled but 'target_temperature' is missing or zero. "
                "Set 'target_temperature' or use thermostat='no'."
            )
            Logger.rank0.error(err_str)
            exit()

        if config_dict["thermostat"] == "v-rescale":
            if "tau_t" not in config_dict or config_dict["tau_t"] is None:
                err_str = (
                    "thermostat='v-rescale' requires 'tau' (or 'tau_t') in the input TOML."
                )
                Logger.rank0.error(err_str)
                exit()
            if config_dict["tau_t"] <= 0:
                err_str = "'tau' (or 'tau_t') must be > 0 for thermostat='v-rescale'."
                Logger.rank0.error(err_str)
                exit()

    config_dict["inner_ts"] = config_dict.pop("time_step")
    config_dict["outer_ts"] = config_dict["inner_ts"] * config_dict["respa_inner"]

    for n in ("box_size", "mesh_size"):
        config_dict[n] = jnp.array(config_dict[n])

    if "barostat" in config_dict:
        bval = config_dict["barostat"]
        # Already resolved to int by NVE/NVT enforcement block — skip parsing
        if not isinstance(bval, int):
            barostat_name = {"no": 0, "berendsen": 1, "scr": 2}
            barostat_type = {"isotropic": 1, "semiisotropic": 2, "surface_tension": 3}
            bname = bval.lower()

            if bname not in barostat_name.keys():
                err_str = f"Valid barostats are: 'no', 'berendsen', 'scr'. Got '{bname}'."
                Logger.rank0.error(err_str)
                exit()
            else:
                config_dict["barostat"] = barostat_name[bname]
                if config_dict["barostat"] == 0:
                    config_dict["barostat_type"] = None
                else:
                    if "barostat_type" not in config_dict:
                        err_str = "'barostat_type' is required when barostat is not 'no'."
                        Logger.rank0.error(err_str)
                        exit()

                    btype = config_dict["barostat_type"].lower()
                    if btype not in barostat_type.keys():
                        err_str = f"Valid barostat types are: 'isotropic', 'semiisotropic', 'surface_tension'. Got '{btype}'."
                        Logger.rank0.error(err_str)
                        exit()
                    config_dict["barostat_type"] = barostat_type[btype]
                    if config_dict["barostat"] == 1 and config_dict["barostat_type"] == 3:
                        err_str = "Barostat type 'surface_tension' is currently only available with the 'scr' barostat."
                        Logger.rank0.error(err_str)
                        exit()

    if config_dict["ensemble"] in ("NVT", "NVE"):
        # Warn if barostat-related keys are set but barostat itself is
        # missing/disabled — this is almost always a TOML mistake.
        # Check BEFORE overwriting keys with defaults.
        _baro_keys = {"barostat_type", "tau_p", "target_pressure", "beta"}
        _orphan = _baro_keys & set(config_dict.keys())
        if "barostat" not in config_dict and _orphan:
            Logger.rank0.warning(
                f"Barostat settings {sorted(_orphan)} are present in the TOML "
                f"but ensemble = '{config_dict['ensemble']}' and no "
                f"'barostat' flag was set.  These settings will be IGNORED.  "
                f"If you intended NPT, set  ensemble = 'NPT'  and  "
                f"barostat = 'berendsen'  (or 'scr')  in your TOML."
            )

        config_dict["pressure"] = False
        config_dict["barostat"] = 0
        config_dict["barostat_type"] = None

    # ---- NPT validation ----
    if config_dict["ensemble"] == "NPT":
        config_dict["pressure"] = True
        if not isinstance(config_dict.get("barostat"), int) or config_dict["barostat"] == 0:
            err_str = (
                "ensemble = 'NPT' requires a barostat. "
                "Set barostat = 'berendsen' or 'scr' in the input TOML."
            )
            Logger.rank0.error(err_str)
            exit()
        if config_dict.get("tau_p") is None or config_dict["tau_p"] <= 0:
            err_str = "'tau_p' must be > 0 for NPT ensemble."
            Logger.rank0.error(err_str)
            exit()
        tp = config_dict.get("target_pressure")
        if tp is None:
            err_str = "'target_pressure' is required for NPT ensemble."
            Logger.rank0.error(err_str)
            exit()
        # Ensure target_pressure is a jnp array of the right shape
        tp = jnp.atleast_1d(jnp.asarray(tp, dtype=jnp.float64))
        if config_dict["barostat_type"] == 1:
            # Isotropic: scalar or (3,) all equal
            if tp.size == 1:
                tp = jnp.broadcast_to(tp, (3,))
            config_dict["target_pressure"] = tp
        elif config_dict["barostat_type"] in (2, 3):
            # Semi-isotropic / surface tension: need at least [Pxy, Pz]
            if tp.size == 2:
                tp = jnp.array([tp[0], tp[0], tp[1]])
            config_dict["target_pressure"] = tp
    
    # ── coulombtype string → int conversion (must happen unconditionally) ──
    # The TOML always stores a string ("no", "pme", "reaction-field").
    # Convert to int here so all downstream code sees an integer, regardless
    # of whether the system has charges.
    _coulomb_map = {"no": 0, "pme": 1, "reaction-field": 2}
    ctype = config_dict.get("coulombtype", "no")
    if isinstance(ctype, str):
        if ctype not in _coulomb_map:
            err_str = f"Valid electrostatic interaction options are: 'no', 'pme', 'reaction-field'. Got '{ctype}'."
            Logger.rank0.error(err_str)
            exit()
        config_dict["coulombtype"] = _coulomb_map[ctype]

    if charges is None:
        # No charges → force electrostatics off, regardless of TOML setting
        if config_dict["coulombtype"] != 0:
            Logger.rank0.info(
                f"No charges in system — overriding coulombtype "
                f"from {config_dict['coulombtype']} to 0 (none)."
            )
            config_dict["coulombtype"] = 0

    if charges is not None:
        # Use float64 for the summation to avoid accumulated float32 rounding
        # errors on large systems (e.g. 22k atoms can drift ~1e-6).
        tot_charge = float(np.sum(np.asarray(charges, dtype=np.float64)))
        if not np.isclose(tot_charge, 0.0, atol=1e-4):
            err_str = f"The sum of all charges should be equal to zero to avoid artifacts. Got {tot_charge}."
            Logger.rank0.error(err_str)
            exit()

        if config_dict["coulombtype"] == 0:
            err_str = "Charged particles are present in the system but coulombtype = no. Electrostatic interactions will not be calculated."
            Logger.rank0.info(err_str)

        # --- Validate Ewald sigma vs rc for PME ---
        if config_dict["coulombtype"] == 1:
            # Validate pme_order
            _pme_order = int(config_dict["pme_order"])
            if _pme_order < 2 or _pme_order > 8:
                Logger.rank0.error(
                    f"pme_order must be between 2 and 8 (got {_pme_order})."
                )
                exit()
            config_dict["pme_order"] = _pme_order

            _sigma = float(config_dict["sigma"])
            _rc = float(config_dict["rc"])
            _alpha = 1.0 / (2.0 * _sigma * _sigma)
            _beta = np.sqrt(_alpha)
            from scipy.special import erfc as _erfc, erfcinv as _erfcinv
            _rtol = float(config_dict["ewald_rtol"])

            erfc_at_rc = float(_erfc(_beta * _rc))
            if erfc_at_rc > 1e-3:
                # Compute recommended sigma for configured tolerance
                x_target = float(_erfcinv(_rtol))
                sigma_rec = _rc / (x_target * np.sqrt(2.0))
                Logger.rank0.warning(
                    f"Ewald real-space truncation error is large: "
                    f"erfc(beta*rc) = erfc({_beta * _rc:.3f}) = {erfc_at_rc:.4e} "
                    f"(sigma={_sigma}, rc={_rc}). "
                    f"This means ~{erfc_at_rc*100:.1f}% of the Coulomb interaction "
                    f"leaks past the cutoff, causing inaccurate electrostatic energies "
                    f"in dense systems. "
                    f"Recommended sigma for rc={_rc}: {sigma_rec:.4f} nm "
                    f"(target ewald_rtol={_rtol:.2e}). "
                    f"Set sigma = {sigma_rec:.4f} in your options.toml [field] section."
                )

    # if charges is not None:
    #     if not jnp.isclose(tot_charge := jnp.sum(charges), 0):
    #         err_str = f"The sum of all charges should be equal to zero to avoid artifacts. Got {tot_charge}."
    #         Logger.rank0.error(err_str)
    #         exit()
    #     config_dict["coulombtype"] = 1
    # else:
    #     config_dict["coulombtype"] = 0

    # HyMD options not used in Diff-MD
    for opt in ("integrator", "hamiltonian"):
        if opt in config_dict:
            del config_dict[opt]

    # Reassingn types to the correct names in shared Chi matrix
    # when training multiple systems
    if ext_name_to_type is not None:
        # Build a lookup that also resolves _spc <-> _sp aliases
        _ext_lookup = dict(ext_name_to_type)
        for _name, _tid in ext_name_to_type.items():
            if _name.endswith("_spc"):
                _ext_lookup.setdefault(_name[:-1], _tid)
            elif _name.endswith("_sp"):
                _ext_lookup.setdefault(_name + "c", _tid)
        types = np.array([_ext_lookup[n.decode("UTF-8")] for n in names])
        unique_types = types[np.sort(name_idx)]
        config_dict["particle_per_type"] = {
            t: len(types[types == t]) for t in unique_types
        }

        

    config_dict["unique_types"] = tuple(unique_types)

    # Optional CMAP backbone-correction grids — only meaningful for
    # ``ff_family == "amber19sb"``.  Parsed once here at config-load
    # time so the per-residue 24x24 grids ride along on the (static)
    # ``Config.cmap_grid_bank`` field; the per-cell bicubic
    # coefficients are then materialised by ``topology.py`` for the
    # specific chains that actually have ``[cmap]`` entries.
    cmap_bank = parse_cmap_section(toml_config)
    if cmap_bank is not None and ff_family != "amber19sb":
        Logger.rank0.warning(
            f"Found a [cmap] block in '{file_path}' but ff_family='{ff_family}'. "
            "CMAP corrections will be ignored (set ff_family='amber19sb' to enable)."
        )
        cmap_bank = None
    if cmap_bank is None and ff_family == "amber19sb":
        Logger.rank0.warning(
            f"ff_family='amber19sb' but '{file_path}' has no [cmap] block. "
            "Backbone CMAP correction will be disabled."
        )
    config_dict["cmap_grid_bank"] = cmap_bank
    config = Config.constructor(types, charges, **config_dict)
    Logger.rank0.info(
        f"Config file '{file_path}' parse successfully.",
    )

    # Only print if we have a single system
    if name_to_type is None:
        Logger.rank0.info(str(config))
    

    return config, jnp.array(types)


# @jit
def update_traj(traj, box_size, frame, shifts):
    # Generate all periodic images
    images = (shifts * box_size[frame]) + jnp.expand_dims(traj[frame], 1)
    
    # Calculate displacements
    disp = images - jnp.expand_dims(traj[frame-1], 1)
    
    # Calculate squared distances (avoid sqrt for performance)
    dist_sq = jnp.sum(disp**2, axis=2)
    
    # Find the image with minimum distance for each atom
    min_indices = jnp.argmin(dist_sq, axis=1)
    
    # Select the minimum displacement image for each atom
    new_positions = images[jnp.arange(len(images)), min_indices]
    
    # Update positions
    traj = traj.at[frame].set(new_positions)
    
    return traj


def unwrap(traj, box_size):

    shifts = jnp.array([
        [x, y, z] for x in [-1, 0, 1] 
                  for y in [-1, 0, 1] 
                  for z in [-1, 0, 1]
    ])
    
    for frame, _ in enumerate(traj):
        if frame == 0: 
            continue
        
        traj = update_traj(traj, box_size, frame, shifts)

        # images = (shifts * box_size[frame]) + jnp.expand_dims(traj[frame], 1)
        # disp = images - jnp.expand_dims(traj[frame-1], 1)
        # dist_sq = jnp.sum(disp**2, axis=2)
        # min_indices = jnp.argmin(dist_sq, axis=1)
        # new_positions = images[jnp.arange(len(images)), min_indices]
        # traj = traj.at[frame].set(new_positions)
    
    return traj


def center_molecule(traj, box_size, chain_indices):
    traj = jnp.asarray(traj)
    box_size = jnp.asarray(box_size)
    
    traj = unwrap(traj, box_size)

    for frame, _ in enumerate(traj):
        # Calculate center of geometry
        cog = jnp.mean(traj[frame][chain_indices], axis=1)
        
        # Calculate shift needed to center COG in box
        box_center = box_size[frame] / 2
        shift_to_center = box_center - cog
        
        # Apply shift to center the molecule
        traj = traj.at[frame].add(shift_to_center)
        
        # WRAP BACK INTO THE BOX
        traj = traj.at[frame].set(jnp.mod(traj[frame], box_size[frame]))
    
    return traj  


