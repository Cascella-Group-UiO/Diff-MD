from typing import Any, Optional, Self, Tuple, Union

import jax.numpy as jnp
import numpy as np
import tomli
from flax import struct
from jax import Array, jit
from jax.typing import ArrayLike

from .logger import Logger


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
    range_types: ArrayLike = struct.field(pytree_node=False)
    unique_types: tuple = struct.field(pytree_node=False)
    type_to_charge_map: dict = struct.field(pytree_node=False)
    inner_ts: float = struct.field(pytree_node=False)
    outer_ts: float = struct.field(pytree_node=False)
    empty_mesh: Array = struct.field(pytree_node=False)
    n_mesh_cells: ArrayLike = struct.field(pytree_node=False)
    fft_shape: tuple = struct.field(pytree_node=False)
    # window: ArrayLike = struct.field(pytree_node=False)
    elec_const: float = struct.field(pytree_node=False)
    self_energy: float = struct.field(pytree_node=False)

    # Field options
    # chi: Array = struct.field(pytree_node=False)
    # chi_dict: dict = struct.field(pytree_node=False)
    # type_to_chi: Array = struct.field(pytree_node=False)
    mesh_size: Array = struct.field(pytree_node=False)
    rho0: float = struct.field(pytree_node=False)
    kappa: float = struct.field(pytree_node=False)
    sigma: float = struct.field(pytree_node=False) 

    # LJ options
    LJ_param: Array = struct.field(pytree_node=False)
    sgm_dict: dict = struct.field(pytree_node=False)
    epsl_dict: dict = struct.field(pytree_node=False)
    type_to_LJ: Array = struct.field(pytree_node=False)    
    sgm_table: Array = struct.field(pytree_node=False)
    epsl_table: Array = struct.field(pytree_node=False)

    rv: float = struct.field(pytree_node=False)
    rc: float = struct.field(pytree_node=False)
    rlj: float = struct.field(pytree_node=False)
    ns_nlist: int = struct.field(pytree_node=False)
    nrexcl: int = struct.field(pytree_node=False)
    epsilon_rf: float = struct.field(pytree_node=False)

    thermostat_coupling_groups: tuple[int, ArrayLike] = struct.field(pytree_node=False)

    respa_inner: int = struct.field(pytree_node=False, default=1)
    n_print: int = struct.field(pytree_node=False, default=None)
    n_flush: int = struct.field(pytree_node=False, default=None)

    # Gas constant (in kJ/mol K)
    R: float = struct.field(pytree_node=False, default=0.00831446261815324)
    mass: Union[float, ArrayLike] = struct.field(pytree_node=False, default=72.0)
    cancel_com_momentum: Union[int, bool] = struct.field(
        pytree_node=False, default=False
    )
    coulombtype: int = struct.field(pytree_node=False, default=0)
    elec_conversion: float = struct.field(
        pytree_node=False, default=(138.935458)
    )

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
    target_pressure: ArrayLike = struct.field(pytree_node=False, default=None)

    # Pressure conversion constant
    p_conv: float = struct.field(pytree_node=False, default=16.605390666)

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
            if coulombtype == 1:
                prefac = elec_conversion * jnp.sqrt(1.0 / (2.0 * jnp.pi * sigma * sigma))
                self_energy = prefac * jnp.sum(charges * charges)
            elif coulombtype == 2:
                prefac = elec_conversion * (3 * erf / (2 * erf + er)) / (2 * rc)
                self_energy = prefac * 0.5 * (jnp.sum(charges * charges) - 1/erf * jnp.sum(charges) ** 2)
            return self_energy   

        step = args["box_size"] / (2 * jnp.pi * args["mesh_size"])
        total_volume = jnp.prod(args["box_size"])
        n_mesh_cells = jnp.prod(args["mesh_size"])
        n_particles = len(types)

        kx = jnp.fft.fftfreq(args["mesh_size"][0], step[0])
        ky = jnp.fft.fftfreq(args["mesh_size"][1], step[1])
        kz = jnp.fft.rfftfreq(args["mesh_size"][2], step[2])
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
        empty_mesh = jnp.zeros(args["mesh_size"])

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
            empty_mesh=empty_mesh,
            elec_const=elec_const,
            self_energy=self_energy,
            type_to_charge_map=type_to_charge_map,
            n_mesh_cells=n_mesh_cells,
            **args,
        )
    
    # Doesn't seem important right now
    #
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
            ):
                ret_str += f"\t{k}: {v}\n"
        ret_str += "\tLJ parameters:\n"
        # for k, v in self.sgm_dict.items(): 
        #     ret_str += f"\t\t{k[0]}\t{k[1]}\t{v:>8}\n"
        for k, v in zip(self.sgm_dict.items(), self.epsl_dict.items()):
            ret_str += f"\t\t{k[0][0]}\t{k[0][1]}\t{k[1]}\t{v[1]}\n"
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
        db_chi = db_data["nn"]["model"].pop("chi")

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

    config_dict["box_size"] = box_size
    config_dict["range_types"] = jnp.arange(n_types)
    config_dict["n_types"] = n_types
    config_dict["particle_per_type"] = particle_per_type
    config_dict["sgm_dict"] = {}
    config_dict["epsl_dict"] = {}
    config_dict["sgm_table"] = {}
    config_dict["epsl_table"] = {}
    config_dict["type_to_LJ"] = get_type_to_LJ(n_types)

    config_dict["dielectric_const"] = 1.0  # default

    if len(jnp.unique(masses)) > 1:
        config_dict["mass"] = jnp.reshape(masses, (-1, 1))
        # config_dict["mass"] = masses

    for k, v in toml_config.items():
        if isinstance(v, dict):
            if k in ("simulation", "field"):
                for nested_k, nested_v in v.items():
                    config_dict[nested_k] = nested_v
        else:
            config_dict[k] = v

    for k, v in config_dict.items():
        if k == "LJ_param": 
            sgm_dict = {}
            epsl_dict = {}

            sgm = np.zeros((n_types, n_types))
            epsl = np.zeros((n_types, n_types))
            LJ_param = jnp.zeros(len(v))

            for i, c in enumerate(v):
                type_0, type_1 = (name_to_type[c[0]], name_to_type[c[1]])
                
                # This will probably be used in training
                val = c[2]
                if database is not None:
                    for dbc in db_chi:
                        if c[:2] == dbc[:2]:
                            val = dbc[2]
                            break
                        
                sgm_dict[(type_0, type_1)] = c[2]    # Stil don't know where this is used. Will leave it here anyway
                epsl_dict[(type_0, type_1)] = c[3]   

                sgm[type_0, type_1] = c[2]    
                epsl[type_0, type_1] = c[3]         

                LJ_param = LJ_param.at[i].set(c[3])  # TODO: Check if this is not dangerous
            
            config_dict["sgm_dict"] = sgm_dict
            config_dict["epsl_dict"] = epsl_dict
            
            config_dict["epsl_table"] = jnp.array(epsl + epsl.T - np.diag(np.diag(epsl)))
            config_dict["sgm_table"] = jnp.array(sgm + sgm.T - np.diag(np.diag(sgm)))

            config_dict["LJ_param"] = LJ_param

        if k == "thermostat_coupling_groups":
            for i, name_list in enumerate(v):
                name_list = np.array([n.encode("UTF-8") for n in name_list])
                group = np.where(np.isin(names, name_list), 1, 0)
                group_n_particles = np.sum(group)

                config_dict["thermostat_coupling_groups"][i] = (
                    group_n_particles,
                    jnp.array(group).reshape(-1, 1),
                )

        if k == "target_pressure":
            if isinstance(v, list):
                if len(v) == 2:
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
        config_dict["thermostat_coupling_groups"] = [
            (
                n_particles,
                jnp.ones((n_particles, 1)),
            )
        ]

    # TODO: remove this backward compatibility when it's time
    if "tau" in config_dict:
        config_dict["tau_t"] = config_dict.pop("tau")

    if "rho0" not in config_dict:
        config_dict["rho0"] = n_particles / total_volume

    if "a" not in config_dict:
        config_dict["a"] = n_particles / total_volume

    config_dict["inner_ts"] = config_dict.pop("time_step")
    config_dict["outer_ts"] = config_dict["inner_ts"] * config_dict["respa_inner"]

    for n in ("box_size", "mesh_size"):
        config_dict[n] = jnp.array(config_dict[n])

    if "barostat" in config_dict:
        barostat_name = {"no": 0, "berendsen": 1, "scr": 2}
        barostat_type = {"isotropic": 1, "semiisotropic": 2, "surface_tension": 3}
        bname = config_dict["barostat"].lower()
        btype = config_dict["barostat_type"].lower()

        if bname not in barostat_name.keys():
            err_str = f"Valid barostats are: 'no', 'berendsen', 'scr'. Got '{bname}'."
            Logger.rank0.error(err_str)
            exit()
        else:
            if btype not in barostat_type.keys():
                err_str = f"Valid barostat types are: 'isotropic', 'semiisotropic', 'surface_tension'. Got '{bname}'."
                Logger.rank0.error(err_str)
                exit()
            config_dict["barostat"] = barostat_name[bname]
            config_dict["barostat_type"] = barostat_type[btype]
            if config_dict["barostat"] == 1 and config_dict["barostat_type"] == 3:
                err_str = "Barostat type 'surface_tension' is currently only available with the 'scr' barostat."
                Logger.rank0.error(err_str)
                exit()
    
    if charges is not None:
        if not jnp.isclose(tot_charge := jnp.sum(charges), 0):
            err_str = f"The sum of all charges should be equal to zero to avoid artifacts. Got {tot_charge}."
            Logger.rank0.error(err_str)
            exit()
        
        coulombtype = {"no": 0, "pme": 1, "reaction-field": 2}
        ctype = config_dict["coulombtype"]

        if ctype not in coulombtype.keys():
            err_str = f"Valid electrostatic interaction options are: 'no', 'pme', 'reaction-field'. Got '{ctype}'."
            Logger.rank0.error(err_str)
            exit()
        else:
            config_dict["coulombtype"] = coulombtype[ctype]

        if config_dict["coulombtype"] == 0:
            err_str = "Charged particles are present in the system but coulombtype = no. Electrostatic interactions will not be calculated."
            Logger.rank0.error(err_str)

    if charges is not None:
        if not jnp.isclose(tot_charge := jnp.sum(charges), 0):
            err_str = f"The sum of all charges should be equal to zero to avoid artifacts. Got {tot_charge}."
            Logger.rank0.error(err_str)
            exit()
        config_dict["coulombtype"] = 1
    else:
        config_dict["coulombtype"] = 0

    # HyMD options not used in Diff-MD
    for opt in ("integrator", "hamiltonian"):
        if opt in config_dict:
            del config_dict[opt]

    # Reassingn types to the correct names in shared Chi matrix
    # when training multiple systems
    if ext_name_to_type is not None:
        types = np.array([ext_name_to_type[n.decode("UTF-8")] for n in names])
        unique_types = types[np.sort(name_idx)]
        config_dict["particle_per_type"] = {
            t: len(types[types == t]) for t in unique_types
        }

    config_dict["unique_types"] = tuple(unique_types)
    config = Config.constructor(types, charges, **config_dict)
    Logger.rank0.info(
        f"Config file '{file_path}' parse successfully.",
    )

    # Only print if we have a single system
    if name_to_type is None:
        Logger.rank0.info(str(config))

    return config, jnp.array(types)
