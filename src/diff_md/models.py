"""Classes used to learn model parameters through MD trajectories"""

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_payload(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_payload(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class StaticArray:
    payload: Any
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_value(cls, value: Any) -> "StaticArray":
        arr = np.asarray(value)
        payload = arr.item() if arr.ndim == 0 else _freeze_payload(arr.tolist())
        return cls(payload=payload, shape=tuple(arr.shape), dtype=arr.dtype.name)

    def __array__(self, dtype=None):
        arr = np.asarray(self.payload, dtype=self.dtype)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __jax_array__(self):
        return jnp.asarray(np.asarray(self.payload, dtype=self.dtype))

    @property
    def at(self):
        return jnp.asarray(self).at

    def __getitem__(self, item):
        return np.asarray(self)[item]

    def __len__(self):
        return self.shape[0]


@struct.dataclass
class GeneralModel:
    n_types: int = struct.field(pytree_node=False, default=1)
    type_to_LJ: Optional[StaticArray] = struct.field(pytree_node=False, default=None) # Related to epsl only
    self_interaction: bool = struct.field(pytree_node=False, default=True)
    lj_mode: str = struct.field(pytree_node=False, default="pair")
    lj_name_to_type: dict = struct.field(pytree_node=False, default_factory=dict)
    lj_sigma_idx: Optional[StaticArray] = struct.field(pytree_node=False, default=None)
    lj_epsilon_idx: Optional[StaticArray] = struct.field(pytree_node=False, default=None)
    lj_sigma_ref: Optional[StaticArray] = struct.field(pytree_node=False, default=None)
    lj_epsilon_ref: Optional[StaticArray] = struct.field(pytree_node=False, default=None)
    n_sigma_train: int = struct.field(pytree_node=False, default=0)
    n_epsilon_train: int = struct.field(pytree_node=False, default=0)
    epsl_constraints: dict = struct.field(pytree_node=False, default_factory=dict)
    LJ_param: Optional[Array] = None
    bonds: Optional[dict] = None  # -> bonds_2
    angles: Optional[dict] = None  # -> bonds_3
    dihedrals: Optional[dict] = None  # -> bonds_4

    def __post_init__(self):
        for field_name in (
            "type_to_LJ",
            "lj_sigma_idx",
            "lj_epsilon_idx",
            "lj_sigma_ref",
            "lj_epsilon_ref",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, StaticArray):
                object.__setattr__(self, field_name, StaticArray.from_value(value))

    def __str__(self):
        ret_str = ""
        if self.LJ_param is not None:
            ret_str += f"LJ parameters:\t{self.LJ_param}\n"

        # if self.chi_constraints:
        #     ret_str += f"\t\tchi_constraints:\t{self.chi_constraints}\n"

        # NOTE: not implemented
        if self.bonds:
            pass
        if self.angles:
            pass
        if self.dihedrals:
            pass

        return ret_str  
    

@struct.dataclass
class ChiModel:
    type_to_chi: StaticArray = struct.field(pytree_node=False)
    self_interaction: bool = struct.field(pytree_node=False, default=False)
    chi: Optional[Array] = None

    def __post_init__(self):
        if not isinstance(self.type_to_chi, StaticArray):
            object.__setattr__(self, "type_to_chi", StaticArray.from_value(self.type_to_chi))
