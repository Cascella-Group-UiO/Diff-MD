import ctypes
import gc
import os
import random
import re
import subprocess
from typing import Any, Tuple
import copy

import jax
import jax.random
import mpi4jax
import numpy as onp
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from jax import value_and_grad, debug


def _malloc_trim():
    """Force glibc to return free'd arena memory to the OS.

    This only reclaims unused heap pages. It does not evict compiled
    XLA executables or touch JAX compilation caches.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
import jax.numpy as jnp
from mpi4py import MPI

from .file_io import OutDataset, save_params, store_static, write_full_trajectory, write_energy_log_from_trj
from .input_parser import System
from .logger import Logger
from .losses import get_LJ_param
from .nn_options import get_training_parameters, get_system_options
from .simulate import simulator
from .models import GeneralModel
from .config import center_molecule
from .diagnostics import print_startup_diagnostics, format_simulation_config, format_run_header, print_run_signature

# NOTE: double precision is controlled via 'double_precision = true' in [nn] TOML
# or via the --double-precision CLI flag.  It is enabled early in main() below.



def save_state(dirname: str, data: dict[str, Any]) -> None:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(data)
    # TODO: create new directory if 'dirname' exists, so we don't overwrite with force
    orbax_checkpointer.save(dirname, data, save_args=save_args, force=True)


def load_state(dirname: str, target: dict[str, Any]) -> dict[str, Any]:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(dirname, item=target)


RESTART_STATE_OPTIMIZER = "optimizer"
RESTART_STATE_KINEMATIC = "kinematic"
RESTART_STATE_EXACT = "exact"


_OUTPUT_ARRAY_FILENAMES = {
    "density": "density.npy",
    "Rg PDF": "Rg.npy",
}


def _diagnostic_output_filename(name: str) -> str | None:
    if name in _OUTPUT_ARRAY_FILENAMES:
        return _OUTPUT_ARRAY_FILENAMES[name]

    normalized = name.strip().lower()
    if not any(token in normalized for token in ("kde", "pdf", "distribution")):
        return None

    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return f"{slug}.npy" if slug else None


def _save_output_diagnostics(output: dict[str, Any], destdir: str, system_name: str) -> None:
    for key, value in output.items():
        filename = _diagnostic_output_filename(key)
        try:
            array = onp.asarray(value)
        except Exception:
            Logger.rank0.debug(f"{system_name} {key} = {value}")
            continue

        if filename is not None and array.ndim > 0:
            out_path = os.path.join(destdir, f"{system_name}_{filename}")
            Logger.rank0.debug(f"Saving {key} to '{out_path}'")
            onp.save(out_path, array)
            continue

        if array.ndim == 0:
            Logger.rank0.debug(f"{system_name} {key} = {array.item()}")
        else:
            Logger.rank0.debug(f"{system_name} {key} shape={array.shape}")


# ── Per-epoch loss-component breakdown ────────────────────────────────
# Helpers consume the diag dict produced by losses.py (post-allreduce,
# in `output`) plus per-rank pre-allreduce snapshots gathered to rank 0
# (`all_locals`, multidir only) and emit a single human-readable block
# via Logger.rank0.info → stdout + sim.log.  No file I/O; no JIT impact
# (host-side, runs once per epoch on rank 0 after step() returns).
#
# Schema produced by losses.py:
#   loss_total          — total error returned by the loss
#   loss_<comp>         — weighted contribution per component (= w × err)
#   err_<comp>          — raw metric error per component
#   value_<comp>        — mean observable value (scalar components only)
#   weight_<comp>       — static weight per component
# Bare `loss_<comp>` with no matching value/err (e.g. loss_constraint,
# loss_boundary) is rendered as contrib-only.

def _extract_scalar_components(diag) -> dict[str, float]:
    """Filter a diag dict to scalar numeric entries; skip arrays and
    the reserved '_local' key.  Returns an alphabetically-sorted dict
    of Python floats (stable iteration order across calls).
    """
    if not diag:
        return {}
    out: dict[str, float] = {}
    for key, value in diag.items():
        if key == "_local":
            continue
        try:
            arr = onp.asarray(value)
        except Exception:
            continue
        if arr.ndim == 0 and onp.issubdtype(arr.dtype, onp.number):
            try:
                out[key] = float(arr.item())
            except Exception:
                pass
    return dict(sorted(out.items()))


def _component_pairs(scalars: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group scalar entries by component suffix.

    Returns a dict keyed by component name with sub-keys 'loss', 'err',
    'value', 'weight'.  `loss_total` is excluded (it's the grand total,
    rendered separately in the header line).
    """
    components: dict[str, dict[str, float]] = {}
    for key, val in scalars.items():
        if key == "loss_total":
            continue
        for prefix in ("value_", "err_", "loss_", "weight_"):
            if key.startswith(prefix):
                comp = key[len(prefix):]
                components.setdefault(comp, {})[prefix.rstrip("_")] = val
                break
    return components


def _fmt_component_row(name: str, fields: dict[str, float], name_width: int = 24) -> str:
    """Format a single 'value=… err=… contrib=… (w=…)' row."""
    parts = [f"{name:<{name_width}s}"]
    if "value" in fields:
        parts.append(f"value={fields['value']:.4g}")
    if "err" in fields:
        parts.append(f"err={fields['err']:.4g}")
    if "loss" in fields:
        parts.append(f"contrib={fields['loss']:.3e}")
    if "weight" in fields:
        parts.append(f"(w={fields['weight']:.3g})")
    return "  ".join(parts)


def _format_loss_breakdown_block(
    epoch: int,
    loss_value,
    output,
    all_locals,
    multi_dir: bool,
    dataset,
    replica_map: dict | None = None,
    pair_classification: dict | None = None,
    hetero_mode: bool = False,
) -> str:
    """Build the per-epoch loss-component breakdown text block.

    Single-dir layout (one block):
        Epoch <N>, mean loss = <loss_total>
          Components:
            <comp>  value=…  err=…  contrib=…  (w=…)
            …

    Multidir layout (top-line summary + per-system table):
        Epoch <N> — mean across K system(s): loss = <loss_total>
          Components (post-allreduce mean; range = pre-allreduce min/max):
            <comp>  mean=…  contrib=…  (w=…)  range=[lo, hi]
            …

          Per-system breakdown (pre-allreduce):
            <sysname>  loss=…  <comp1>=…  <comp2>=…  …
            …
    """
    scalars_post = _extract_scalar_components(output or {})
    # In heterogeneous multidir mode `output["loss_total"]` is rank 0's
    # local wasserstein-vs-local-target, NOT a mean across systems.  Trust
    # the explicit `loss_value` (which main() already replaced with the
    # mean of per-rank pre-allreduce `loss_total`s) instead.
    prefer_loss_value = hetero_mode and bool(all_locals)
    if not prefer_loss_value and "loss_total" in scalars_post:
        loss_total_post = scalars_post["loss_total"]
    else:
        try:
            loss_total_post = float(onp.asarray(loss_value).item())
        except Exception:
            loss_total_post = float("nan")

    components = _component_pairs(scalars_post)
    name_width = max([24] + [len(c) + 2 for c in components])

    lines: list[str] = []

    if not multi_dir:
        lines.append(f"Epoch {epoch}, mean loss = {loss_total_post:.6e}")
        if components:
            lines.append("  Components:")
            for comp_name in sorted(components):
                lines.append(
                    "    " + _fmt_component_row(comp_name, components[comp_name], name_width)
                )
        return "\n".join(lines) + "\n"

    # Multidir: cross-rank summary + per-system table
    n_sys = len(all_locals) if all_locals else 0
    lines.append(
        f"Epoch {epoch} — mean across {n_sys} system(s): loss = {loss_total_post:.6e}"
    )

    per_system_scalars: list[dict[str, float]] = []
    if all_locals:
        for r in range(n_sys):
            per_system_scalars.append(_extract_scalar_components(all_locals[r] or {}))

    if components:
        lines.append(
            "  Components (post-allreduce mean; range = pre-allreduce min/max across systems):"
        )
        for comp_name in sorted(components):
            fields = components[comp_name]
            row_parts = [f"    {comp_name:<{name_width}s}"]
            if "value" in fields:
                row_parts.append(f"mean={fields['value']:.4g}")
            elif "err" in fields:
                row_parts.append(f"err={fields['err']:.4g}")
            if "loss" in fields:
                row_parts.append(f"contrib={fields['loss']:.3e}")
            if "weight" in fields:
                row_parts.append(f"(w={fields['weight']:.3g})")

            value_key = f"value_{comp_name}"
            err_key = f"err_{comp_name}"
            range_key = value_key if value_key in scalars_post else err_key
            vals = [s[range_key] for s in per_system_scalars if range_key in s]
            if vals:
                row_parts.append(f"range=[{min(vals):.4g}, {max(vals):.4g}]")

            lines.append("  ".join(row_parts))

    if all_locals and n_sys > 0:
        lines.append("")

        # Group rank indices by replica-group name when replica_map is
        # provided AND at least one group has >1 member.  Otherwise fall
        # back to the per-rank table (current behavior).
        use_grouping = False
        rank_to_group: dict[int, str] = {}
        group_to_ranks: dict[str, list[int]] = {}
        if replica_map and dataset is not None:
            for r in range(n_sys):
                if r < len(dataset):
                    grp = replica_map.get(dataset[r].name, dataset[r].name)
                else:
                    grp = f"sys_{r:02d}"
                rank_to_group[r] = grp
                group_to_ranks.setdefault(grp, []).append(r)
            use_grouping = any(len(v) > 1 for v in group_to_ranks.values())

        # Pair-count strings per logical group (only meaningful in multidir).
        def _pair_counts(group_name: str) -> str:
            if pair_classification is None:
                return ""
            shared = len(pair_classification.get("shared_rows", []))
            private_per_sys = pair_classification.get("private_per_system", {})
            # Sum private rows over all member dir-names in this group.
            members = [
                dataset[r].name for r in group_to_ranks.get(group_name, [])
                if dataset is not None and r < len(dataset)
            ] if use_grouping else [group_name]
            private_count = sum(len(private_per_sys.get(m, [])) for m in members)
            return f"  shared={shared}  private={private_count}"

        if use_grouping:
            lines.append("  Per-logical-system breakdown (avg across replicas, pre-allreduce):")
            name_width = max([24] + [len(g) + 2 for g in group_to_ranks])
            for grp in sorted(group_to_ranks):
                ranks_in = group_to_ranks[grp]
                n_replicas = len(ranks_in)
                group_scalars = [per_system_scalars[r] for r in ranks_in
                                 if r < len(per_system_scalars)]

                def _avg(key: str):
                    vals = [s[key] for s in group_scalars if key in s]
                    return sum(vals) / len(vals) if vals else None

                avg_loss = _avg("loss_total")
                parts = [f"    {grp:<{name_width}s}"]
                if avg_loss is not None:
                    parts.append(f"loss={avg_loss:.4e}")
                parts.append(f"({n_replicas} replica{'s' if n_replicas != 1 else ''})")
                for comp_name in sorted(components):
                    v = _avg(f"value_{comp_name}")
                    e = _avg(f"err_{comp_name}")
                    if v is not None:
                        parts.append(f"{comp_name}={v:.4g}")
                    elif e is not None:
                        parts.append(f"{comp_name}_err={e:.4g}")
                pc = _pair_counts(grp)
                if pc:
                    parts.append(pc)
                lines.append("  ".join(parts))
        else:
            lines.append("  Per-system breakdown (pre-allreduce):")
            sysname_width = max(
                [24]
                + [
                    len(dataset[r].name) + 2
                    for r in range(n_sys)
                    if dataset is not None and r < len(dataset)
                ]
            )
            for r in range(n_sys):
                if dataset is not None and r < len(dataset):
                    sysname = dataset[r].name
                else:
                    sysname = f"sys_{r:02d}"
                ls = per_system_scalars[r] if r < len(per_system_scalars) else {}
                sys_loss = ls.get("loss_total")
                parts = [f"    {sysname:<{sysname_width}s}"]
                if sys_loss is not None:
                    parts.append(f"loss={sys_loss:.4e}")
                for comp_name in sorted(components):
                    value_key = f"value_{comp_name}"
                    err_key = f"err_{comp_name}"
                    if value_key in ls:
                        parts.append(f"{comp_name}={ls[value_key]:.4g}")
                    elif err_key in ls:
                        parts.append(f"{comp_name}_err={ls[err_key]:.4g}")
                pc = _pair_counts(sysname)
                if pc:
                    parts.append(pc)
                lines.append("  ".join(parts))

    return "\n".join(lines) + "\n"


# ── Heterogeneous multidir: LJ-row classification + replica-aware breakdown ──
# In multidir mode each rank runs a different system.  Some LJ rows are
# touched by >=2 systems (shared rows of LJ_param, trained jointly via
# allreduce(SUM, WORLD) on grads) and others by only one system (private
# rows).  Pair-mode rows are owned when both pair types are present;
# type-mode rows are owned when the single trainable type is present.
# The classification is a startup-only host-side scan; nothing JIT-traced.


def _system_type_sets(dataset):
    system_types: dict[str, set[int]] = {}
    for system in dataset:
        ut = onp.asarray(system.config.unique_types).ravel()
        system_types[system.name] = set(int(x) for x in ut.tolist())
    return system_types


def _empty_lj_classification(dataset, mode: str):
    return {
        "shared_rows": [],
        "private_per_system": {s.name: [] for s in dataset},
        "unused_rows": [],
        "row_pair_label": {},
        "row_owners": {},
        "n_owners_per_row": {},
        "lj_mode": mode,
    }


def _finish_lj_row_classification(classification, row: int, label: str, owners: list[str]):
    classification["row_pair_label"][row] = label
    classification["row_owners"][row] = list(owners)
    classification["n_owners_per_row"][row] = len(owners)
    if len(owners) >= 2:
        classification["shared_rows"].append(row)
    elif len(owners) == 1:
        classification["private_per_system"][owners[0]].append(row)
    else:
        classification["unused_rows"].append(row)

def _classify_lj_pairs(params, dataset):
    """Classify every trainable row k of LJ_param by which systems use it.

    Returns a dict with keys:
      shared_rows:        list[int] rows used by >=2 systems
      private_per_system: dict[str, list[int]]  rows used by exactly one
      unused_rows:        list[int] rows used by zero systems (TOML smell)
      row_pair_label:     dict[int, str] e.g. {0: 'SCM-SCM (sigma)'}
    """
    n_types = int(params.n_types)
    ttlj = onp.asarray(params.type_to_LJ)
    flat_to_pair: dict[int, tuple[int, int]] = {}
    for i in range(n_types):
        for j in range(i, n_types):
            flat_to_pair[int(ttlj[i, j])] = (i, j)

    type_to_name = {t: n for n, t in dict(params.lj_name_to_type).items()}

    sigma_idx = (
        onp.asarray(params.lj_sigma_idx) if params.lj_sigma_idx is not None
        else onp.array([], dtype=int)
    )
    epsilon_idx = (
        onp.asarray(params.lj_epsilon_idx) if params.lj_epsilon_idx is not None
        else onp.array([], dtype=int)
    )
    n_sigma = int(params.n_sigma_train)

    system_types = _system_type_sets(dataset)
    classification = _empty_lj_classification(dataset, "pair")

    def _label(i: int, j: int, kind: str) -> str:
        ni = type_to_name.get(i, f"T{i}")
        nj = type_to_name.get(j, f"T{j}")
        return f"{ni}-{nj} ({kind})"

    def _classify(row: int, flat_idx: int, kind: str):
        i, j = flat_to_pair[int(flat_idx)]
        owners = [s.name for s in dataset
                  if i in system_types[s.name] and j in system_types[s.name]]
        _finish_lj_row_classification(classification, row, _label(i, j, kind), owners)

    for k in range(len(sigma_idx)):
        _classify(k, int(sigma_idx[k]), "sigma")
    for k in range(len(epsilon_idx)):
        _classify(n_sigma + k, int(epsilon_idx[k]), "epsilon")

    return classification


def _classify_lj_types(params, dataset):
    """Classify trainable LJ_type_param rows by owning systems."""
    type_to_name = {t: n for n, t in dict(params.lj_name_to_type).items()}
    sigma_idx = (
        onp.asarray(params.lj_sigma_idx) if params.lj_sigma_idx is not None
        else onp.array([], dtype=int)
    )
    epsilon_idx = (
        onp.asarray(params.lj_epsilon_idx) if params.lj_epsilon_idx is not None
        else onp.array([], dtype=int)
    )
    n_sigma = int(params.n_sigma_train)
    system_types = _system_type_sets(dataset)
    classification = _empty_lj_classification(dataset, "type")

    def _label(type_id: int, kind: str) -> str:
        return f"{type_to_name.get(type_id, f'T{type_id}')} ({kind})"

    def _classify(row: int, type_id: int, kind: str):
        owners = [s.name for s in dataset if type_id in system_types[s.name]]
        _finish_lj_row_classification(classification, row, _label(type_id, kind), owners)

    for k in range(len(sigma_idx)):
        _classify(k, int(sigma_idx[k]), "sigma")
    for k in range(len(epsilon_idx)):
        _classify(n_sigma + k, int(epsilon_idx[k]), "epsilon")

    return classification


def _classify_lj_parameters(params, dataset):
    if getattr(params, "lj_mode", "pair") == "type":
        return _classify_lj_types(params, dataset)
    return _classify_lj_pairs(params, dataset)


def _lj_train_count(params) -> int:
    return int(params.n_sigma_train) + int(params.n_epsilon_train)


def _owner_counts_array(params, classification, owner_multiplier: int = 1):
    n_train = _lj_train_count(params)
    owner_counts = onp.zeros(n_train, dtype=onp.int32)
    for row, n_owners in classification["n_owners_per_row"].items():
        row = int(row)
        if 0 <= row < n_train:
            owner_counts[row] = max(int(n_owners), 0) * max(int(owner_multiplier), 1)
    return owner_counts


def _build_grad_normalizer(params, classification, world_size: int, owner_multiplier: int = 1):
    dtype = onp.asarray(params.LJ_param).dtype
    owner_counts = _owner_counts_array(params, classification, owner_multiplier)
    if owner_counts.size == 0:
        return jnp.ones(0, dtype=dtype)
    normalizer = onp.zeros(owner_counts.shape, dtype=onp.float64)
    owned = owner_counts > 0
    normalizer[owned] = float(world_size) / owner_counts[owned].astype(onp.float64)
    return jnp.asarray(normalizer.astype(dtype))


def _apply_lj_grad_normalizer(grads, grad_normalizer):
    scaled_lj = grads.LJ_param * grad_normalizer
    scaled_lj = jnp.where(grad_normalizer == 0, 0.0, scaled_lj)
    return grads.replace(LJ_param=scaled_lj)


def _format_pair_classification(classification, dataset, replica_map=None):
    """Build a human-readable rank-0 startup block for the LJ-row
    classification.  Returns a multi-line string."""
    shared = classification["shared_rows"]
    private = classification["private_per_system"]
    unused = classification["unused_rows"]
    labels = classification["row_pair_label"]
    owners_by_row = classification.get("row_owners", {})
    n_total = len(shared) + len(unused) + sum(len(v) for v in private.values())
    n_systems = len(dataset)

    def _owner_groups(row):
        owners = owners_by_row.get(row, [])
        groups = sorted({(replica_map or {}).get(owner, owner) for owner in owners})
        return owners, groups

    cross_system_shared = []
    replica_shared = []
    for row in shared:
        _, groups = _owner_groups(row)
        if len(groups) >= 2:
            cross_system_shared.append(row)
        else:
            replica_shared.append(row)

    lines = [
        f"Heterogeneous multidir LJ-parameter classification "
        f"(mode={classification.get('lj_mode', 'unknown')}) "
        f"({n_systems} system(s), {n_total} trainable row(s)):"
    ]
    lines.append(
        f"  Shared across owning ranks: {len(shared)} row(s) "
        f"({len(cross_system_shared)} cross-system, "
        f"{len(replica_shared)} replica-shared/group-private)"
    )
    lines.append(
        "    NOTE: 'shared' is rank-based. A row used only by replicas of one "
        "replica_of group is shared across those ranks, but still private to "
        "that logical system."
    )

    lines.append(
        f"  Cross-system shared (used by >=2 replica groups): "
        f"{len(cross_system_shared)} row(s)"
    )
    for k in cross_system_shared:
        owners, groups = _owner_groups(k)
        lines.append(
            f"    [{k:3d}] {labels.get(k, '?')}  "
            f"owners={len(owners)}  groups={', '.join(groups)}"
        )

    lines.append(
        f"  Replica-shared / group-private (one replica group, >=2 ranks): "
        f"{len(replica_shared)} row(s)"
    )
    for k in replica_shared:
        owners, groups = _owner_groups(k)
        group = groups[0] if groups else "?"
        lines.append(
            f"    [{k:3d}] {labels.get(k, '?')}  "
            f"owners={len(owners)}  group={group}"
        )

    # Order private blocks by replica group (so replicas of the same
    # system are listed together).
    if replica_map:
        order = []
        seen = set()
        for system in dataset:
            grp = replica_map.get(system.name, system.name)
            if grp not in seen:
                seen.add(grp)
                order.append(system.name)
    else:
        order = [s.name for s in dataset]

    for sys_name in order:
        rows = private.get(sys_name, [])
        grp = (replica_map or {}).get(sys_name, sys_name)
        suffix = f" (group '{grp}')" if grp != sys_name else ""
        lines.append(f"  Private to single rank '{sys_name}'{suffix}: {len(rows)} row(s)")
        for k in rows:
            lines.append(f"    [{k:3d}] {labels.get(k, '?')}")

    if unused:
        lines.append(
            f"  Unused (trainable but no system uses the pair): {len(unused)} row(s)"
        )
        for k in unused:
            lines.append(f"    [{k:3d}] {labels.get(k, '?')}")
        lines.append(
            "    NOTE: unused rows have zero gradient on every rank. They stay "
            "at their initial value forever. Consider removing them from "
            "[nn.model.LJ_param] in the training TOML."
        )

    return "\n".join(lines)


def _restart_mode_has_runtime_state(restart_mode: str) -> bool:
    return restart_mode in (RESTART_STATE_KINEMATIC, RESTART_STATE_EXACT)


def _slot_value(values, slot: int, fallback_slot: int):
    if slot < len(values):
        return values[slot]
    return values[fallback_slot]


def _rank_slot_arrays(gathered_rank_state, name: str):
    return [onp.asarray(gathered_rank_state[r][name]) for r in range(len(gathered_rank_state))]


def _stacked_rank_slot_arrays(gathered_rank_state, name: str):
    return [onp.stack(_rank_slot_arrays(gathered_rank_state, name))]


def _build_checkpoint_payload(
    epoch: int,
    params,
    opt_state,
    gathered_rank_state,
    init_temps_serialized,
    multi_dir_replicas: bool,
    include_equilibration_state: bool = False,
):
    # v3 stores one homogeneous stacked array per field.  v4 stores one
    # per-rank array leaf per field, so heterogeneous atom counts do not need
    # padding and exact restarts keep each rank's native topology shape.
    payload = {
        "epoch": epoch,
        "params": params,
        "state": opt_state,
        "keys": onp.stack(_rank_slot_arrays(gathered_rank_state, "key")),
        "init_temps": onp.asarray(init_temps_serialized),
    }

    if multi_dir_replicas:
        payload.update(
            {
                "positions":  _rank_slot_arrays(gathered_rank_state, "positions"),
                "velocities": _rank_slot_arrays(gathered_rank_state, "velocities"),
                "box_sizes":  _rank_slot_arrays(gathered_rank_state, "box_sizes"),
            }
        )
        if include_equilibration_state:
            payload["start_positions"]  = _rank_slot_arrays(gathered_rank_state, "start_positions")
            payload["start_velocities"] = _rank_slot_arrays(gathered_rank_state, "start_velocities")
            payload["start_box_sizes"]  = _rank_slot_arrays(gathered_rank_state, "start_box_sizes")
        return payload

    payload.update(
        {
            "positions":  _stacked_rank_slot_arrays(gathered_rank_state, "positions"),
            "velocities": _stacked_rank_slot_arrays(gathered_rank_state, "velocities"),
            "box_sizes":  _stacked_rank_slot_arrays(gathered_rank_state, "box_sizes"),
        }
    )
    if include_equilibration_state:
        payload["start_positions"]  = _stacked_rank_slot_arrays(gathered_rank_state, "start_positions")
        payload["start_velocities"] = _stacked_rank_slot_arrays(gathered_rank_state, "start_velocities")
        payload["start_box_sizes"]  = _stacked_rank_slot_arrays(gathered_rank_state, "start_box_sizes")
    return payload


def _restart_mode_has_equilibration_state(restart_mode: str) -> bool:
    return restart_mode == RESTART_STATE_EXACT


def _build_restart_target(
    layout: str,
    params,
    opt_state,
    key,
    dataset,
    size: int,
    i_sys: int,
    start_pos=None,
    start_vel=None,
    start_config=None,
    include_equilibration_state: bool = False,
):
    target = {
        "epoch": 0,
        "params": params,
        "state": opt_state,
    }

    if layout == "minimal":
        return target

    # v3: single homogeneous stacked per-rank state slot.
    # v4: one per-rank array leaf per field, allowing heterogeneous atom counts.
    # dataset[i_sys] is the entry this rank owns; for v4 rank slots map to
    # dataset[slot] in multidir mode and fall back to dataset[i_sys] in
    # single-dir replica mode.
    ref = dataset[i_sys]
    if layout == "v4":
        target.update(
            {
                "keys":       onp.zeros((size,) + onp.asarray(key).shape),
                "positions":  [onp.zeros_like(onp.asarray(_slot_value(dataset, r, i_sys).positions)) for r in range(size)],
                "velocities": [onp.zeros_like(onp.asarray(_slot_value(dataset, r, i_sys).velocities)) for r in range(size)],
                "box_sizes":  [onp.zeros_like(onp.asarray(_slot_value(dataset, r, i_sys).config.box_size)) for r in range(size)],
                "init_temps": onp.zeros(len(dataset)),
            }
        )
    elif layout == "v3":
        target.update(
            {
                "keys":       onp.zeros((size,) + onp.asarray(key).shape),
                "positions":  [onp.zeros((size,) + onp.asarray(ref.positions).shape)],
                "velocities": [onp.zeros((size,) + onp.asarray(ref.velocities).shape)],
                "box_sizes":  [onp.zeros((size,) + onp.asarray(ref.config.box_size).shape)],
                "init_temps": onp.zeros(len(dataset)),
            }
        )
    elif layout == "v2":
        target.update(
            {
                "key": key,
                "positions":  [ref.positions],
                "velocities": [ref.velocities],
                "box_sizes":  [ref.config.box_size],
                "init_temps": onp.zeros(len(dataset)),
            }
        )
    else:
        raise ValueError(f"Unknown restart target layout: {layout}")

    # Equilibration-state placeholders.  Orbax restoration is strict by
    # default: if the on-disk checkpoint has keys the user-provided
    # template does not, the load fails (ValueError on tree-structure
    # mismatch).  Therefore, whenever the *caller* knows the checkpoint
    # may carry equilibration buffers — i.e. start_pos/start_vel/
    # start_config were passed — we always populate the corresponding
    # template entries, regardless of the requested restart mode.
    # ``include_equilibration_state`` only controls whether the
    # restored values are USED downstream (kinematic mode loads them
    # but discards them; exact mode applies them via
    # ``_restore_saved_equilibration_state``).
    have_equil_buffers = (
        start_pos is not None and start_vel is not None and start_config is not None
    )
    if have_equil_buffers:
        sp_ref = start_pos[i_sys]
        sv_ref = start_vel[i_sys]
        sc_ref = start_config[i_sys]
        if layout == "v4":
            target["start_positions"]  = [onp.zeros_like(onp.asarray(_slot_value(start_pos, r, i_sys))) for r in range(size)]
            target["start_velocities"] = [onp.zeros_like(onp.asarray(_slot_value(start_vel, r, i_sys))) for r in range(size)]
            target["start_box_sizes"]  = [onp.zeros_like(onp.asarray(_slot_value(start_config, r, i_sys).box_size)) for r in range(size)]
        elif layout == "v3":
            target["start_positions"]  = [onp.zeros((size,) + onp.asarray(sp_ref).shape)]
            target["start_velocities"] = [onp.zeros((size,) + onp.asarray(sv_ref).shape)]
            target["start_box_sizes"]  = [onp.zeros((size,) + onp.asarray(sc_ref.box_size).shape)]
        else:
            target["start_positions"]  = [sp_ref]
            target["start_velocities"] = [sv_ref]
            target["start_box_sizes"]  = [sc_ref.box_size]
    elif include_equilibration_state:
        # Caller explicitly asked for the equilibration restoration
        # path but did not pass the buffers — that's a programming
        # error, not a checkpoint-shape issue.
        raise ValueError(
            "Equilibration restart state requested without start_pos/start_vel/start_config."
        )

    return target


def _restart_load_candidates(requested_mode: str, include_equilibration_state: bool):
    # Only v4 -> v3 -> v2 layout fallback is allowed: that is a disk-format concern,
    # not a runtime-state downgrade. No cross-mode fallback — if the disk
    # doesn't have what the requested mode needs, _load_restart_checkpoint
    # raises and the user picks a different --restart-state.
    if requested_mode == RESTART_STATE_OPTIMIZER:
        return [(RESTART_STATE_OPTIMIZER, "minimal", False)]

    if requested_mode == RESTART_STATE_KINEMATIC:
        return [
            (RESTART_STATE_KINEMATIC, "v4", False),
            (RESTART_STATE_KINEMATIC, "v3", False),
            (RESTART_STATE_KINEMATIC, "v2", False),
        ]

    if requested_mode == RESTART_STATE_EXACT:
        include = include_equilibration_state
        return [
            (RESTART_STATE_EXACT, "v4", include),
            (RESTART_STATE_EXACT, "v3", include),
            (RESTART_STATE_EXACT, "v2", include),
        ]

    raise ValueError(f"Unknown restart mode: {requested_mode}")


def _summarise_tree_mismatch(exc: Exception) -> str:
    """Extract the key-level diff from an orbax tree-mismatch ValueError.

    Orbax labels its sides as ``Source`` (= the in-memory user-provided
    template, the "source" of the restoration request) and ``Target``
    (= the saved data on disk, where the values are sourced *from*).
    The labels are counter-intuitive; for our user-facing message we
    invert them:

      * ``Source: MISSING`` -> template lacks the key the disk has
        -> from the user's POV: ``extra on disk``.
      * ``Target: MISSING`` -> disk lacks the key the template asks for
        -> from the user's POV: ``checkpoint lacks``.

    Confirmed empirically against orbax 0.x by saving a dict with
    ``start_*`` and restoring with a target that lacks them: orbax
    reports ``Source: MISSING`` for those keys.
    """
    msg = str(exc)
    template_missing, disk_missing = [], []
    current_key = None
    for raw_line in msg.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if stripped == line and line.endswith(":"):
            current_key = line[:-1].strip()
            continue
        if current_key is None:
            continue
        if "Source: MISSING" in line:
            template_missing.append(current_key)
        elif "Target: MISSING" in line:
            disk_missing.append(current_key)

    parts = []
    if disk_missing:
        parts.append(f"checkpoint lacks {disk_missing}")
    if template_missing:
        parts.append(f"extra on disk: {template_missing}")
    if not parts:
        return type(exc).__name__ + ": " + msg.splitlines()[0][:160]
    return "; ".join(parts)


def _load_restart_checkpoint(
    dirname: str,
    requested_mode: str,
    params,
    opt_state,
    key,
    dataset,
    size: int,
    i_sys: int,
    start_pos=None,
    start_vel=None,
    start_config=None,
):
    # G5: before trying to load, surface a targeted hint if the directory
    # doesn't exist and a sibling step_{N-1}/cpt does.
    if not os.path.isdir(dirname):
        hint = ""
        m = re.match(r"(.*/step_)(\d+)(/cpt/?)$", dirname)
        if m is not None:
            prev_n = int(m.group(2)) - 1
            if prev_n >= 0:
                sibling = f"{m.group(1)}{prev_n}{m.group(3).rstrip('/')}"
                if os.path.isdir(sibling):
                    hint = f" Did you mean '{sibling}' (one epoch earlier)?"
        raise RuntimeError(
            f"Restart directory '{dirname}' does not exist.{hint} "
            "Pass an existing checkpoint via --restart."
        )

    include_equilibration_state = start_pos is not None and start_vel is not None and start_config is not None
    attempt_trace: list[tuple[str, str, str]] = []

    for effective_mode, layout, include_start_state in _restart_load_candidates(
        requested_mode,
        include_equilibration_state,
    ):
        target = _build_restart_target(
            layout,
            params,
            opt_state,
            key,
            dataset,
            size,
            i_sys,
            start_pos=start_pos,
            start_vel=start_vel,
            start_config=start_config,
            include_equilibration_state=include_start_state,
        )
        try:
            restored = load_state(dirname, target)
            if layout in {"v4", "v3"} and restored.get("keys") is None:
                raise KeyError("keys")
            if layout == "v2" and restored.get("key") is None:
                raise KeyError("key")
            return restored, layout
        except Exception as exc:
            attempt_trace.append(
                (effective_mode, layout, _summarise_tree_mismatch(exc))
            )

    trace_str = "; ".join(
        f"[{m}/{lay}: {reason}]" for (m, lay, reason) in attempt_trace
    )
    raise RuntimeError(
        f"Could not load '{dirname}' as a '{requested_mode}' restart.\n"
        f"Tried: {trace_str}.\n"
        "If the checkpoint was saved with older code it may lack fields required "
        f"by '{requested_mode}'; re-save with the current code or pick a "
        "different --restart-state."
    )


def _apply_saved_box(config, saved_box):
    saved_box = jnp.asarray(saved_box)
    scaling = saved_box / config.box_size
    return config.update_box(scaling)


def _decode_init_temps(saved_init_temps):
    return [False if float(v) < 0 else float(v) for v in saved_init_temps]


def _resolve_restored_init_temps(saved_init_temps, restart_mode: str, n_systems: int):
    if restart_mode == RESTART_STATE_KINEMATIC:
        return [False] * n_systems
    return _decode_init_temps(saved_init_temps)


def _describe_invalid_state(
    name: str,
    value,
    require_positive: bool = False,
    bounded_abs_max: float | None = None,
) -> str | None:
    array = onp.asarray(value)
    finite_mask = onp.isfinite(array)
    if not onp.all(finite_mask):
        n_bad = int(array.size - onp.count_nonzero(finite_mask))
        preview = array.reshape(-1)[: min(3, array.size)].tolist()
        return f"{name} has {n_bad} non-finite value(s); preview={preview}"

    if require_positive and not onp.all(array > 0.0):
        if array.ndim == 1 and array.size <= 8:
            preview = array.tolist()
        else:
            preview = array.reshape(-1)[: min(3, array.size)].tolist()
        return f"{name} must stay positive; preview={preview}"

    # Sanity envelope: catches runaway-but-finite values that downstream
    # code (jax_md cell-list allocate) cannot handle. A box that has grown
    # 100s of x → huge cells_per_side → reshape/segment_sum failures that
    # surface as opaque jaxmd errors. Catch it here with a clear message.
    if bounded_abs_max is not None:
        abs_max = float(onp.max(onp.abs(array)))
        if abs_max > bounded_abs_max:
            if array.ndim == 1 and array.size <= 8:
                preview = array.tolist()
            else:
                preview = array.reshape(-1)[: min(3, array.size)].tolist()
            return (
                f"{name} max-abs={abs_max:.3e} exceeds sanity bound "
                f"{bounded_abs_max:.3e}; preview={preview}"
            )

    return None


# Sanity envelopes for runtime state. Box ceiling: 1000 nm (1 µm) is
# orders of magnitude larger than any realistic biomolecular box, so any
# value beyond it is a barostat runaway, not an unusually-large system.
# Positions: should always be wrapped within the box; allow 10x box
# ceiling for extreme transients. Velocities: thermal scale at 300 K is
# ~1 nm/ps for water; 1e4 nm/ps is well past any physical threshold.
_BOX_SIZE_ABS_MAX_NM = 1.0e3
_POSITION_ABS_MAX_NM = 1.0e4
_VELOCITY_ABS_MAX = 1.0e4


def _validate_runtime_state(
    label: str,
    system_name: str,
    positions,
    velocities,
    config,
    *,
    rank: int,
    epoch: int,
) -> None:
    issues = []
    for issue in (
        _describe_invalid_state(
            "positions", positions,
            bounded_abs_max=_POSITION_ABS_MAX_NM,
        ),
        _describe_invalid_state(
            "velocities", velocities,
            bounded_abs_max=_VELOCITY_ABS_MAX,
        ),
        _describe_invalid_state(
            "box_size", config.box_size,
            require_positive=True,
            bounded_abs_max=_BOX_SIZE_ABS_MAX_NM,
        ),
    ):
        if issue is not None:
            issues.append(issue)

    if issues:
        joined = "; ".join(issues)
        raise RuntimeError(
            f"{label} invalid for system '{system_name}' on rank {rank} at epoch {epoch}: {joined}"
        )


def _validate_v3_rank_count(restored, expected_size: int):
    saved_size = int(onp.asarray(restored["keys"]).shape[0])
    if saved_size != expected_size:
        raise ValueError(
            "MPI rank mismatch for v3 checkpoint: "
            f"checkpoint was saved with {saved_size} ranks but current run uses {expected_size}."
        )


def _validate_v4_rank_count(restored, expected_size: int):
    saved_size = int(onp.asarray(restored["keys"]).shape[0])
    slot_counts = {
        name: len(restored[name])
        for name in ("positions", "velocities", "box_sizes")
    }
    bad_slots = {name: n for name, n in slot_counts.items() if n != expected_size}
    if saved_size != expected_size or bad_slots:
        details = [f"keys={saved_size}"]
        details.extend(f"{name}={n}" for name, n in bad_slots.items())
        raise ValueError(
            "MPI rank mismatch for v4 checkpoint: "
            f"checkpoint slots ({', '.join(details)}) do not match current run size {expected_size}."
        )


def _restore_dataset_state(restored, dataset, rank: int, expected_size: int, layout: str, i_sys: int):
    # Restore writes only into dataset[i_sys] — the entry this rank owns.
    # Other dataset[j] stay at their fresh-load H5 state and are never
    # exercised in the loop.
    system = dataset[i_sys]
    if layout == "v4":
        _validate_v4_rank_count(restored, expected_size)
        key = jnp.asarray(restored["keys"][rank])
        system.positions  = jnp.asarray(restored["positions"][rank])
        system.velocities = jnp.asarray(restored["velocities"][rank])
        system.config = _apply_saved_box(system.config, restored["box_sizes"][rank])
        return key

    if layout == "v3":
        _validate_v3_rank_count(restored, expected_size)
        key = jnp.asarray(restored["keys"][rank])
        system.positions  = jnp.asarray(restored["positions"][0][rank])
        system.velocities = jnp.asarray(restored["velocities"][0][rank])
        system.config = _apply_saved_box(system.config, restored["box_sizes"][0][rank])
        return key

    if layout == "v2":
        key = jax.random.fold_in(jnp.asarray(restored["key"]), rank)
        system.positions  = jnp.asarray(restored["positions"][0])
        system.velocities = jnp.asarray(restored["velocities"][0])
        system.config = _apply_saved_box(system.config, restored["box_sizes"][0])
        return key

    raise ValueError(f"Cannot restore molecular state from checkpoint layout {layout!r}.")


def _seed_equilibration_from_dataset(dataset, start_pos, start_vel, start_config):
    for i, system in enumerate(dataset):
        start_pos[i] = system.positions
        start_vel[i] = system.velocities
        start_config[i] = system.config


def _restore_saved_equilibration_state(restored, start_pos, start_vel, start_config, rank: int, expected_size: int, layout: str, i_sys: int):
    if layout == "v4":
        _validate_v4_rank_count(restored, expected_size)
        start_pos[i_sys] = jnp.asarray(restored["start_positions"][rank])
        start_vel[i_sys] = jnp.asarray(restored["start_velocities"][rank])
        start_config[i_sys] = _apply_saved_box(start_config[i_sys], restored["start_box_sizes"][rank])
        return

    if layout == "v3":
        _validate_v3_rank_count(restored, expected_size)
        start_pos[i_sys] = jnp.asarray(restored["start_positions"][0][rank])
        start_vel[i_sys] = jnp.asarray(restored["start_velocities"][0][rank])
        start_config[i_sys] = _apply_saved_box(start_config[i_sys], restored["start_box_sizes"][0][rank])
        return

    if layout == "v2":
        start_pos[i_sys] = jnp.asarray(restored["start_positions"][0])
        start_vel[i_sys] = jnp.asarray(restored["start_velocities"][0])
        start_config[i_sys] = _apply_saved_box(start_config[i_sys], restored["start_box_sizes"][0])
        return

    raise ValueError(f"Cannot restore equilibration state from checkpoint layout {layout!r}.")


def _write_trajectory(destdir, system, types, trj, 
                      config, rank):
    """Write an H5MD trajectory for one system/rank."""
    _charges = onp.asarray(system.charges) if system.charges is not None else None
    out_dataset = OutDataset(destdir, "trajectory", double_out=False)
    store_static(
        out_dataset, system.names, types, system.indices, config,
        system.topol.bonds_2[0], system.topol.bonds_2[1],
        system.topol.molecules, molecules=system.molecules,
        velocity_out=False, force_out=False, charges=_charges,
    )
    write_full_trajectory(out_dataset, trj, system.indices, config, charge_out=_charges is not None)
    out_dataset.file.close()


def _run_debug_simulation(args, dataset, params, key, init_temps, nn_options, system_options, comm, rank, i_sys):
    """Run a single forward pass on this rank's assigned system, save diagnostics, and exit."""
    Logger.rank0.debug("Executing in debug mode...")
    system = dataset[i_sys]
    Logger.rank0.debug(f"Simulating system: {system.name}")

    loss_value, (output, trj, _, config, types) = nn_options.loss(
        params, system, key, init_temps[i_sys], comm,
        **nn_options.loss_args, **system_options.system_args[system.name],
    )
    Logger.rank0.debug(f"Loss = {loss_value}")

    output.pop("_local", None)
    _save_output_diagnostics(output, args.destdir, system.name)

    _write_trajectory(
        f"{args.destdir}/{system.name}/{rank:04d}",
        system, types, trj, config, rank,
    )
    exit()


def main(args, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ── Early precision setup (must happen before any JAX array creation) ──
    # Honor both the CLI flag (--double-precision) and the TOML key
    # (double_precision = true in [nn]).Peek at the TOML directly here
    # so it is possible to enable x64 before get_training_parameters() creates any arrays.
    import tomllib as _tomllib
    _model_path = os.path.abspath(args.model)
    try:
        with open(_model_path, "rb") as _f:
            _toml_peek = _tomllib.load(_f)
        _toml_dp = bool(_toml_peek.get("nn", {}).get("double_precision", False))
    except Exception:
        _toml_dp = False
    _cli_dp = bool(getattr(args, "double_precision", False))
    if _cli_dp or _toml_dp:
        jax.config.update("jax_enable_x64", True)
        Logger.rank0.info(
            "Double precision enabled (jax_enable_x64=True) — "
            f"source: {'--double-precision CLI flag' if _cli_dp else 'TOML double_precision=true'}"
        )

    # Orbax checkpointing requires absolute paths
    args.destdir = os.path.abspath(args.destdir)
    if args.restart:
        args.restart = os.path.abspath(args.restart)
    requested_restart_mode = getattr(args, "restart_state", RESTART_STATE_OPTIMIZER)

    # Print startup diagnostics after config is loaded (see unified banner below)
    # print_startup_diagnostics is called once after config/nn_options are available.

    # Log MPI topology and device assignment
    device = jax.devices()[0]

    def _gpu_uuid() -> str:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
            return lines[0] if lines else "<empty>"
        except Exception as exc:
            return f"<{exc}>"

    local_info = {
        "rank": rank,
        "device": str(device),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_uuid": _gpu_uuid(),
    }
    all_info = comm.gather(local_info, root=0)
    if rank == 0:
        header = f"MPI topology: {size} rank(s), backend={device.platform}"
        lines = [header]
        uuids = [info['gpu_uuid'] for info in all_info]
        n_unique = len(set(uuids))
        lines.append(
            f"  GPU isolation: {n_unique}/{size} unique physical GPUs"
            + (" OK" if n_unique == size else " WARNING: ranks sharing a GPU!")
        )
        for info in all_info:
            lines.append(
                f"  Rank {info['rank']} -> {info['device']} "
                f"(CUDA_VISIBLE_DEVICES={info['cuda_visible']}, uuid={info['gpu_uuid']})"
            )
        Logger.rank0.info("\n".join(lines))

    # Initialize PRNG keys (each rank gets a unique seed)
    key = jax.random.PRNGKey(args.seed + rank)

    def _step_reverse(params, opt_state, key):
        """Gradient via reverse-mode AD (value_and_grad).

        The VJP of mpi4jax.allreduce(SUM) is identity, so the gradient
        returned by value_and_grad is rank-local and requires an explicit
        allreduce to obtain the true (N-invariant) global gradient.
        """
        (loss_value, (output, trj, key, config, types)), grads = value_and_grad(
            nn_options.loss, has_aux=True
        )(
            params,
            system,
            key,
            start_temperature,
            comm,
            **nn_options.loss_args,
            **system_options.system_args[system.name],
            replica_comm=replica_subcomm,
        )

        # ── Sanitize rank-local gradients BEFORE allreduce ──────────────
        # Float32 backward pass through long simulations can overflow to
        # NaN/Inf.  Replacing them with 0 *before* the allreduce prevents
        # one bad rank from poisoning all ranks via the SUM reduction.
        grads = jax.tree.map(
            lambda g: (
                jnp.where(jnp.isfinite(g), g, 0.0)
                if g is not None and hasattr(g, "dtype") else g
            ),
            grads,
        )

        # Allreduce: rank-local → global gradient
        grads = jax.tree.map(
            lambda g: mpi4jax.allreduce(g, op=MPI.SUM, comm=comm) if g is not None else None,
            grads,
        )
        # Owner-averaged normalization (identity in homogeneous mode).
        grads = _apply_lj_grad_normalizer(grads, grad_normalizer)
        return loss_value, grads, output, trj, key, config, types

    def _step_jvp(params, opt_state, key):
        """Gradient via forward-mode AD (jax.jvp).

        One JVP pass per trainable parameter.  The JVP of
        mpi4jax.allreduce(SUM) is allreduce(SUM) of the tangent, so
        each directional derivative is already the global gradient
        component — no post-hoc allreduce is needed.

        Memory usage is O(1) in n_steps (no backward-pass carry stack),
        making this suitable for long simulations where reverse-mode OOMs.

        Note: unlike _step_reverse (which masks rank-local NaN/Inf BEFORE
        the allreduce SUM at lines ~590-602), the JVP path cannot mask
        rank-locally — the SUM lives inside jax.jvp's trace.  A single
        rank's NaN tangent therefore poisons every rank's gradient.  The
        outer post-allreduce mask in step() (~line 745) catches the value
        but cannot identify the offending rank; use
        `tools/diagnose_multidir_optimize.py --restart-from <cpt>` to
        capture per-rank pre-reduce tangents under COMM_SELF.
        """
        loss_args_merged = {
            **nn_options.loss_args,
            **system_options.system_args[system.name],
            "replica_comm": replica_subcomm,
        }

        # Pin key BEFORE defining the closure — Python closures capture by
        # name, so without this the JVP passes would see the post-simulation
        # key after the primal call rebinds ``key`` below.
        _key_for_grad = key

        def _loss_scalar(p):
            loss_val, _ = nn_options.loss(
                p, system, _key_for_grad, start_temperature, comm, **loss_args_merged
            )
            return loss_val

        # Primal pass (for loss value, aux data, key propagation)
        loss_value, (output, trj, key, config, types) = nn_options.loss(
            params, system, _key_for_grad, start_temperature, comm, **loss_args_merged
        )

        # One JVP per parameter → directional derivatives (already global)
        n_lj = params.LJ_param.shape[0]
        grad_components = []
        for i in range(n_lj):
            tangent_lj = jnp.zeros(n_lj).at[i].set(1.0)
            tangent = params.replace(LJ_param=tangent_lj)
            _, dloss_di = jax.jvp(_loss_scalar, (params,), (tangent,))
            grad_components.append(dloss_di)

        grads = _apply_lj_grad_normalizer(
            params.replace(LJ_param=jnp.stack(grad_components)), grad_normalizer
        )
        return loss_value, grads, output, trj, key, config, types

    def _step_fd(params, opt_state, key):
        """Gradient via central finite differences.

        Evaluates L(θ+ε) and L(θ-ε) for each parameter.  Each evaluation
        runs the full loss including allreduce, so the resulting gradient
        is already global — no post-hoc allreduce is needed.

        The relative step size is controlled by nn_options.fd_epsilon.

        Note: same rank-local sanitisation gap as _step_jvp — see the
        note there.
        """
        loss_args_merged = {
            **nn_options.loss_args,
            **system_options.system_args[system.name],
            "replica_comm": replica_subcomm,
        }
        eps_rel = nn_options.fd_epsilon

        # Pin key BEFORE the primal pass — the primal call rebinds ``key``
        # via tuple-unpacking, and the FD evaluations must use the SAME
        # initial key to obtain a comparable gradient.
        _key_for_grad = key

        loss_value, (output, trj, key, config, types) = nn_options.loss(
            params, system, _key_for_grad, start_temperature, comm, **loss_args_merged
        )

        n_lj = params.LJ_param.shape[0]
        grad_components = []
        for i in range(n_lj):
            p_i = params.LJ_param[i]
            # Minimum absolute step prevents the relative step from
            # collapsing for sub-nm sigma / sub-kJ epsilon values.
            # For f32 stochastic losses the noise floor is ~1e-6, so
            # eps_i must stay well above that — floor of 1e-4 keeps the
            # signal-to-noise ratio above ~100 for typical CG/atomistic losses.
            eps_i = eps_rel * jnp.maximum(jnp.abs(p_i), 1e-4)

            lj_plus = params.LJ_param.at[i].set(p_i + eps_i)
            lj_minus = params.LJ_param.at[i].set(p_i - eps_i)

            loss_plus, _ = nn_options.loss(
                params.replace(LJ_param=lj_plus), system, _key_for_grad, start_temperature,
                comm, **loss_args_merged,
            )
            loss_minus, _ = nn_options.loss(
                params.replace(LJ_param=lj_minus), system, _key_for_grad, start_temperature,
                comm, **loss_args_merged,
            )
            grad_components.append((loss_plus - loss_minus) / (2.0 * eps_i))

        grads = _apply_lj_grad_normalizer(
            params.replace(LJ_param=jnp.stack(grad_components)), grad_normalizer
        )
        return loss_value, grads, output, trj, key, config, types

    def step(params, opt_state, key):
        # Dispatch gradient computation based on grad_method
        if nn_options.grad_method == "jvp":
            loss_value, grads, output, trj, key, config, types = _step_jvp(
                params, opt_state, key
            )
        elif nn_options.grad_method == "finite_diff":
            loss_value, grads, output, trj, key, config, types = _step_fd(
                params, opt_state, key
            )
        else:
            loss_value, grads, output, trj, key, config, types = _step_reverse(
                params, opt_state, key
            )

        # Save stuff for plotting.  Multi-dir replicas: each rank simulates
        # its own system, but the loss's internal allreduce mixes per-rank
        # diag arrays into a cross-system mean — so the post-allreduce
        # `output` dict is mislabeled with dataset[0].name on rank 0 and
        # the per-system breakdown is lost.  Each composite loss now also
        # returns a per-rank pre-allreduce snapshot under output["_local"];
        # we gather it across ranks and emit one row per system.  In the
        # single-dir replica case the cross-replica mean IS meaningful and
        # is logged as before; the per-rank snapshots would be N noisy
        # replicas of the same system, so we drop them.
        local_diag = output.pop("_local", None)
        all_locals = None
        if multi_dir_replicas and local_diag is not None:
            payload = {k: onp.asarray(v) for k, v in local_diag.items()}
            all_locals = comm.gather(payload, root=0)
            if rank == 0:
                for r in range(size):
                    _save_output_diagnostics(all_locals[r], destdir, dataset[r].name)
        elif rank == 0:
            _save_output_diagnostics(output, destdir, system.name)

        # ── NaN / Inf gradient sanitization ──────────────────────────────
        # Replace any remaining NaN/Inf with 0.  The per-rank masking in
        # _step_reverse catches most cases, but this also protects JVP/FD
        # paths and any NaN introduced during allreduce.
        _grad_leaves = jax.tree.leaves(grads)
        _grad_bad = any(
            jnp.any(jnp.isnan(g)) | jnp.any(jnp.isinf(g))
            for g in _grad_leaves if g is not None and hasattr(g, "dtype")
        )
        if _grad_bad:
            # The post-allreduce gradient is rank-invariant (every rank
            # holds the same SUM), so this site cannot identify which
            # rank's pre-reduce contribution introduced the NaN/Inf.  In
            # multidir mode `system.name` is rank-local — logging it here
            # would be misleading.  Use the diagnostic tool to localise.
            Logger.rank0.warning(
                "NaN/Inf in post-allreduce gradients (rank-invariant view "
                "of the cross-rank SUM) — replacing with 0 to protect "
                "optimizer state. The offending rank cannot be identified "
                "from this site; run "
                "`tools/diagnose_multidir_optimize.py --restart-from <cpt>` "
                "to localise per-rank pre-reduce NaN."
            )
            grads = jax.tree.map(
                lambda g: (
                    jnp.where(jnp.isfinite(g), g, 0.0)
                    if g is not None and hasattr(g, "dtype") else g
                ),
                grads,
            )

        # Update parameters
        updates, opt_state = nn_options.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Hard clipping with configurable bounds (different for sigma vs epsilon)
        # LJ_param is packed as [sigma_values..., epsilon_values...]
        n_sigma = params.n_sigma_train
        n_eps = params.n_epsilon_train
        
        # Build lower/upper bound arrays: sigma bounds for first n_sigma, epsilon for rest
        lower_bounds = jnp.concatenate([
            jnp.full((n_sigma,), nn_options.clip_sigma_min),
            jnp.full((n_eps,), nn_options.clip_epsilon_min),
        ])
        upper_bounds = jnp.concatenate([
            jnp.full((n_sigma,), nn_options.clip_sigma_max),
            jnp.full((n_eps,), nn_options.clip_epsilon_max),
        ])
        
        tree_lower = params.replace(LJ_param=lower_bounds)
        tree_upper = params.replace(LJ_param=upper_bounds)

        params = optax.projections.projection_box(params, tree_lower, tree_upper)

        # Log the current loss and gradients
        Logger.rank0.debug(
            f"System {system.name}, current_loss: {loss_value}\n{50*'*'}\n"
            f"Gradients\n{grads}{50*'-'}\n"
            f"Updated parameters\n{params}{50*'-'}",
        )

        return params, opt_state, loss_value, trj, key, config, types, output, all_locals

    # Read tomli file
    nn_options, params, toml_input = get_training_parameters(args.model)

    n_systems = len(nn_options.systems)
    # Two supported modes:
    #   1. Single-dir replicas: n_systems == 1, any size (all ranks share dir[0],
    #      diverge via per-rank PRNG key).  i_sys = 0 for every rank.
    #   2. Multi-dir replicas:  n_systems > 1 and size == n_systems, rank r
    #      pinned to dir[r] (strict 1:1, explicit starting conformations).
    multi_dir_replicas = n_systems > 1
    if multi_dir_replicas and size != n_systems:
        raise RuntimeError(
            f"'systems' lists {n_systems} directories; strict 1:1 mapping "
            f"requires MPI world size to match (launch with -np {n_systems}; "
            f"current size={size}). Either reduce 'systems' to one entry for "
            f"single-dir replica mode, or relaunch with -np {n_systems}."
        )
    i_sys = rank if multi_dir_replicas else 0
    if multi_dir_replicas and nn_options.shuffle:
        Logger.rank0.warning(
            "'shuffle' is ignored in multi-dir replica mode (each rank is "
            "pinned to its own directory)."
        )

    # Get system specific information
    dataset = []
    for dir in nn_options.systems:
        dataset.append(System.constructor(args, nn_options.name_to_type, dir, params, rank=rank))

    # Upcast positions/velocities to float64 when running in double precision.
    # This mirrors the same logic in mdrun.main() and ensures the simulator
    # receives f64 arrays from the start.
    if jax.config.jax_enable_x64:
        for _sys in dataset:
            _sys.positions  = _sys.positions.astype(jnp.float64)
            _sys.velocities = _sys.velocities.astype(jnp.float64)

    system_options, toml_input = get_system_options(toml_input, dataset, nn_options.name_to_type)

    # Heterogeneous-multidir classification + replica sub-communicator.
    # Classification is rank-0 informational only; gradient reduction stays
    # `allreduce(SUM, WORLD)` on the full grads tree.  The sub-comm is used
    # only for diagnostic gathers (plain mpi4py, NOT mpi4jax).
    pair_classification = _classify_lj_parameters(params, dataset)
    replica_map = system_options.replica_map
    if multi_dir_replicas:
        Logger.rank0.info(
            _format_pair_classification(pair_classification, dataset, replica_map)
        )
        all_groups = sorted(set(replica_map.values()))
        my_group = replica_map[dataset[i_sys].name]
        my_color = all_groups.index(my_group)
        replica_subcomm = comm.Split(color=my_color, key=rank)
        replica_size = replica_subcomm.Get_size()
        if replica_size > 1:
            Logger.rank0.info(
                f"Replica group '{my_group}' (rank {rank}): "
                f"{replica_size} replica(s) declared via `replica_of`."
            )
    else:
        # Single-dir replica mode: all ranks share dir[0]; no need for a
        # separate sub-comm (WORLD already is the replica group).
        replica_subcomm = comm
        my_group = dataset[i_sys].name

    # Heterogeneous-multidir predicate: more than one distinct replica group
    # spans the world.  In homogeneous multidir (one logical system, N
    # replicas) and single-dir mode this is False, and reporting paths fall
    # back to bit-identical pre-change behavior.
    hetero_mode = (
        multi_dir_replicas
        and replica_map is not None
        and len(set(replica_map.values())) > 1
    )

    # ── Owner-averaged gradient normalizer ──────────────────────────────
    # The KDE allreduce(SUM)-then-/N inside losses.py:703-704 bakes a
    # `/world_size` factor into every per-rank simulator gradient.  After
    # `allreduce(SUM, WORLD)` on grads the result is
    # `(g_A + g_B + 0_C) / N` for a row owned by 2 of N systems.  That
    # is world-averaged, NOT owner-averaged.  Multiply each row by
    # `N / n_owners(k)` to recover `(g_A + g_B) / n_owners` (each owning
    # system contributes the same effective gradient regardless of how
    # many ranks share the row).
    #
    # Back-compat: when every row is owned by every rank (single-dir,
    # homogeneous multidir, replicas of one logical system) the
    # multiplier reduces to `N / N = 1` for every k -> identity.
    # Bit-equivalent to prior behavior on those paths.
    n_train = _lj_train_count(params)
    if n_train > 0:
        owner_multiplier = 1 if multi_dir_replicas else size
        _n_own = _owner_counts_array(params, pair_classification, owner_multiplier)
        grad_normalizer = _build_grad_normalizer(
            params, pair_classification, size, owner_multiplier=owner_multiplier
        )
        if multi_dir_replicas and bool(onp.any((_n_own > 0) & (_n_own < size))):
            Logger.rank0.info(
                f"Heterogeneous multidir gradient normalization active "
                f"(world_size / n_owners per row). Rows: "
                f"homogeneous(n=N)={int(onp.sum(_n_own == size))}, "
                f"shared(1<n<N)={int(onp.sum((_n_own > 1) & (_n_own < size)))}, "
                f"private(n=1)={int(onp.sum(_n_own == 1))}, "
                f"unused(n=0)={int(onp.sum(_n_own == 0))}. "
                f"Effective regularizer strengths are independent of world_size "
                f"(regularizers are globalized inside losses.py)."
            )
    else:
        grad_normalizer = jnp.ones(0, dtype=onp.asarray(params.LJ_param).dtype)

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
        _run_debug_simulation(
            args, dataset, params, key, init_temps,
            nn_options, system_options, comm, rank, i_sys,
        )

    start_epoch = 0
    opt_state = nn_options.optimizer.init(params)
    out_loss = f"{args.destdir}/loss.dat"

    # Save original pytree treedefs — needed to reconstruct params/opt_state
    # after orbax restore + comm.bcast, which create new array objects for
    # static (metadata) fields.  JAX 0.9+ requires metadata to be hashable
    # with simple equality; numpy/jax arrays in GeneralModel's static fields
    # (lj_sigma_idx, etc.) break this when they are new objects.
    _params_treedef = jax.tree_util.tree_structure(params)
    _optstate_treedef = jax.tree_util.tree_structure(opt_state)

    if args.restart:
        # Rank 0 loads checkpoint; broadcast to all ranks.  With no cross-mode
        # fallback the effective mode is always requested_restart_mode, so we
        # only need restored + restore_layout across ranks.
        if rank == 0:
            restored, restore_layout = _load_restart_checkpoint(
                args.restart,
                requested_restart_mode,
                params,
                opt_state,
                key,
                dataset,
                size,
                i_sys,
                start_pos=start_pos if nn_options.equilibration else None,
                start_vel=start_vel if nn_options.equilibration else None,
                start_config=start_config if nn_options.equilibration else None,
            )

            if requested_restart_mode == RESTART_STATE_OPTIMIZER:
                Logger.rank0.info(
                    "Optimizer-only restart: restored epoch, params, and optimizer state. "
                    "Saved positions, velocities, box, PRNG keys, and carried equilibration "
                    "state were intentionally ignored. Use '--restart-state kinematic' or "
                    "'--restart-state exact' to resume the saved molecular state."
                )

            if restore_layout == "v2" and _restart_mode_has_runtime_state(requested_restart_mode):
                Logger.rank0.warning(
                    "Old checkpoint (v2): only rank 0's molecular state was saved. "
                    "All ranks will start from rank 0's positions/velocities. "
                    "Re-save with current code for replica-safe restarts."
                )

            if requested_restart_mode == RESTART_STATE_KINEMATIC:
                if restore_layout == "v4":
                    Logger.rank0.info(
                        f"Kinematic restart (v4): heterogeneous per-rank positions, velocities, box, optimizer state, and PRNG keys restored ({size} ranks). "
                        "Temperature and pressure will be recomputed from the resumed state."
                    )
                elif restore_layout == "v3":
                    Logger.rank0.info(
                        f"Kinematic restart (v3): per-rank positions, velocities, box, optimizer state, and PRNG keys restored ({size} ranks). "
                        "Temperature and pressure will be recomputed from the resumed state."
                    )
                elif restore_layout == "v2":
                    Logger.rank0.info(
                        "Kinematic restart (v2): positions, velocities, box, optimizer state, and PRNG key restored. "
                        "Temperature and pressure will be recomputed from the resumed state."
                    )
            elif requested_restart_mode == RESTART_STATE_EXACT:
                if restore_layout == "v4":
                    Logger.rank0.info(
                        f"Exact restart (v4): heterogeneous per-rank positions, velocities, box, optimizer state, PRNG keys, and carried equilibration state restored ({size} ranks)."
                    )
                elif restore_layout == "v3":
                    Logger.rank0.info(
                        f"Exact restart (v3): per-rank positions, velocities, box, optimizer state, PRNG keys, and carried equilibration state restored ({size} ranks)."
                    )
                elif restore_layout == "v2":
                    Logger.rank0.info(
                        "Exact restart (v2): positions, velocities, box, optimizer state, PRNG key, and carried equilibration state restored."
                    )
        else:
            restored = None
            restore_layout = None

        # Broadcast checkpoint to all ranks
        restored = comm.bcast(restored, root=0)
        restore_layout = comm.bcast(restore_layout, root=0)

        start_epoch = restored["epoch"]
        # Reconstruct params and opt_state using original treedefs so that
        # static metadata fields (arrays in GeneralModel) keep their original
        # Python object identity.  Without this, JAX 0.9+ raises
        # "unhashable metadata" when comparing pytree treedefs inside
        # optax / lax.cond.  Only the leaf values come from the checkpoint.
        params = jax.tree_util.tree_unflatten(
            _params_treedef, jax.tree_util.tree_leaves(restored["params"])
        )
        opt_state = jax.tree_util.tree_unflatten(
            _optstate_treedef, jax.tree_util.tree_leaves(restored["state"])
        )

        if _restart_mode_has_runtime_state(requested_restart_mode):
            key = _restore_dataset_state(restored, dataset, rank, size, restore_layout, i_sys)
            init_temps = _resolve_restored_init_temps(
                restored["init_temps"], requested_restart_mode, n_systems
            )

            if nn_options.equilibration:
                if requested_restart_mode == RESTART_STATE_EXACT:
                    _restore_saved_equilibration_state(
                        restored,
                        start_pos,
                        start_vel,
                        start_config,
                        rank,
                        size,
                        restore_layout,
                        i_sys,
                    )
                else:
                    # KINEMATIC: no true equilibration seed on disk. Using the
                    # post-step dataset state as a seed is a known
                    # approximation — warn so the user can choose EXACT next
                    # time.
                    Logger.rank0.warning(
                        "Equilibration seed not present on disk for mode "
                        f"'{requested_restart_mode}'; seeding "
                        "start_positions/start_velocities/start_config from "
                        "the post-step dataset state. The differentiable "
                        "trajectory will be reproducible only within "
                        "tolerance of the restarted step. Use "
                        "--restart-state exact to restore the saved seed."
                    )
                    _seed_equilibration_from_dataset(dataset, start_pos, start_vel, start_config)

    # Unified startup banner (hardware + simulation config in one table)
    print_startup_diagnostics(
        title="DIFF-MD OPTIMIZE",
        mpi_comm=comm,
        logger=Logger.rank0,
        config=dataset[0].config,
        nn_options=nn_options,
        params=params,
        restart_path=args.restart if args.restart else None,
        restart_mode=requested_restart_mode if args.restart else None,
    )

    Logger.rank0.info(f"\n\tInitial parameters:\n" f"\t\t{params}")
    Logger.rank0.info(
        f"\tGradient method: {nn_options.grad_method}"
        + (f" (fd_epsilon={nn_options.fd_epsilon})" if nn_options.grad_method == "finite_diff" else "")
    )

    # ── 2A: synchronise loss.dat with start_epoch ────────────────────────
    # Fresh run: truncate any stale file from a previous run in the same
    # destdir.  Restart: keep only rows whose epoch < start_epoch (drops
    # rows from a prior run that was killed between loss-append and
    # checkpoint-save).  Rank 0 only; barrier prevents races with the
    # first epoch's write.
    if rank == 0:
        if not args.restart:
            open(out_loss, "w").close()
        else:
            try:
                with open(out_loss, "r") as _lf:
                    _existing = _lf.readlines()
            except FileNotFoundError:
                _existing = []
            _kept = []
            for _line in _existing:
                _parts = _line.split("\t", 1)
                if not _parts:
                    continue
                try:
                    _e = int(_parts[0].strip())
                except ValueError:
                    continue
                if _e < start_epoch:
                    _kept.append(_line if _line.endswith("\n") else _line + "\n")
            with open(out_loss, "w") as _lf:
                _lf.writelines(_kept)
    comm.Barrier()

    # Run training loop
    for epoch in range(start_epoch, start_epoch + nn_options.n_epochs):
        epoch_loss = 0

        destdir = f"{args.destdir}/step_{epoch}"
        params_file = f"{destdir}/training.toml"
        if rank == 0:
            os.makedirs(destdir, exist_ok=True)
            save_params(params_file, toml_input, params)

        Logger.rank0.debug(f"Starting epoch {epoch}\n{50*'='}\n")

        # Each rank runs exactly one simulation per epoch (scenario 1: all
        # ranks share dataset[0] and diverge via their PRNG key; scenario 2:
        # rank r runs dataset[r] with its own starting conformation).  The
        # loss's internal allreduce(SUM)/comm_size produces the cross-rank
        # mean scalar, identical on every rank.
        system = dataset[i_sys]
        start_temperature = init_temps[i_sys]

        if nn_options.equilibration:
            # Run a longer simulation after every N epochs
            if nn_options.n_epochs_longer and onp.mod(epoch, nn_options.n_epochs_longer) == 0 and epoch != 0:
                equilibration = nn_options.n_steps_longer
            else:
                equilibration = nn_options.equilibration

            # Restarts from initial positions
            _validate_runtime_state(
                "Equilibration input",
                system.name,
                start_pos[i_sys],
                start_vel[i_sys],
                start_config[i_sys],
                rank=rank,
                epoch=epoch,
            )
            sgm, epsl, _, eq_types = get_LJ_param(params, system.config, jnp.array(system.types))
            trj, key, config = simulator(
                # fmt: off
                params, start_pos[i_sys], start_vel[i_sys], eq_types, system.masses, system.charges,
                sgm, epsl, key, system.topol, start_config[i_sys], start_temperature, equilibration,
                differentiable=False,
            )

            _validate_runtime_state(
                "Equilibration output",
                system.name,
                trj["positions"][-1],
                trj["velocities"][-1],
                config,
                rank=rank,
                epoch=epoch,
            )

            # Save raw positions for the differentiable simulation BEFORE
            # center_molecule (which is only for Rg trajectory output).
            system.positions, system.velocities = (
                trj["positions"][-1],
                trj["velocities"][-1],
            )
            system.config = config
            start_temperature = False

            # center_molecule only for correct Rg trajectory output
            if 'radius_of_gyration' in nn_options.loss.__name__:
                trj["positions"] = center_molecule(
                        trj["positions"],
                        trj["box"],
                        system_options.system_args[system.name]['chain_indices']
                    )

            del trj  # free equilibration trajectory immediately
            gc.collect()
            _malloc_trim()

        _validate_runtime_state(
            "Differentiable step input",
            system.name,
            system.positions,
            system.velocities,
            system.config,
            rank=rank,
            epoch=epoch,
        )

        if nn_options.teacher_forcing:
            # Teacher forcing == continuous simulation, restarting from the last step
            (
                params, opt_state, loss_value, trj, key, config, types,
                output_diag, all_locals_diag,
            ) = step(params, opt_state, key)

            _validate_runtime_state(
                "Differentiable step output",
                system.name,
                trj["positions"][-1],
                trj["velocities"][-1],
                config,
                rank=rank,
                epoch=epoch,
            )

            # Save raw positions/velocities for next epoch BEFORE
            # center_molecule, which applies unwrap+shift+mod that is
            # only appropriate for trajectory output / visualisation.
            if nn_options.equilibration:
                start_pos[i_sys], start_vel[i_sys] = (
                    trj["positions"][-1],
                    trj["velocities"][-1],
                )
                start_config[i_sys] = config
            else:
                system.positions, system.velocities = (
                    trj["positions"][-1],
                    trj["velocities"][-1],
                )
                system.config = config

            # center_molecule only for output (Rg visualisation)
            if 'radius_of_gyration' in nn_options.loss.__name__:
                trj["positions"] = center_molecule(
                        trj["positions"],
                        trj["box"],
                        system_options.system_args[system.name]['chain_indices']
                    )

            init_temps[i_sys] = False
            epoch_loss = loss_value

        else:
            (
                params, opt_state, loss_value, trj, key, config, types,
                output_diag, all_locals_diag,
            ) = step(params, opt_state, key)
            _validate_runtime_state(
                "Differentiable step output",
                system.name,
                trj["positions"][-1],
                trj["velocities"][-1],
                config,
                rank=rank,
                epoch=epoch,
            )
            epoch_loss = loss_value

        # In heterogeneous multidir mode, `loss_value` is rank-local and
        # differs across systems (each rank's wasserstein is against its
        # own target).  Replace it on rank 0 with the rank-weighted mean
        # of pre-allreduce per-rank `loss_total`s gathered into
        # `all_locals_diag`.  This is the number that matches the
        # per-logical-system breakdown below and is meaningful as a
        # convergence signal.  In homogeneous / single-dir modes the
        # assignment to `loss_value` is preserved.
        if hetero_mode and rank == 0 and all_locals_diag:
            _losses = [
                float(onp.asarray(s["loss_total"]).item())
                for s in all_locals_diag
                if s is not None and "loss_total" in s
            ]
            if _losses:
                epoch_loss = sum(_losses) / len(_losses)

        if multi_dir_replicas:
            simu_filename = f"{args.destdir}/step_{epoch}/{system.name}/{rank:04d}"
        else:
            simu_filename = f"{args.destdir}/step_{epoch}/{rank:04d}"

        _write_trajectory(simu_filename, system, types, trj, config, rank)

        # Write energy.log (rank 0 only, negligible overhead — data already on host)
        if rank == 0:
            n_print_cfg = int(config.n_print) if config.n_print else 0
            if n_print_cfg > 0:
                energy_log_path = f"{destdir}/{system.name}_energy.log" if multi_dir_replicas else f"{destdir}/energy.log"
                write_energy_log_from_trj(energy_log_path, trj, config, n_print_cfg)

        # Free trajectory data — each trj holds ~n_chunks × (pos+vel+forces)
        # JAX arrays; without explicit deletion they linger until next GC
        # cycle, worsening heap fragmentation on CPU (AMD EPYC / Betzy).
        del trj
        gc.collect()
        _malloc_trim()

        Logger.rank0.info(
            _format_loss_breakdown_block(
                epoch=epoch,
                loss_value=epoch_loss,
                output=output_diag,
                all_locals=all_locals_diag,
                multi_dir=multi_dir_replicas,
                dataset=dataset,
                replica_map=replica_map,
                pair_classification=pair_classification,
                hetero_mode=hetero_mode,
            )
        )

        # ── Gather per-rank state for replica-safe checkpoint (all ranks) ──
        # Each rank contributes a single slot (its own dataset[i_sys]).  Rank 0
        # writes v3 stacked arrays for single-dir replicas and v4 rank-slot
        # lists for multidir, where atom counts may differ by rank.
        _local_cpt = {
            "positions":  onp.asarray(dataset[i_sys].positions),
            "velocities": onp.asarray(dataset[i_sys].velocities),
            "box_sizes":  onp.asarray(dataset[i_sys].config.box_size),
            "key":        onp.asarray(key),
            "init_temp":  (-1.0 if init_temps[i_sys] is False else float(init_temps[i_sys])),
        }
        if nn_options.equilibration:
            _local_cpt["start_positions"]  = onp.asarray(start_pos[i_sys])
            _local_cpt["start_velocities"] = onp.asarray(start_vel[i_sys])
            _local_cpt["start_box_sizes"]  = onp.asarray(start_config[i_sys].box_size)
        _all_cpt = comm.gather(_local_cpt, root=0)

        if rank == 0:
            # Merge per-rank init_temp scalars back into a full length-n_systems
            # list so restart sees each rank's update (only rank r modifies
            # init_temps[i_sys=r] under multi_dir_replicas; scenario 1 all
            # ranks share init_temps[0] identically so either view works).
            if multi_dir_replicas:
                init_temps_serialized = onp.array(
                    [_all_cpt[r]["init_temp"] for r in range(size)]
                )
            else:
                init_temps_serialized = onp.array(
                    [-1.0 if v is False else float(v) for v in init_temps]
                )

            cpt_data = _build_checkpoint_payload(
                epoch + 1,
                params,
                opt_state,
                _all_cpt,
                init_temps_serialized,
                multi_dir_replicas,
                include_equilibration_state=nn_options.equilibration,
            )
            save_state(f"{destdir}/cpt", cpt_data)
            del cpt_data

            # 2B: checkpoint is durable — only now append to loss.dat.  If
            # loss.dat write fails (disk full etc.) the checkpoint still
            # represents the truth of the run, so we log and continue.
            try:
                with open(out_loss, "a") as outfile:
                    print(f"{epoch}\t{epoch_loss}", file=outfile)
            except OSError as exc:
                Logger.rank0.warning(
                    f"Could not append epoch {epoch} loss to '{out_loss}': {exc}"
                )
        del _local_cpt, _all_cpt
        gc.collect()
        _malloc_trim()

        # ── Free mmap'd LLVM code pages (VMA cleanup) ────────────────
        # XLA on CPU compiles each traced function into native code via
        # LLVM, which mmaps executable pages (~5,000 new VMAs per epoch).
        # Linux limits VMAs to vm.max_map_count (default 65,530); without
        # periodic eviction the process hits this limit after ~10 epochs
        # and LLVM's allocateMappedMemory fails with "Cannot allocate
        # memory".  Clearing the compilation cache lets Python GC the
        # XlaComputation/PjRtExecutable objects → ~PjRtExecutable() →
        # LLVM ExecutionEngine destructor → munmap → VMAs freed.
        # The NEXT epoch will retrace and recompile (adding ~30-40 s
        # wall time), but this is far cheaper than an OOM crash.
        # Controlled by  clear_xla_cache = N  in [nn]  (0 = off).
        if (nn_options.clear_xla_cache > 0
                and (epoch + 1) % nn_options.clear_xla_cache == 0):
            gc.collect()
            try:
                jax.clear_caches()
            except AttributeError:
                pass  # JAX < 0.4.20
            gc.collect()
            _malloc_trim()
            Logger.rank0.info(
                f"Epoch {epoch}: XLA compilation cache cleared "
                f"(every {nn_options.clear_xla_cache} epoch(s))."
            )

    if rank == 0:
        # save toml with final parameters
        save_params(f"{args.destdir}/final.toml", toml_input, params)

    print_run_signature()
