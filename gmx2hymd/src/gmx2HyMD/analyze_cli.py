"""diffmd-analyze — Structural analysis tools for Diff-MD and GROMACS trajectories.

Supported modes:
  -q4 / --tetrahedral
      Tetrahedral order parameter Q4 and coordination-distance distributions
      for metal coordination sites.

    --q5 / --trigonal-bipyramidal
            Addison tau5 index and coordination-distance distributions for
            five-coordinate trigonal-bipyramidal sites.

    --density-apl / --density-and-apl
            Lateral density profiles plus area-per-lipid diagnostics using the same
            reflected fixed-width KDE and membrane-centering math as
            ``diff_md.losses.density_and_apl``.

  --rg / --radius-of-gyration
      Radius-of-gyration distributions for polymer or protein chains, with
      optional comparison against saved reference PDFs.

    --dist / --distribution
            Compare saved .xvg/.npy distribution files directly, without requiring
            an input trajectory.

All KDE outputs use the same fixed absolute bandwidth semantics as
``src/diff_md/losses.py``:

    bandwidth = bw * bin_width

This makes the analysis outputs directly compatible with the current loss
functions.
"""

import argparse
import json
import math
import re
import textwrap
import tomllib
from argparse import RawDescriptionHelpFormatter
from itertools import combinations
from pathlib import Path

import numpy as np


# Conversion factors from internal ns to output time units
_TIME_FACTORS: dict[str, float] = {"fs": 1_000_000.0, "ps": 1_000.0, "ns": 1.0}
_BOTTICELLI_PALETTE: tuple[str, ...] = (
    "#2f4858",
    "#8d6e63",
    "#bc6c25",
    "#6d597a",
    "#386641",
    "#b56576",
    "#a98467",
    "#457b9d",
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _minimum_image(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention for orthorhombic PBC."""
    return dr - box * np.around(dr / box)


def _unwrap_chain_pbc(chains_pos: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Unwrap contiguous chains under orthorhombic PBC.

    Matches the chain unwrapping used by the Diff-MD Rg loss.
    """
    if box is None:
        return chains_pos

    dr = np.diff(chains_pos, axis=1)
    dr = _minimum_image(dr, box)
    cumul = np.concatenate(
        [
            np.zeros((chains_pos.shape[0], 1, 3), dtype=chains_pos.dtype),
            np.cumsum(dr, axis=1),
        ],
        axis=1,
    )
    return chains_pos[:, :1, :] + cumul


def tetrahedral_q(
    center: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray | None = None,
) -> float:
    """Compute Q4 for one metal and its four ligand positions."""
    dr = neighbors - center
    if box is not None:
        dr = _minimum_image(dr, box)
    norms = np.linalg.norm(dr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit_vectors = dr / norms

    q_sum = 0.0
    for j, k in combinations(range(4), 2):
        cos_psi = np.dot(unit_vectors[j], unit_vectors[k])
        q_sum += (cos_psi + 1.0 / 3.0) ** 2
    return 1.0 - (3.0 / 8.0) * q_sum


def trigonal_bipyramidal_tau5(
    center: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray | None = None,
) -> float:
    """Compute Addison tau5 for one metal and five ligand positions.

    tau5 = (beta - alpha) / 60, where beta and alpha are the largest and
    second-largest ligand-metal-ligand angles in degrees. Ideal trigonal
    bipyramidal geometry gives 1; ideal square-pyramidal geometry gives 0.
    """
    neighbors = np.asarray(neighbors, dtype=np.float64)
    if neighbors.shape != (5, 3):
        raise ValueError(f"tau5 requires exactly five ligand positions; got shape {neighbors.shape}.")

    dr = neighbors - center
    if box is not None:
        dr = _minimum_image(dr, box)
    norms = np.linalg.norm(dr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit_vectors = dr / norms

    angles = []
    for j, k in combinations(range(5), 2):
        cos_theta = np.clip(np.dot(unit_vectors[j], unit_vectors[k]), -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(cos_theta))))
    beta, alpha = sorted(angles, reverse=True)[:2]
    return (beta - alpha) / 60.0


def site_distances(
    center: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray | None = None,
) -> np.ndarray:
    """Metal-ligand distances for one coordination site."""
    dr = neighbors - center
    if box is not None:
        dr = _minimum_image(dr, box)
    return np.linalg.norm(dr, axis=1)


def _frame_rg(chains_pos: np.ndarray, chain_masses: np.ndarray) -> np.ndarray:
    """Per-chain radius of gyration for one frame.

    Mass-weighted about the center of mass (matches ``gmx gyrate`` and the
    Diff-MD loss in ``diff_md.losses._chain_rg``).
    """
    total_mass = np.sum(chain_masses, axis=1)
    com = np.sum(chain_masses[:, :, None] * chains_pos, axis=1) / total_mass[:, None]
    rg2 = np.sum(
        chain_masses * np.sum((chains_pos - com[:, None, :]) ** 2, axis=2),
        axis=1,
    ) / total_mass
    return np.sqrt(np.maximum(rg2, 1e-30))


# ---------------------------------------------------------------------------
# Trajectory readers
# ---------------------------------------------------------------------------

def _iter_h5_frames(h5_path, start=0, stop=None, stride=1):
    """Yield frames from a Diff-MD H5 trajectory (coordinates in nm)."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        if "particles/all/position/value" in handle:
            coords_ds = handle["particles/all/position/value"]
        elif "coordinates" in handle:
            coords_ds = handle["coordinates"]
        else:
            raise KeyError("Cannot locate coordinate dataset in H5 file.")

        if "particles/all/box/edges/value" in handle:
            box_ds = handle["particles/all/box/edges/value"]
        elif "box" in handle.attrs:
            box_ds = None
            box_static = np.asarray(handle.attrs["box"], dtype=np.float64)
        else:
            box_ds = None
            box_static = None

        n_frames = coords_ds.shape[0]
        frame_stop = n_frames if stop is None else min(stop, n_frames)

        if "particles/all/position/time" in handle:
            time_ds = handle["particles/all/position/time"]
        else:
            time_ds = None

        for idx in range(start, frame_stop, stride):
            pos = np.asarray(coords_ds[idx], dtype=np.float64)
            if np.all(pos == 0) or np.any(np.isnan(pos)):
                break

            if box_ds is not None:
                raw_box = np.asarray(box_ds[idx], dtype=np.float64)
                box = np.diag(raw_box) if raw_box.ndim == 2 else raw_box
            elif box_static is not None:
                box = box_static.copy()
            else:
                box = None

            if time_ds is not None:
                time_ns = float(time_ds[idx]) / 1000.0
            else:
                time_ns = float(idx)

            yield idx, time_ns, pos, box


def _iter_mda_frames(topology_path, traj_path, start=0, stop=None, stride=1):
    """Yield frames from GROMACS trajectory via MDAnalysis (A -> nm)."""
    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path), str(traj_path))
    for ts in universe.trajectory[start:stop:stride]:
        time_ns = ts.time / 1000.0
        pos_nm = ts.positions / 10.0
        dims = ts.dimensions
        box_nm = dims[:3] / 10.0 if dims is not None else None
        yield ts.frame, time_ns, pos_nm, box_nm


def _select_atoms_mda(topology_path, traj_path, metal_sel, ligand_sel, expected_ligands=4):
    """Resolve name-based coordination selections to 0-based indices."""
    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path), str(traj_path))
    metal_atoms = universe.select_atoms(metal_sel)
    if len(metal_atoms) != 1:
        raise ValueError(
            f"Metal selection '{metal_sel}' matched {len(metal_atoms)} atoms "
            f"(expected exactly 1). Refine your selection."
        )

    ligand_atoms = universe.select_atoms(ligand_sel)
    if len(ligand_atoms) < expected_ligands:
        raise ValueError(
            f"Ligand selection '{ligand_sel}' matched {len(ligand_atoms)} atoms "
            f"(need at least {expected_ligands}). Refine your selection."
        )

    if len(ligand_atoms) > expected_ligands:
        print(
            f"WARNING: Ligand selection matched {len(ligand_atoms)} atoms. "
            f"Using the {expected_ligands} closest to the metal in the first frame."
        )
        universe.trajectory[0]
        dists = np.linalg.norm(
            ligand_atoms.positions - metal_atoms.positions,
            axis=1,
        )
        ligand_atoms = ligand_atoms[np.argsort(dists)[:expected_ligands]]

    metal_idx = metal_atoms[0].index
    ligand_indices = [atom.index for atom in ligand_atoms]
    ligand_names = [f"{atom.resname}:{atom.name}" for atom in ligand_atoms]
    metal_name = f"{metal_atoms[0].resname}:{metal_atoms[0].name}"
    return metal_idx, ligand_indices, metal_name, ligand_names


def _select_rg_atoms_mda(topology_path, traj_path, selection, n_chains):
    """Resolve chain atoms for Rg analysis from an MDAnalysis selection."""
    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path), str(traj_path))
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(
            f"Rg selection '{selection}' matched no atoms. Refine your selection."
        )
    if len(atoms) % n_chains != 0:
        raise ValueError(
            f"Rg selection '{selection}' matched {len(atoms)} atoms, which is not "
            f"divisible by n_chains={n_chains}."
        )

    masses = np.asarray(atoms.masses, dtype=np.float64)
    if masses.size != len(atoms) or not np.all(np.isfinite(masses)):
        masses = np.ones(len(atoms), dtype=np.float64)

    n_atoms_per_chain = len(atoms) // n_chains
    chain_indices = np.asarray([atom.index for atom in atoms], dtype=np.int64)
    chain_indices = chain_indices.reshape(n_chains, n_atoms_per_chain)
    chain_masses = masses.reshape(n_chains, n_atoms_per_chain)
    return chain_indices, chain_masses, selection


def _load_h5_rg_selection(h5_path, resname, n_chains):
    """Resolve chain atoms for Rg analysis from H5 metadata."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        if "masses" not in handle:
            raise KeyError(f"H5 file '{h5_path}' does not contain a 'masses' dataset.")
        if "resnames" not in handle:
            raise KeyError(
                f"H5 file '{h5_path}' does not contain a 'resnames' dataset."
            )
        masses = np.asarray(handle["masses"], dtype=np.float64).reshape(-1)
        resnames = np.asarray(handle["resnames"])

    chain_indices = np.where(resnames == np.bytes_(resname))[0]
    if chain_indices.size == 0:
        raise ValueError(f"No atoms with resname='{resname}' found in '{h5_path}'.")
    if chain_indices.size % n_chains != 0:
        raise ValueError(
            f"Found {chain_indices.size} atoms for resname='{resname}', which is not "
            f"divisible by n_chains={n_chains}."
        )

    n_atoms_per_chain = chain_indices.size // n_chains
    chain_indices = chain_indices.reshape(n_chains, n_atoms_per_chain)
    chain_masses = masses[chain_indices]
    return chain_indices, chain_masses, resname


def _decode_h5_label(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(value)


def _load_h5_density_metadata(h5_path: str | Path):
    """Load per-particle types and one stable label per type from a Diff-MD H5."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        if "types" not in handle:
            raise KeyError(f"H5 file '{h5_path}' does not contain a 'types' dataset.")
        particle_types = np.asarray(handle["types"], dtype=np.int64).reshape(-1)

        if "names" in handle:
            raw_names = np.asarray(handle["names"]).reshape(-1)
        else:
            raw_names = None

    type_ids = np.unique(particle_types)
    raw_labels = []
    for type_id in type_ids:
        if raw_names is None:
            raw_labels.append(f"type{int(type_id)}")
            continue

        unique_names = np.unique(raw_names[particle_types == type_id])
        decoded = [_decode_h5_label(name) for name in unique_names]
        decoded = [label for label in decoded if label]
        raw_labels.append(decoded[0] if decoded else f"type{int(type_id)}")

    seen = {}
    type_labels = []
    for type_id, label in zip(type_ids, raw_labels):
        count = seen.get(label, 0)
        seen[label] = count + 1
        type_labels.append(label if count == 0 else f"{label}_{int(type_id)}")

    type_counts = {
        int(type_id): int(np.count_nonzero(particle_types == type_id))
        for type_id in type_ids
    }
    return particle_types, type_ids, type_labels, type_counts


def _resolve_type_identifier(
    selector: str | int,
    type_ids: np.ndarray,
    type_labels: list[str],
) -> int:
    """Resolve a user-provided type label or integer id to a type id."""
    label_to_type = {label: int(type_id) for label, type_id in zip(type_labels, type_ids)}
    if isinstance(selector, str) and selector in label_to_type:
        return label_to_type[selector]

    try:
        selected_type = int(selector)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unknown type selector '{selector}'. Available labels: {', '.join(type_labels)}"
        ) from exc

    if selected_type not in set(int(type_id) for type_id in type_ids):
        raise ValueError(
            f"Type id {selected_type} is not present. Available ids: {', '.join(str(int(t)) for t in type_ids)}"
        )
    return selected_type


def _default_density_grid(box_z: float, nbins: int) -> np.ndarray:
    """Default z-grid centered on zero, matching Diff-MD density bin semantics."""
    if nbins < 2:
        raise ValueError("Density/APL analysis requires at least 2 bins.")
    bin_size = float(box_z) / float(nbins)
    lo = -0.5 * float(box_z) + 0.5 * bin_size
    hi = 0.5 * float(box_z) - 0.5 * bin_size
    return np.linspace(lo, hi, nbins, dtype=np.float64)


def _default_density_grid_from_h5(
    h5_path: str | Path,
    nbins: int,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Build a default density grid wide enough for all selected H5 frames."""
    max_box_z = None
    for _frame_idx, _time_ns, _pos, box in _iter_h5_frames(
        h5_path,
        start=start,
        stop=stop,
        stride=stride,
    ):
        if box is None:
            raise SystemExit(
                "ERROR: Density/APL analysis requires box information in every frame."
            )
        box_z = float(box[2])
        max_box_z = box_z if max_box_z is None else max(max_box_z, box_z)

    if max_box_z is None:
        raise SystemExit("ERROR: No valid frames found in trajectory.")
    return _default_density_grid(max_box_z, nbins)


def _compute_membrane_com(z_pos: np.ndarray, box_z: float) -> float:
    """Circular COM used by diff_md.losses._compute_com."""
    pos_map = 2.0 * math.pi * np.asarray(z_pos, dtype=np.float64) / float(box_z)
    theta = math.atan2(-float(np.sum(np.sin(pos_map))), -float(np.sum(np.cos(pos_map)))) + math.pi
    return float(box_z) * theta / (2.0 * math.pi)


def _center_z_positions(z_pos: np.ndarray, com: float, box_z: float) -> np.ndarray:
    """Reproduce diff_md.losses._center exactly on NumPy arrays."""
    centered = np.asarray(z_pos, dtype=np.float64) + 0.5 * float(box_z) - float(com)
    centered = np.where(centered > float(box_z), centered - float(box_z), centered)
    centered = np.where(centered < 0.0, centered + float(box_z), centered)
    return centered - 0.5 * float(box_z)


def _frame_density_and_apl(
    centered_z: np.ndarray,
    particle_types: np.ndarray,
    type_ids: np.ndarray,
    type_counts: dict[int, int],
    z_grid: np.ndarray,
    bandwidth: float,
    box: np.ndarray,
    n_lipids: int,
) -> tuple[np.ndarray, float]:
    """Per-frame density rows and APL matching diff_md.losses.density_and_apl."""
    box_x, box_y, box_z = (float(box[0]), float(box[1]), float(box[2]))
    xy_area = box_x * box_y
    bin_size = float(z_grid[1] - z_grid[0])
    z_length = float(z_grid[-1] - z_grid[0] + bin_size)
    scaling_factor = xy_area * z_length / z_grid.size

    frame_density = np.zeros((type_ids.size, z_grid.size), dtype=np.float64)
    for row_idx, type_id in enumerate(type_ids):
        type_z = centered_z[particle_types == type_id]
        reflected = np.concatenate((type_z - box_z, type_z, type_z + box_z))
        kde_value = _fixed_bandwidth_kde(reflected, z_grid, bandwidth)
        frame_density[row_idx] = (
            kde_value * bin_size * type_counts[int(type_id)] / scaling_factor * 3.0
        )

    apl = 2.0 * xy_area / float(n_lipids)
    return frame_density, apl


def _select_uniform_frame_indices(
    frame_values: np.ndarray,
    n_samples: int,
    *,
    observable_label: str = "frame series",
) -> np.ndarray:
    """Pick unique frames spread uniformly across an empirical frame-value CDF."""
    frame_values = np.asarray(frame_values, dtype=np.float64).reshape(-1)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if frame_values.size == 0:
        raise ValueError(f"No frames are available for sampling from '{observable_label}'.")
    if n_samples > frame_values.size:
        raise ValueError(
            f"Requested {n_samples} sampled frames from '{observable_label}' but only "
            f"{frame_values.size} analyzed frames are available."
        )

    order = np.argsort(frame_values, kind="stable")
    ranks = np.arange(frame_values.size, dtype=np.float64)
    targets = np.linspace(0.0, frame_values.size - 1, n_samples)
    used = np.zeros(frame_values.size, dtype=bool)
    selected = []

    for target in targets:
        candidate_ranks = np.argsort(np.abs(ranks - target), kind="stable")
        chosen_rank = None
        for candidate_rank in candidate_ranks:
            if not used[candidate_rank]:
                chosen_rank = int(candidate_rank)
                break
        if chosen_rank is None:
            raise RuntimeError("Failed to assign a unique sampled frame.")
        used[chosen_rank] = True
        selected.append(int(order[chosen_rank]))

    return np.asarray(selected, dtype=np.int64)


def _resolve_sample_series(
    series_map: dict[str, np.ndarray],
    selected_label: str | None,
    *,
    default_label: str | None = None,
) -> tuple[str, np.ndarray]:
    """Resolve the frame-level scalar series used for representative frame sampling."""
    if not series_map:
        raise ValueError("No frame-level scalar series are available for sampling.")

    label = selected_label
    if label is None:
        if default_label is not None:
            label = default_label
        elif len(series_map) == 1:
            label = next(iter(series_map))
        else:
            available = ", ".join(series_map.keys())
            raise ValueError(
                "Sampling requires --sample-series when multiple frame-level series are "
                f"available. Choose one of: {available}"
            )

    if label not in series_map:
        available = ", ".join(series_map.keys())
        raise ValueError(
            f"Unknown sample series '{label}'. Available series: {available}"
        )

    return label, np.asarray(series_map[label], dtype=np.float64).reshape(-1)


def _locate_h5_frame_datasets(handle):
    """Return the H5 datasets used to read positions/velocities/box/time."""
    if "particles/all/position/value" in handle:
        pos_ds = handle["particles/all/position/value"]
        step_ds = handle["particles/all/position/step"] if "particles/all/position/step" in handle else None
        time_ds = handle["particles/all/position/time"] if "particles/all/position/time" in handle else None
    elif "coordinates" in handle:
        pos_ds = handle["coordinates"]
        step_ds = None
        time_ds = None
    else:
        raise KeyError("Cannot locate coordinate dataset in H5 file.")

    if "particles/all/velocity/value" in handle:
        vel_ds = handle["particles/all/velocity/value"]
    elif "velocities" in handle:
        vel_ds = handle["velocities"]
    else:
        vel_ds = None

    if "particles/all/box/edges/value" in handle:
        box_ds = handle["particles/all/box/edges/value"]
        box_static = None
    elif "box" in handle.attrs:
        box_ds = None
        box_static = np.asarray(handle.attrs["box"], dtype=np.float64)
    else:
        box_ds = None
        box_static = None

    return pos_ds, vel_ds, box_ds, box_static, step_ds, time_ds


def _load_restart_metadata(
    source_h5: str | Path,
    template_h5: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load flat restart metadata from the source trajectory or a template H5."""
    import h5py

    required = ("indices", "types", "names", "masses", "molecules", "resnames")
    optional = ("charge",)
    metadata: dict[str, np.ndarray] = {}

    sources = [Path(source_h5)]
    if template_h5 is not None:
        sources.append(Path(template_h5))

    for path in sources:
        with h5py.File(path, "r") as handle:
            for name in (*required, *optional):
                if name in metadata or name not in handle:
                    continue
                metadata[name] = np.asarray(handle[name])

    missing = [name for name in required if name not in metadata]
    if missing:
        src = Path(source_h5)
        if template_h5 is None:
            raise KeyError(
                f"H5 trajectory '{src}' is missing restart metadata {missing}. "
                "Provide --sample-template-h5 to fill them from a finish/input H5."
            )
        raise KeyError(
            f"Restart metadata {missing} were missing from both '{src}' and "
            f"template '{template_h5}'."
        )

    return metadata


def _write_restart_ready_h5(
    output_path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    box: np.ndarray,
    metadata: dict[str, np.ndarray],
    step: int,
    time_ps: float,
) -> None:
    """Write one single-frame H5 that diff_md can use directly as input."""
    import h5py

    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    box = np.asarray(box, dtype=np.float32).reshape(3)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        handle.attrs["box"] = box
        if "molecules" in metadata:
            handle.attrs["n_molecules"] = int(np.unique(np.asarray(metadata["molecules"]).reshape(-1)).size)

        handle.create_dataset("coordinates", data=positions[None, ...], dtype=np.float32)
        handle.create_dataset("velocities", data=velocities[None, ...], dtype=np.float32)

        for name, values in metadata.items():
            array = np.asarray(values)
            if name in {"masses", "charge"}:
                array = array.reshape(-1)
            handle.create_dataset(name, data=array)

        handle.create_group("h5md")
        particles = handle.require_group("particles").require_group("all")

        position_group = particles.require_group("position")
        position_group.create_dataset("step", data=np.asarray([step], dtype=np.int32))
        position_group.create_dataset("time", data=np.asarray([time_ps], dtype=np.float32))
        position_group.create_dataset("value", data=positions[None, ...], dtype=np.float32)

        velocity_group = particles.require_group("velocity")
        velocity_group.create_dataset("step", data=np.asarray([step], dtype=np.int32))
        velocity_group.create_dataset("time", data=np.asarray([time_ps], dtype=np.float32))
        velocity_group.create_dataset("value", data=velocities[None, ...], dtype=np.float32)

        box_group = particles.require_group("box").require_group("edges")
        box_group.create_dataset("step", data=np.asarray([step], dtype=np.int32))
        box_group.create_dataset("time", data=np.asarray([time_ps], dtype=np.float32))
        box_matrix = np.zeros((1, 3, 3), dtype=np.float32)
        box_matrix[0, np.arange(3), np.arange(3)] = box
        box_group.create_dataset("value", data=box_matrix, dtype=np.float32)

        particles.create_dataset("species", data=np.asarray(metadata["types"]))
        particles.create_dataset("names", data=np.asarray(metadata["names"]))
        particles.create_dataset(
            "mass",
            data=np.asarray(metadata["masses"], dtype=np.float32).reshape(-1, 1),
            dtype=np.float32,
        )
        if "charge" in metadata:
            particles.create_dataset(
                "charge",
                data=np.asarray(metadata["charge"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )


def _export_sampled_structures(
    source_path: str | Path,
    selected_analysis_indices: np.ndarray,
    trajectory_frame_indices: np.ndarray,
    time_arr: np.ndarray,
    time_unit: str,
    output_dir: Path,
    sample_series_label: str,
    frame_series: dict[str, np.ndarray],
    *,
    source_kind: str,
    template_h5: str | Path | None = None,
    topology_path: str | Path | None = None,
    extra_sample_fields=None,
) -> dict[str, object]:
    """Export sampled structures and a manifest for one frame-level scalar series."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_analysis_indices = np.asarray(selected_analysis_indices, dtype=np.int64).reshape(-1)
    trajectory_frame_indices = np.asarray(trajectory_frame_indices, dtype=np.int64).reshape(-1)

    sample_values = np.asarray(frame_series[sample_series_label], dtype=np.float64).reshape(-1)

    def _record(sample_idx: int, analysis_idx: int, source_frame_idx: int, output_key: str, output_path: Path):
        record = {
            "sample_index": int(sample_idx),
            "analysis_frame_index": int(analysis_idx),
            "frame_index": int(source_frame_idx),
            "time": float(time_arr[analysis_idx]),
            "time_unit": time_unit,
            "observable_label": sample_series_label,
            "observable_value": float(sample_values[analysis_idx]),
            "frame_observables": {
                label: float(np.asarray(series, dtype=np.float64)[analysis_idx])
                for label, series in frame_series.items()
            },
            output_key: str(output_path),
        }
        if extra_sample_fields is not None:
            record.update(extra_sample_fields(int(analysis_idx)))
        return record

    samples = []
    source_path = Path(source_path)

    if source_kind == "h5":
        import h5py
        metadata = _load_restart_metadata(source_path, template_h5=template_h5)
        with h5py.File(source_path, "r") as handle:
            pos_ds, vel_ds, box_ds, box_static, step_ds, time_ds = _locate_h5_frame_datasets(handle)

            for sample_idx, analysis_idx in enumerate(selected_analysis_indices):
                source_frame_idx = int(trajectory_frame_indices[analysis_idx])
                positions = np.asarray(pos_ds[source_frame_idx], dtype=np.float32)
                if vel_ds is not None:
                    velocities = np.asarray(vel_ds[source_frame_idx], dtype=np.float32)
                else:
                    velocities = np.zeros_like(positions, dtype=np.float32)

                if box_ds is not None:
                    raw_box = np.asarray(box_ds[source_frame_idx], dtype=np.float32)
                    box = np.diag(raw_box) if raw_box.ndim == 2 else raw_box
                elif box_static is not None:
                    box = np.asarray(box_static, dtype=np.float32)
                else:
                    raise KeyError(
                        f"H5 trajectory '{source_path}' does not contain box information for sampled structure export."
                    )

                step = int(step_ds[source_frame_idx]) if step_ds is not None else int(source_frame_idx)
                time_ps = float(time_ds[source_frame_idx]) if time_ds is not None else float(source_frame_idx)

                output_path = output_dir / f"sample_{sample_idx:03d}_frame_{source_frame_idx:06d}_input.h5"
                _write_restart_ready_h5(
                    output_path,
                    positions,
                    velocities,
                    box,
                    metadata,
                    step=step,
                    time_ps=time_ps,
                )
                samples.append(_record(sample_idx, int(analysis_idx), source_frame_idx, "output_h5", output_path))

    elif source_kind == "mda":
        import MDAnalysis as mda

        if topology_path is None:
            raise ValueError("GROMACS/MDA sampled structure export requires a topology path.")

        universe = mda.Universe(str(topology_path), str(source_path))
        for sample_idx, analysis_idx in enumerate(selected_analysis_indices):
            source_frame_idx = int(trajectory_frame_indices[analysis_idx])
            universe.trajectory[source_frame_idx]
            output_path = output_dir / f"sample_{sample_idx:03d}_frame_{source_frame_idx:06d}.gro"
            with mda.Writer(str(output_path), universe.atoms.n_atoms) as writer:
                writer.write(universe.atoms)
            samples.append(_record(sample_idx, int(analysis_idx), source_frame_idx, "output_gro", output_path))

    else:
        raise ValueError(f"Unsupported sampled structure source kind '{source_kind}'.")

    manifest = {
        "source_trajectory": str(source_path),
        "source_kind": source_kind,
        "topology": None if topology_path is None else str(topology_path),
        "template_h5": None if template_h5 is None else str(template_h5),
        "sample_mode": "uniform_empirical_cdf_frame_series",
        "sample_series_label": sample_series_label,
        "n_samples": int(len(samples)),
        "output_dir": str(output_dir),
        "samples": samples,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------

def _sanitize_bandwidth(bin_centers: np.ndarray, bandwidth: float) -> float:
    if bin_centers.size < 2:
        raise ValueError("At least two bin centers are required to define a bandwidth.")
    bin_width = float(bin_centers[1] - bin_centers[0])
    min_bw = max(abs(bin_width) * 1e-3, 1e-6)
    return max(float(bandwidth), min_bw)


def _fixed_bandwidth_kde(
    samples: np.ndarray,
    bin_centers: np.ndarray,
    bandwidth: float,
    normalize: bool = False,
) -> np.ndarray:
    """Fixed-bandwidth Gaussian KDE matching Diff-MD loss semantics."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    if samples.size == 0:
        return np.zeros_like(bin_centers)

    bandwidth = _sanitize_bandwidth(bin_centers, bandwidth)
    norm = bandwidth * math.sqrt(2.0 * math.pi)
    diffs = (bin_centers[None, :] - samples[:, None]) / bandwidth
    pdf = np.mean(np.exp(-0.5 * diffs**2) / norm, axis=0)

    if normalize:
        area = np.trapezoid(pdf, bin_centers)
        if area > 0.0:
            pdf = pdf / area
    return pdf


def _compute_kde(
    samples: np.ndarray,
    bin_centers: np.ndarray,
    bw_factor: float = 1.0,
    normalize: bool = False,
) -> np.ndarray:
    """Fixed-width Gaussian KDE with bandwidth = bw_factor * bin_width."""
    bin_width = float(bin_centers[1] - bin_centers[0])
    return _fixed_bandwidth_kde(
        samples,
        bin_centers,
        bw_factor * bin_width,
        normalize=normalize,
    )


def _legacy_adaptive_kde(
    samples: np.ndarray,
    bin_centers: np.ndarray,
    bw_factor: float,
    normalize: bool = False,
) -> np.ndarray:
    """Diagnostic helper reproducing the historical adaptive scipy semantics."""
    from scipy.stats import gaussian_kde

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if samples.size < 2 or np.std(samples) < 1e-12:
        return np.zeros_like(bin_centers, dtype=np.float64)

    bin_width = float(bin_centers[1] - bin_centers[0])
    pdf = gaussian_kde(samples, bw_method=bw_factor * bin_width)(bin_centers)
    if normalize:
        area = np.trapezoid(pdf, bin_centers)
        if area > 0.0:
            pdf = pdf / area
    return np.asarray(pdf, dtype=np.float64)


# ---------------------------------------------------------------------------
# Distribution I/O and metrics
# ---------------------------------------------------------------------------

def _write_xvg(
    path: Path,
    bin_centers: np.ndarray,
    distributions: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Write an XVG file compatible with np.loadtxt(...).T."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Created by diffmd-analyze\n")
        handle.write(f'@    title "{title}"\n')
        handle.write(f'@    xaxis  label "{xlabel}"\n')
        handle.write(f'@    yaxis  label "{ylabel}"\n')
        handle.write("@TYPE xy\n")
        for idx, label in enumerate(distributions):
            handle.write(f'@ s{idx} legend "{label}"\n')

        for row_idx, x_val in enumerate(bin_centers):
            row = f"{x_val:14.6f}"
            for label in distributions:
                row += f"{distributions[label][row_idx]:14.6f}"
            handle.write(row + "\n")


def _write_distribution_npy(
    path: Path,
    bin_centers: np.ndarray,
    distributions: dict[str, np.ndarray],
) -> None:
    """Write Diff-MD-compatible distribution array: [grid; pdf1; pdf2; ...]."""
    stacked = np.vstack([bin_centers] + [np.asarray(distributions[k]) for k in distributions])
    np.save(path, stacked)


def _looks_like_grid(values: np.ndarray) -> bool:
    """Return True when a 1D array is a plausible explicit grid."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return False
    diffs = np.diff(values)
    return bool(np.all(diffs > 0.0) or np.all(diffs < 0.0))


def _load_xvg_legends(path: str | Path) -> list[str]:
    legends: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("@ s") or " legend " not in stripped:
                continue
            first_quote = stripped.find('"')
            last_quote = stripped.rfind('"')
            if first_quote != -1 and last_quote > first_quote:
                legends.append(stripped[first_quote + 1:last_quote])
    return legends


def _load_distribution_family(path: str | Path) -> dict[str, np.ndarray | list[str] | None]:
    """Load one saved distribution source.

    Supported layouts:
      - XVG: grid in column 0, one PDF per remaining column.
      - NPY 1D: one PDF with no embedded grid.
      - NPY 2D (Diff-MD loss-ready): [grid; pdf1; pdf2; ...].
      - NPY 2D (membrane diagnostics): [pdf1; pdf2; ...] with no grid row.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".xvg":
        data = np.loadtxt(path, comments=["#", "@"]).T
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError(f"Reference XVG '{path}' has invalid shape {data.shape}.")
        return {
            "grid": np.asarray(data[0], dtype=np.float64),
            "pdfs": np.asarray(data[1:], dtype=np.float64),
            "labels": _load_xvg_legends(path),
        }

    if suffix == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            return {
                "grid": None,
                "pdfs": np.asarray(data[None, :], dtype=np.float64),
                "labels": [],
            }
        if data.ndim == 2:
            array = np.asarray(data, dtype=np.float64)
            if array.shape[0] >= 2 and _looks_like_grid(array[0]):
                return {
                    "grid": np.asarray(array[0], dtype=np.float64),
                    "pdfs": np.asarray(array[1:], dtype=np.float64),
                    "labels": [],
                }
            return {
                "grid": None,
                "pdfs": array,
                "labels": [],
            }
        raise ValueError(
            f"Reference NPY '{path}' has unsupported shape {data.shape}."
        )

    raise ValueError(
        f"Unsupported reference file extension '{suffix}'. Use '.npy' or '.xvg'."
    )


def _resample_pdf_to_grid(
    pdf: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    """Resample one PDF onto a shared comparison grid."""
    pdf = np.asarray(pdf, dtype=np.float64)
    source_grid = np.asarray(source_grid, dtype=np.float64)
    target_grid = np.asarray(target_grid, dtype=np.float64)

    if pdf.ndim != 1 or source_grid.ndim != 1 or target_grid.ndim != 1:
        raise ValueError("PDF resampling expects 1D PDF and grid arrays.")
    if pdf.shape != source_grid.shape:
        raise ValueError(
            "Embedded distribution grid length does not match the PDF length."
        )
    if source_grid.shape == target_grid.shape and np.allclose(source_grid, target_grid):
        return pdf

    if source_grid[0] > source_grid[-1]:
        source_grid = source_grid[::-1]
        pdf = pdf[::-1]

    resampled = np.interp(target_grid, source_grid, pdf, left=0.0, right=0.0)
    source_area = float(np.trapezoid(pdf, source_grid))
    target_area = float(np.trapezoid(resampled, target_grid))
    if source_area > 0.0 and target_area > 0.0:
        resampled *= source_area / target_area
    return resampled


def _load_reference_grid(path: str | Path, dist_index: int = 0) -> np.ndarray | None:
    """Return a grid from a reference file if it carries one, else None."""
    family = _load_distribution_family(path)
    pdfs = np.asarray(family["pdfs"], dtype=np.float64)
    if dist_index < 0 or dist_index >= pdfs.shape[0]:
        raise ValueError(
            f"dist_index={dist_index} is out of range for reference '{path}'."
        )
    grid = family["grid"]
    return None if grid is None else np.asarray(grid, dtype=np.float64)


def _load_reference_pdf(
    path: str | Path,
    dist_index: int,
    expected_grid: np.ndarray,
) -> np.ndarray:
    """Load a single reference PDF column/row on the expected grid."""
    family = _load_distribution_family(path)
    pdfs = np.asarray(family["pdfs"], dtype=np.float64)
    if dist_index < 0 or dist_index >= pdfs.shape[0]:
        raise ValueError(
            f"dist_index={dist_index} is out of range for reference '{path}'."
        )

    pdf = np.asarray(pdfs[dist_index], dtype=np.float64)
    grid = family["grid"]
    if grid is None:
        if pdf.shape != expected_grid.shape:
            raise ValueError(
                f"Reference '{path}' has {pdf.shape[0]} bins but expected {expected_grid.shape[0]}."
            )
        return pdf

    grid = np.asarray(grid, dtype=np.float64)
    return _resample_pdf_to_grid(pdf, grid, expected_grid)


def _find_reference_grid(references, dist_index: int) -> np.ndarray | None:
    if not references:
        return None
    for _label, ref_path in references:
        grid = _load_reference_grid(ref_path, dist_index=dist_index)
        if grid is not None:
            return grid
    return None


def _cdf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    total = np.clip(np.sum(pdf), 1e-30, None)
    return np.cumsum(pdf / total)


def _pdf_summary(grid: np.ndarray, pdf: np.ndarray) -> dict[str, float | None]:
    area = float(np.trapezoid(pdf, grid))
    peak_idx = int(np.argmax(pdf)) if pdf.size else 0
    info = {
        "area": area,
        "mean": None,
        "std": None,
        "peak_value": float(np.max(pdf)) if pdf.size else 0.0,
        "peak_position": float(grid[peak_idx]) if pdf.size else None,
    }
    if area > 0.0:
        mean = float(np.trapezoid(grid * pdf, grid) / area)
        var = float(np.trapezoid((grid - mean) ** 2 * pdf, grid) / area)
        info["mean"] = mean
        info["std"] = math.sqrt(max(var, 0.0))
    return info


def _kl_forward(pred: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-8
    p = target / np.clip(np.sum(target), 1e-30, None)
    q = pred / np.clip(np.sum(pred), 1e-30, None)
    mask = p > eps
    return float(
        np.sum(
            np.where(
                mask,
                p * np.log(np.maximum(p, eps) / np.maximum(q, eps)),
                0.0,
            )
        )
    )


def _wasserstein_1d(pred: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-30
    p = target / np.clip(np.sum(target), eps, None)
    q = pred / np.clip(np.sum(pred), eps, None)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_labeled_mapping(
    entries: list[str] | None,
    option_name: str,
    caster=lambda x: x,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"{option_name} expects LABEL=VALUE entries. Got '{entry}'.")
        label, value = entry.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label:
            raise ValueError(f"{option_name} got an empty label in '{entry}'.")
        mapping[label] = caster(value)
    return mapping


def _load_botticelli_config(path: str | Path) -> dict:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    raise ValueError(
        f"Unsupported Botticelli config format '{config_path.suffix}'. Use .json or .toml."
    )


def _build_botticelli_style(args) -> dict | None:
    has_overrides = any(
        [
            args.botticelli,
            args.botticelli_config,
            args.figure_size is not None,
            args.plot_dpi is not None,
            args.legend_loc is not None,
            args.hide_integral_panel,
            getattr(args, "hide_title", False),
            args.font_family is not None,
            args.font_size is not None,
            args.background_color is not None,
            args.legend_label,
            args.series_color,
            args.series_linestyle,
            args.series_marker,
            args.series_linewidth,
            args.series_alpha,
        ]
    )
    if not has_overrides:
        return None

    style: dict = {}
    if args.botticelli:
        style = {
            "use_palette": True,
            "figure": {
                "width": 10.5,
                "height": 7.2,
                "dpi": 300,
                "facecolor": "#fbf7f0",
                "axes_facecolor": "#fffdf8",
            },
            "font": {
                "family": "DejaVu Serif",
                "size": 12,
            },
            "legend": {
                "loc": "best",
                "frameon": False,
            },
            "grid": {
                "enabled": True,
                "alpha": 0.22,
                "linestyle": ":",
                "linewidth": 0.8,
            },
            "distribution": {
                "show_integral_panel": True,
            },
            "sample_points": {
                "label": "sampled starts",
                "size": 40,
                "facecolors": "white",
                "edgecolors": "#1f1f1f",
                "linewidths": 0.9,
            },
        }

    if args.botticelli_config is not None:
        style = _deep_update(style, _load_botticelli_config(args.botticelli_config))

    if args.figure_size is not None:
        style = _deep_update(
            style,
            {"figure": {"width": float(args.figure_size[0]), "height": float(args.figure_size[1])}},
        )
    if args.plot_dpi is not None:
        style = _deep_update(style, {"figure": {"dpi": int(args.plot_dpi)}})
    if args.legend_loc is not None:
        style = _deep_update(style, {"legend": {"loc": args.legend_loc}})
    if args.hide_integral_panel:
        style = _deep_update(style, {"distribution": {"show_integral_panel": False}})
    if getattr(args, "hide_title", False):
        style = _deep_update(style, {"figure": {"show_title": False}})
    if args.font_family is not None:
        style = _deep_update(style, {"font": {"family": args.font_family}})
    if args.font_size is not None:
        style = _deep_update(style, {"font": {"size": float(args.font_size)}})
    if args.background_color is not None:
        style = _deep_update(style, {"figure": {"facecolor": args.background_color}})

    labels = _parse_labeled_mapping(args.legend_label, "--legend-label", str)
    if labels:
        style = _deep_update(style, {"labels": labels})

    series_updates: dict[str, dict[str, object]] = {}
    for label, color in _parse_labeled_mapping(args.series_color, "--series-color", str).items():
        series_updates.setdefault(label, {})["color"] = color
    for label, linestyle in _parse_labeled_mapping(args.series_linestyle, "--series-linestyle", str).items():
        series_updates.setdefault(label, {})["linestyle"] = linestyle
    for label, marker in _parse_labeled_mapping(args.series_marker, "--series-marker", str).items():
        series_updates.setdefault(label, {})["marker"] = marker
    for label, linewidth in _parse_labeled_mapping(args.series_linewidth, "--series-linewidth", float).items():
        series_updates.setdefault(label, {})["linewidth"] = linewidth
    for label, alpha in _parse_labeled_mapping(args.series_alpha, "--series-alpha", float).items():
        series_updates.setdefault(label, {})["alpha"] = alpha
    if series_updates:
        style = _deep_update(style, {"series": series_updates})

    return style


def _resolve_figsize(style: dict | None, default: tuple[float, float], section: str | None = None) -> tuple[float, float]:
    if style is None:
        return default

    fig_cfg = dict(style.get("figure", {}))
    if section is not None and isinstance(style.get(section), dict):
        fig_cfg = _deep_update(fig_cfg, style[section])
    return (
        float(fig_cfg.get("width", default[0])),
        float(fig_cfg.get("height", default[1])),
    )


def _style_rc_params(style: dict | None) -> dict[str, object]:
    if style is None:
        return {}

    font_cfg = style.get("font", {}) if isinstance(style.get("font"), dict) else {}
    fig_cfg = style.get("figure", {}) if isinstance(style.get("figure"), dict) else {}
    rc_params = {}
    if "family" in font_cfg:
        rc_params["font.family"] = font_cfg["family"]
    if "size" in font_cfg:
        rc_params["font.size"] = float(font_cfg["size"])
    if "facecolor" in fig_cfg:
        rc_params["figure.facecolor"] = fig_cfg["facecolor"]
    if "axes_facecolor" in fig_cfg:
        rc_params["axes.facecolor"] = fig_cfg["axes_facecolor"]
    return rc_params


def _resolve_series_style(style: dict | None, label: str, index: int) -> tuple[str, dict[str, object]]:
    display_label = label
    plot_kwargs: dict[str, object] = {}
    if style is None:
        return display_label, plot_kwargs

    label_map = style.get("labels", {}) if isinstance(style.get("labels"), dict) else {}
    series_cfg = style.get("series", {}) if isinstance(style.get("series"), dict) else {}
    series_style = series_cfg.get(label, {}) if isinstance(series_cfg.get(label, {}), dict) else {}

    display_label = str(series_style.get("label", label_map.get(label, label)))
    if "color" in series_style:
        plot_kwargs["color"] = series_style["color"]
    elif style.get("use_palette"):
        plot_kwargs["color"] = _BOTTICELLI_PALETTE[index % len(_BOTTICELLI_PALETTE)]

    for key in ("linestyle", "marker"):
        if key in series_style:
            plot_kwargs[key] = series_style[key]
    for key in ("linewidth", "markersize", "alpha"):
        if key in series_style:
            plot_kwargs[key] = float(series_style[key])
    return display_label, plot_kwargs


def _resolve_distribution_x_display(dist_cfg: dict[str, object]) -> tuple[float, str | None]:
    x_scale = 1.0
    if "x_scale" in dist_cfg:
        x_scale *= float(dist_cfg["x_scale"])

    raw_unit = dist_cfg.get("x_unit")
    if raw_unit is None:
        return x_scale, None

    unit = str(raw_unit).strip().lower()
    if unit in {"", "nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return x_scale, "nm"
    if unit in {"angstrom", "angstroms", "angstroem", "angstroems", "ångström", "ångströms"}:
        return x_scale * 10.0, r"$\AA$"

    raise ValueError(
        f"Unsupported distribution x_unit='{raw_unit}'. "
        "Use 'nm' or 'angstrom'."
    )


def _convert_length_xlabel(label: str | None, unit_label: str | None) -> str | None:
    if label is None or unit_label is None or unit_label == "nm":
        return label

    converted = str(label)
    replacements = (
        ("(nm)", f"({unit_label})"),
        ("[nm]", f"[{unit_label}]"),
        (" nm)", f" {unit_label})"),
        (" nm]", f" {unit_label}]"),
        (" nm", f" {unit_label}"),
    )
    for old, new in replacements:
        converted = converted.replace(old, new)
    return converted


def _scale_annotation_axes(
    annotation: dict[str, object],
    x_value_scale: float = 1.0,
    y_value_scale: float = 1.0,
) -> dict[str, object]:
    if x_value_scale == 1.0 and y_value_scale == 1.0:
        return annotation

    scaled = dict(annotation)
    for coord_key, ref_key in (("xy", "xycoords"), ("xytext", "textcoords")):
        coords = scaled.get(coord_key)
        ref = str(scaled.get(ref_key, "data")).lower()
        if coords is None or "data" not in ref:
            continue
        if not isinstance(coords, (list, tuple)) or len(coords) < 1:
            continue

        coords_list = list(coords)
        if x_value_scale != 1.0:
            coords_list[0] = float(coords_list[0]) * x_value_scale
        if len(coords_list) > 1 and y_value_scale != 1.0:
            coords_list[1] = float(coords_list[1]) * y_value_scale
        scaled[coord_key] = coords_list
    return scaled


def _apply_axis_grid(ax, style: dict | None, default_linestyle: str = "--", default_alpha: float = 0.3) -> None:
    if style is None:
        ax.grid(True, linestyle=default_linestyle, alpha=default_alpha)
        return

    grid_cfg = style.get("grid", {}) if isinstance(style.get("grid"), dict) else {}
    if not grid_cfg.get("enabled", True):
        ax.grid(False)
        return

    ax.grid(
        True,
        linestyle=grid_cfg.get("linestyle", default_linestyle),
        alpha=float(grid_cfg.get("alpha", default_alpha)),
        linewidth=float(grid_cfg.get("linewidth", 0.8)),
    )


def _legend_kwargs(style: dict | None, default_frameon: bool = False) -> dict[str, object]:
    if style is None:
        return {"frameon": default_frameon}

    legend_cfg = style.get("legend", {}) if isinstance(style.get("legend"), dict) else {}
    kwargs: dict[str, object] = {"frameon": bool(legend_cfg.get("frameon", default_frameon))}
    if "loc" in legend_cfg:
        kwargs["loc"] = legend_cfg["loc"]
    if "fontsize" in legend_cfg:
        kwargs["fontsize"] = float(legend_cfg["fontsize"])
    if "title" in legend_cfg:
        kwargs["title"] = legend_cfg["title"]
    if "ncol" in legend_cfg:
        kwargs["ncol"] = int(legend_cfg["ncol"])
    if "bbox_to_anchor" in legend_cfg:
        kwargs["bbox_to_anchor"] = tuple(legend_cfg["bbox_to_anchor"])
    return kwargs


def _resolve_panel_style(style: dict | None, section: str, panel: str) -> dict[str, object]:
    if style is None:
        return {}

    merged: dict[str, object] = {}
    axes_cfg = style.get("axes", {}) if isinstance(style.get("axes"), dict) else {}
    if isinstance(axes_cfg.get(panel), dict):
        merged = _deep_update(merged, axes_cfg[panel])

    section_cfg = style.get(section, {}) if isinstance(style.get(section), dict) else {}
    section_axes = section_cfg.get("axes", {}) if isinstance(section_cfg.get("axes"), dict) else {}
    if isinstance(section_axes.get(panel), dict):
        merged = _deep_update(merged, section_axes[panel])
    return merged


def _resolve_panel_value_scale(style: dict | None, section: str, panel: str, axis: str) -> float:
    panel_cfg = _resolve_panel_style(style, section, panel)
    scale = panel_cfg.get(f"{axis}_value_scale", 1.0)
    return float(scale)


def _titles_enabled(style: dict | None, section: str | None = None, panel: str | None = None) -> bool:
    if style is None:
        return True

    figure_cfg = style.get("figure", {}) if isinstance(style.get("figure"), dict) else {}
    if not bool(figure_cfg.get("show_title", True)):
        return False

    if section is not None:
        section_cfg = style.get(section, {}) if isinstance(style.get(section), dict) else {}
        if not bool(section_cfg.get("show_title", section_cfg.get("show_titles", True))):
            return False

    if section is not None and panel is not None:
        panel_cfg = _resolve_panel_style(style, section, panel)
        if "show_title" in panel_cfg and not bool(panel_cfg["show_title"]):
            return False

    return True


def _resolve_panel_annotations(style: dict | None, section: str, panel: str) -> list[dict[str, object]]:
    if style is None:
        return []

    resolved: list[dict[str, object]] = []
    containers = [style]
    if isinstance(style.get(section), dict):
        containers.append(style[section])

    for container in containers:
        annotations = container.get("annotations")
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            target_panel = annotation.get("axis")
            if target_panel is None or target_panel == panel:
                resolved.append(dict(annotation))
    return resolved


def _normalize_annotation_text(text: object) -> str:
    if not isinstance(text, str):
        return str(text)

    normalized = re.sub(r"\s*\\n\s*", "\n", text)
    # Accept a LaTeX-like spaced "//" marker for manual line breaks without
    # hijacking URL-like text such as https://...
    normalized = re.sub(r"\s+//\s+", "\n", normalized)
    return normalized


def _apply_axis_annotations(ax, annotations: list[dict[str, object]]) -> None:
    for annotation in annotations:
        text = annotation.get("text")
        xy = annotation.get("xy")
        if text is None or xy is None:
            continue

        text_label = _normalize_annotation_text(text)

        xy_tuple = tuple(float(value) for value in xy)
        xytext = annotation.get("xytext")
        kwargs: dict[str, object] = {}
        for key, value in annotation.items():
            if key in {"axis", "text", "xy", "xytext"}:
                continue
            if key in {"fontsize", "alpha", "rotation"}:
                kwargs[key] = float(value)
            elif key in {"arrowprops", "bbox"} and isinstance(value, dict):
                kwargs[key] = dict(value)
            else:
                kwargs[key] = value

        if xytext is None:
            ax.annotate(text_label, xy=xy_tuple, **kwargs)
        else:
            ax.annotate(
                text_label,
                xy=xy_tuple,
                xytext=tuple(float(value) for value in xytext),
                **kwargs,
            )


def _apply_axis_overrides(
    ax,
    style: dict | None,
    section: str,
    panel: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_value_scale: float = 1.0,
    y_value_scale: float = 1.0,
    x_unit_label: str | None = None,
) -> None:
    panel_cfg = _resolve_panel_style(style, section, panel)

    final_title = panel_cfg.get("title", title)
    final_xlabel = panel_cfg.get("xlabel", xlabel)
    final_ylabel = panel_cfg.get("ylabel", ylabel)
    final_xlabel = _convert_length_xlabel(final_xlabel, x_unit_label)

    if not _titles_enabled(style, section, panel):
        final_title = None

    if final_title:
        ax.set_title(str(final_title))
    if final_xlabel is not None:
        ax.set_xlabel(str(final_xlabel))
    if final_ylabel is not None:
        ax.set_ylabel(str(final_ylabel))

    if "xlim" in panel_cfg:
        xlim = panel_cfg["xlim"]
        if len(xlim) == 2:
            ax.set_xlim(float(xlim[0]) * x_value_scale, float(xlim[1]) * x_value_scale)
    if "ylim" in panel_cfg:
        ylim = panel_cfg["ylim"]
        if len(ylim) == 2:
            ax.set_ylim(float(ylim[0]) * y_value_scale, float(ylim[1]) * y_value_scale)
    if "xscale" in panel_cfg:
        ax.set_xscale(str(panel_cfg["xscale"]))
    if "yscale" in panel_cfg:
        ax.set_yscale(str(panel_cfg["yscale"]))
    if "xticks" in panel_cfg:
        ax.set_xticks([float(value) * x_value_scale for value in panel_cfg["xticks"]])
    if "yticks" in panel_cfg:
        ax.set_yticks([float(value) * y_value_scale for value in panel_cfg["yticks"]])
    if "xtick_labels" in panel_cfg:
        ax.set_xticklabels([str(value) for value in panel_cfg["xtick_labels"]])
    if "ytick_labels" in panel_cfg:
        ax.set_yticklabels([str(value) for value in panel_cfg["ytick_labels"]])

    if "xtick_format" in panel_cfg or "ytick_format" in panel_cfg:
        from matplotlib.ticker import FormatStrFormatter

        if "xtick_format" in panel_cfg:
            ax.xaxis.set_major_formatter(FormatStrFormatter(str(panel_cfg["xtick_format"])))
        if "ytick_format" in panel_cfg:
            ax.yaxis.set_major_formatter(FormatStrFormatter(str(panel_cfg["ytick_format"])))

    tick_params = panel_cfg.get("tick_params")
    if isinstance(tick_params, dict):
        kwargs = dict(tick_params)
        for key in ("labelsize", "labelrotation", "rotation", "pad", "length", "width"):
            if key in kwargs:
                kwargs[key] = float(kwargs[key])
        ax.tick_params(**kwargs)
    else:
        tick_kwargs: dict[str, object] = {}
        if "tick_labelsize" in panel_cfg:
            tick_kwargs["labelsize"] = float(panel_cfg["tick_labelsize"])
        if "xtick_rotation" in panel_cfg:
            tick_kwargs["axis"] = "x"
            tick_kwargs["labelrotation"] = float(panel_cfg["xtick_rotation"])
        if tick_kwargs:
            ax.tick_params(**tick_kwargs)
        if "ytick_rotation" in panel_cfg:
            ax.tick_params(axis="y", labelrotation=float(panel_cfg["ytick_rotation"]))

    annotations = _resolve_panel_annotations(style, section, panel)
    if x_value_scale != 1.0 or y_value_scale != 1.0:
        annotations = [
            _scale_annotation_axes(
                annotation,
                x_value_scale=x_value_scale,
                y_value_scale=y_value_scale,
            )
            for annotation in annotations
        ]
    _apply_axis_annotations(ax, annotations)


def _apply_figure_layout(fig, style: dict | None, section: str, has_suptitle: bool = False) -> None:
    layout_cfg: dict[str, object] = {}
    if style is not None and isinstance(style.get("layout"), dict):
        layout_cfg = _deep_update(layout_cfg, style["layout"])
    if style is not None and isinstance(style.get(section), dict):
        section_layout = style[section].get("layout")
        if isinstance(section_layout, dict):
            layout_cfg = _deep_update(layout_cfg, section_layout)

    if layout_cfg.get("tight", True):
        tight_kwargs: dict[str, object] = {}
        if "tight_pad" in layout_cfg:
            tight_kwargs["pad"] = float(layout_cfg["tight_pad"])
        if "tight_w_pad" in layout_cfg:
            tight_kwargs["w_pad"] = float(layout_cfg["tight_w_pad"])
        if "tight_h_pad" in layout_cfg:
            tight_kwargs["h_pad"] = float(layout_cfg["tight_h_pad"])
        if "rect" in layout_cfg:
            tight_kwargs["rect"] = tuple(float(value) for value in layout_cfg["rect"])
        elif has_suptitle:
            tight_kwargs["rect"] = (0.0, 0.0, 1.0, 0.97)
        fig.tight_layout(**tight_kwargs)

    adjust_keys = ("left", "right", "bottom", "top", "wspace", "hspace")
    if any(key in layout_cfg for key in adjust_keys):
        fig.subplots_adjust(
            **{key: float(layout_cfg[key]) for key in adjust_keys if key in layout_cfg}
        )


def _resolve_coordination_visible_panels(
    style: dict | None,
    section: str,
    label: str,
) -> list[str]:
    coord_cfg = style.get(section, {}) if isinstance(style, dict) else {}
    panel_cfg = coord_cfg.get("panels", {}) if isinstance(coord_cfg.get("panels"), dict) else {}
    panel_order = [
        "timeseries",
        "distribution",
        "distance_timeseries",
        "distance_distribution",
    ]
    visible = [panel for panel in panel_order if bool(panel_cfg.get(panel, True))]
    if not visible:
        raise ValueError(f"{label} Botticelli config disabled all panels; enable at least one panel.")
    return visible


def _resolve_q4_visible_panels(style: dict | None) -> list[str]:
    return _resolve_coordination_visible_panels(style, "q4", "Q4")


def _plot_distribution_comparison(
    grid: np.ndarray,
    series: dict[str, np.ndarray],
    title: str | None,
    xlabel: str,
    save_path: Path | None,
    show_plot: bool,
    sample_points: tuple[str, np.ndarray] | None = None,
    style: dict | None = None,
) -> None:
    """Plot PDF + CDF comparison for one distribution family."""
    if save_path is None and not show_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping comparison plot.")
        return

    dist_cfg = style.get("distribution", {}) if isinstance(style, dict) else {}
    show_integral_panel = bool(dist_cfg.get("show_integral_panel", True))
    legend_axis_name = str(dist_cfg.get("legend_axis", "pdf")).strip().lower()
    x_value_scale, x_unit_label = _resolve_distribution_x_display(dist_cfg)
    pdf_y_value_scale = _resolve_panel_value_scale(style, "distribution", "pdf", axis="y")
    integral_y_value_scale = _resolve_panel_value_scale(style, "distribution", "integral", axis="y")
    display_grid = np.asarray(grid, dtype=np.float64) * x_value_scale
    figsize = _resolve_figsize(style, (10.0, 9.0 if show_integral_panel else 6.2), section="distribution")
    sample_cfg = style.get("sample_points", {}) if isinstance(style, dict) else {}

    with plt.rc_context(rc=_style_rc_params(style)):
        if show_integral_panel:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
            pdf_ax, integral_ax = axes
        else:
            fig, pdf_ax = plt.subplots(1, 1, figsize=figsize)
            integral_ax = None

        legend_target_ax = pdf_ax
        legend_source_ax = pdf_ax
        if legend_axis_name in {"integral", "cdf", "lower", "bottom"} and integral_ax is not None:
            legend_target_ax = integral_ax
            legend_source_ax = integral_ax

        for idx, (label, pdf) in enumerate(series.items()):
            display_label, plot_kwargs = _resolve_series_style(style, label, idx)
            plot_kwargs.setdefault("linewidth", 2.0)
            scaled_pdf = np.asarray(pdf, dtype=np.float64) * pdf_y_value_scale
            pdf_ax.plot(display_grid, scaled_pdf, label=display_label, **plot_kwargs)
            if integral_ax is not None:
                scaled_cdf = _cdf_from_pdf(pdf) * integral_y_value_scale
                integral_ax.plot(display_grid, scaled_cdf, label=display_label, **plot_kwargs)

        if sample_points is not None:
            sample_series_label, sample_x = sample_points
            sample_x = np.asarray(sample_x, dtype=np.float64) * x_value_scale
            if sample_series_label in series and sample_x.size > 0:
                sample_pdf = np.asarray(series[sample_series_label], dtype=np.float64)
                sample_pdf_y = np.interp(sample_x, display_grid, sample_pdf) * pdf_y_value_scale
                scatter_kwargs = {
                    "s": float(sample_cfg.get("size", 34.0)),
                    "facecolors": sample_cfg.get("facecolors", "white"),
                    "edgecolors": sample_cfg.get("edgecolors", "black"),
                    "linewidths": float(sample_cfg.get("linewidths", 0.9)),
                    "zorder": 5,
                    "label": sample_cfg.get("label", "sampled starts"),
                }
                pdf_ax.scatter(sample_x, sample_pdf_y, **scatter_kwargs)
                if integral_ax is not None:
                    sample_cdf_y = (
                        np.interp(sample_x, display_grid, _cdf_from_pdf(sample_pdf))
                        * integral_y_value_scale
                    )
                    integral_scatter_kwargs = dict(scatter_kwargs)
                    if legend_target_ax is not integral_ax:
                        integral_scatter_kwargs.pop("label", None)
                    else:
                        legend_source_ax = pdf_ax
                    integral_ax.scatter(sample_x, sample_cdf_y, **integral_scatter_kwargs)

        pdf_ylabel = str(dist_cfg.get("pdf_ylabel", "PDF"))
        pdf_xlabel = None if integral_ax is not None else str(dist_cfg.get("xlabel", xlabel))
        _apply_axis_grid(pdf_ax, style, default_linestyle="-", default_alpha=0.3)
        _apply_axis_overrides(
            pdf_ax,
            style,
            "distribution",
            "pdf",
            xlabel=pdf_xlabel,
            ylabel=pdf_ylabel,
            x_value_scale=x_value_scale,
            y_value_scale=pdf_y_value_scale,
            x_unit_label=x_unit_label,
        )

        if integral_ax is not None:
            _apply_axis_grid(integral_ax, style, default_linestyle="-", default_alpha=0.3)
            _apply_axis_overrides(
                integral_ax,
                style,
                "distribution",
                "integral",
                xlabel=str(dist_cfg.get("xlabel", xlabel)),
                ylabel=str(dist_cfg.get("integral_ylabel", "CDF")),
                x_value_scale=x_value_scale,
                y_value_scale=integral_y_value_scale,
                x_unit_label=x_unit_label,
            )

        legend_handles, legend_labels = legend_source_ax.get_legend_handles_labels()
        if legend_handles:
            legend_target_ax.legend(
                legend_handles,
                legend_labels,
                **_legend_kwargs(style, default_frameon=False),
            )

        final_title = title or dist_cfg.get("title")
        if not _titles_enabled(style, "distribution"):
            final_title = None
        if final_title:
            fig.suptitle(final_title)

        _apply_figure_layout(fig, style, "distribution", has_suptitle=bool(final_title))
        if save_path is not None:
            fig.savefig(save_path, dpi=int(style.get("figure", {}).get("dpi", 160)) if style else 160)
            print(f"  Comparison plot → {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Tetrahedral coordination analysis
# ---------------------------------------------------------------------------

def _plot_q4(
    time_arr,
    q4_arr,
    dist_arr,
    sites,
    q_bins,
    q_dist_dict,
    d_bins,
    per_type_dict,
    save_plot,
    show_plot,
    time_unit: str = "ns",
    style: dict | None = None,
    section: str = "q4",
    value_label: str = "Q4",
    timeseries_title: str = "Tetrahedral Order Parameter",
    distribution_title: str = "Q4 Distribution",
):
    """Generate coordination order-parameter and distance diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed — skipping plots.")
        return

    n_sites = len(sites)
    coord_cfg = style.get(section, {}) if isinstance(style, dict) else {}
    coord_titles = coord_cfg.get("titles", {}) if isinstance(coord_cfg.get("titles", {}), dict) else {}
    visible_panels = _resolve_coordination_visible_panels(style, section, value_label)
    coord_layout = coord_cfg.get("layout", {}) if isinstance(coord_cfg.get("layout"), dict) else {}
    n_panels = len(visible_panels)
    ncols = int(coord_layout.get("ncols", 2 if n_panels > 1 else 1))
    ncols = max(1, min(ncols, n_panels))
    nrows = int(math.ceil(n_panels / ncols))

    with plt.rc_context(rc=_style_rc_params(style)):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=_resolve_figsize(style, (14.0, 10.0), section=section),
            squeeze=False,
        )
        flat_axes = list(np.ravel(axes))
        panel_axes = {panel: flat_axes[idx] for idx, panel in enumerate(visible_panels)}
        for extra_ax in flat_axes[len(visible_panels):]:
            extra_ax.set_visible(False)

        if "timeseries" in panel_axes:
            ax = panel_axes["timeseries"]
            y_value_scale = _resolve_panel_value_scale(style, section, "timeseries", axis="y")
            for site_idx in range(n_sites):
                label = f"site{site_idx}"
                display_label, plot_kwargs = _resolve_series_style(style, label, site_idx)
                plot_kwargs.setdefault("linewidth", 0.8)
                plot_kwargs.setdefault("alpha", 0.8)
                ax.plot(
                    time_arr,
                    np.asarray(q4_arr[site_idx], dtype=np.float64) * y_value_scale,
                    label=display_label,
                    **plot_kwargs,
                )
            ax.legend(**_legend_kwargs(style, default_frameon=False))
            _apply_axis_grid(ax, style)
            _apply_axis_overrides(
                ax,
                style,
                section,
                "timeseries",
                xlabel=f"Time ({time_unit})",
                ylabel=str(coord_cfg.get("timeseries_ylabel", value_label)),
                title=str(coord_titles.get("timeseries", timeseries_title)),
                y_value_scale=y_value_scale,
            )

        if "distribution" in panel_axes:
            ax = panel_axes["distribution"]
            y_value_scale = _resolve_panel_value_scale(style, section, "distribution", axis="y")
            for idx, (label, pdf) in enumerate(q_dist_dict.items()):
                display_label, plot_kwargs = _resolve_series_style(style, label, idx)
                plot_kwargs.setdefault("linewidth", 1.5)
                ax.plot(
                    q_bins,
                    np.asarray(pdf, dtype=np.float64) * y_value_scale,
                    label=display_label,
                    **plot_kwargs,
                )
            ax.legend(**_legend_kwargs(style, default_frameon=False))
            _apply_axis_grid(ax, style)
            _apply_axis_overrides(
                ax,
                style,
                section,
                "distribution",
                xlabel=str(coord_cfg.get("distribution_xlabel", value_label.lower())),
                ylabel=str(coord_cfg.get("distribution_ylabel", f"P({value_label.lower()})")),
                title=str(coord_titles.get("distribution", distribution_title)),
                y_value_scale=y_value_scale,
            )

        if "distance_timeseries" in panel_axes:
            ax = panel_axes["distance_timeseries"]
            y_value_scale = _resolve_panel_value_scale(style, section, "distance_timeseries", axis="y")
            site_idx = 0
            _, _, _metal_name, ligand_names = sites[site_idx]
            for lig_idx, ligand_name in enumerate(ligand_names):
                display_label, plot_kwargs = _resolve_series_style(style, ligand_name, lig_idx)
                plot_kwargs.setdefault("linewidth", 0.8)
                plot_kwargs.setdefault("alpha", 0.8)
                ax.plot(
                    time_arr,
                    np.asarray(dist_arr[site_idx][lig_idx], dtype=np.float64) * 10.0 * y_value_scale,
                    label=display_label,
                    **plot_kwargs,
                )
            ax.legend(**_legend_kwargs(style, default_frameon=False))
            _apply_axis_grid(ax, style)
            _apply_axis_overrides(
                ax,
                style,
                section,
                "distance_timeseries",
                xlabel=f"Time ({time_unit})",
                ylabel=str(coord_cfg.get("distance_timeseries_ylabel", "Distance (A)")),
                title=str(coord_titles.get("distance_timeseries", "Metal-Ligand Distances (site 0)")),
                y_value_scale=y_value_scale,
            )

        if "distance_distribution" in panel_axes:
            ax = panel_axes["distance_distribution"]
            y_value_scale = _resolve_panel_value_scale(style, section, "distance_distribution", axis="y")
            for idx, (label, pdf) in enumerate(per_type_dict.items()):
                display_label, plot_kwargs = _resolve_series_style(style, label, idx)
                plot_kwargs.setdefault("linewidth", 1.5)
                ax.plot(
                    d_bins,
                    np.asarray(pdf, dtype=np.float64) * y_value_scale,
                    label=display_label,
                    **plot_kwargs,
                )
            ax.legend(**_legend_kwargs(style, default_frameon=False))
            _apply_axis_grid(ax, style)
            _apply_axis_overrides(
                ax,
                style,
                section,
                "distance_distribution",
                xlabel=str(coord_cfg.get("distance_distribution_xlabel", "r (nm)")),
                ylabel=str(coord_cfg.get("distance_distribution_ylabel", "P(r)")),
                title=str(coord_titles.get("distance_distribution", "Coordination Distance Distributions")),
                y_value_scale=y_value_scale,
            )

        _apply_figure_layout(fig, style, section)
        if save_plot:
            fig.savefig(
                save_plot,
                dpi=int(style.get("figure", {}).get("dpi", 200)) if style else 200,
                bbox_inches="tight",
            )
            print(f"  Plot saved      → {save_plot}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _run_q4_analysis(args) -> int:
    """Execute the tetrahedral Q4 / coordination-distance analysis."""
    if not args.file:
        raise SystemExit("ERROR: Tetrahedral analysis requires -f / --file.")
    input_path = Path(args.file)
    output_prefix = Path(args.output) if args.output else Path(input_path.stem)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    is_gromacs = suffix in {".xtc", ".trr", ".gro"}
    is_h5 = suffix in {".h5", ".hdf5"}

    if is_gromacs and not args.topology:
        raise SystemExit(
            "ERROR: GROMACS trajectory requires a topology file (-s / --topology)."
        )

    sites = []
    if args.site:
        for site in args.site:
            metal_idx = site[0]
            ligand_indices = list(site[1:])
            sites.append(
                (
                    metal_idx,
                    ligand_indices,
                    f"atom{metal_idx}",
                    [f"atom{i}" for i in ligand_indices],
                )
            )

    if args.metal and args.ligands:
        if not is_gromacs:
            raise SystemExit(
                "ERROR: --metal/--ligands name-based selection requires a GROMACS "
                "trajectory with topology (-s)."
            )
        mid, lids, mname, lnames = _select_atoms_mda(
            args.topology,
            input_path,
            args.metal,
            args.ligands,
        )
        sites.append((mid, lids, mname, lnames))

    if not sites:
        raise SystemExit(
            "ERROR: No coordination site specified. Use --site or --metal/--ligands."
        )

    if is_gromacs:
        frame_iter = _iter_mda_frames(
            args.topology,
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    elif is_h5:
        frame_iter = _iter_h5_frames(
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    else:
        raise SystemExit(f"ERROR: Unsupported trajectory format '{suffix}'.")

    print("=" * 65)
    print("  diffmd-analyze — Tetrahedral Order Parameter (Q4)")
    print("=" * 65)
    print(f"  Trajectory:   {input_path}")
    if args.topology:
        print(f"  Topology:     {args.topology}")
    for site_idx, (mid, lids, metal_name, ligand_names) in enumerate(sites):
        print(
            f"  Site {site_idx}: metal {metal_name} (idx {mid})  "
            f"→ ligands {', '.join(ligand_names)} (idx {', '.join(map(str, lids))})"
        )
    print("-" * 65)

    n_sites = len(sites)
    time_list = []
    frame_index_list = []
    q4_data = [[] for _ in range(n_sites)]
    dist_data = [[[] for _ in range(4)] for _ in range(n_sites)]

    n_frames = 0
    for frame_idx, time_ns, pos, box in frame_iter:
        time_list.append(time_ns)
        frame_index_list.append(frame_idx)
        for site_idx, (mid, lids, _mname, _lnames) in enumerate(sites):
            center = pos[mid]
            neighbors = pos[lids]
            q_val = tetrahedral_q(center, neighbors, box)
            q4_data[site_idx].append(q_val)
            dists = site_distances(center, neighbors, box)
            for lig_idx in range(4):
                dist_data[site_idx][lig_idx].append(dists[lig_idx])
        n_frames += 1

    if n_frames == 0:
        raise SystemExit("ERROR: No valid frames found in trajectory.")

    time_arr = np.asarray(time_list) * _TIME_FACTORS[args.time_unit]
    trajectory_frame_indices = np.asarray(frame_index_list, dtype=np.int64)
    q4_arr = [np.asarray(values, dtype=np.float64) for values in q4_data]
    dist_arr = [
        [np.asarray(values, dtype=np.float64) for values in site]
        for site in dist_data
    ]

    sample_series = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        sample_series[f"q4_site{site_idx}"] = q4_arr[site_idx]
        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            sample_series[f"d_lig{lig_idx}_site{site_idx}_nm"] = dist_arr[site_idx][lig_idx]
            type_groups.setdefault(ligand_name, []).append(lig_idx)
        for ligand_type, indices in type_groups.items():
            stacked = np.vstack([dist_arr[site_idx][idx] for idx in indices])
            sample_series[f"site{site_idx}_{ligand_type}"] = np.mean(stacked, axis=0)

    print(
        f"  Processed {n_frames} frames ({time_arr[0]:.3f} – {time_arr[-1]:.3f} {args.time_unit})"
    )
    print()

    for site_idx, (_mid, _lids, metal_name, ligand_names) in enumerate(sites):
        print(f"  Site {site_idx}: {metal_name}")
        print(
            f"    Q4:  {np.mean(q4_arr[site_idx]):.4f} ± {np.std(q4_arr[site_idx]):.4f}"
            f"  (min {np.min(q4_arr[site_idx]):.4f}, max {np.max(q4_arr[site_idx]):.4f})"
        )

        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            type_groups.setdefault(ligand_name, []).append(lig_idx)

        for ligand_type, indices in type_groups.items():
            all_dists_nm = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
            mean_d = np.mean(all_dists_nm)
            std_d = np.std(all_dists_nm)
            print(
                f"    Dist {ligand_type} ({len(indices)} ligands):  "
                f"{mean_d:.4f} ± {std_d:.4f} nm  "
                f"({mean_d * 10:.3f} ± {std_d * 10:.3f} A)"
            )
        print()

    ts_path = Path(f"{output_prefix}_q4_timeseries.dat")
    header_parts = [f"time_{args.time_unit}"]
    columns = [time_arr]
    for site_idx in range(n_sites):
        header_parts.append(f"q4_site{site_idx}")
        columns.append(q4_arr[site_idx])
        for lig_idx in range(4):
            header_parts.append(f"d_lig{lig_idx}_site{site_idx}_nm")
            columns.append(dist_arr[site_idx][lig_idx])
    np.savetxt(ts_path, np.column_stack(columns), header="  ".join(header_parts), fmt="%.6f")
    print(f"  Time series     → {ts_path}")

    q_bins = np.linspace(args.q_range[0], args.q_range[1], args.nbins)
    q_dist_dict = {}
    for site_idx in range(n_sites):
        q_dist_dict[f"site{site_idx}"] = _compute_kde(
            q4_arr[site_idx],
            q_bins,
            bw_factor=args.bw,
            normalize=args.normalize,
        )
    q_xvg = Path(f"{output_prefix}_q4_dist.xvg")
    q_npy = Path(f"{output_prefix}_q4_dist.npy")
    _write_xvg(q_xvg, q_bins, q_dist_dict, "Q4 distribution", "q", "P(q)")
    _write_distribution_npy(q_npy, q_bins, q_dist_dict)
    print(f"  Q4 distribution → {q_xvg}")
    print(f"  Q4 NPY          → {q_npy}")

    all_dists_flat = np.concatenate(
        [dist_arr[site_idx][lig_idx] for site_idx in range(n_sites) for lig_idx in range(4)]
    )
    if args.dist_range:
        d_lo, d_hi = args.dist_range
    else:
        d_lo = max(0.0, float(np.min(all_dists_flat)) - 0.05)
        d_hi = float(np.max(all_dists_flat)) + 0.05
    d_bins = np.linspace(d_lo, d_hi, args.nbins)

    per_lig_dict = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        for lig_idx, ligand_name in enumerate(ligand_names):
            label = f"site{site_idx}_{ligand_name}_lig{lig_idx}"
            per_lig_dict[label] = _compute_kde(
                dist_arr[site_idx][lig_idx],
                d_bins,
                bw_factor=args.bw,
                normalize=args.normalize,
            )
    per_lig_xvg = Path(f"{output_prefix}_coord_dist_per_ligand.xvg")
    _write_xvg(
        per_lig_xvg,
        d_bins,
        per_lig_dict,
        "Per-ligand coordination distances",
        "r (nm)",
        "P(r)",
    )
    print(f"  Per-ligand dist → {per_lig_xvg}")

    per_type_dict = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            type_groups.setdefault(ligand_name, []).append(lig_idx)
        for ligand_type, indices in type_groups.items():
            all_dists = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
            label = f"site{site_idx}_{ligand_type}"
            per_type_dict[label] = _compute_kde(
                all_dists,
                d_bins,
                bw_factor=args.bw,
                normalize=args.normalize,
            )
    per_type_xvg = Path(f"{output_prefix}_coord_dist.xvg")
    per_type_npy = Path(f"{output_prefix}_coord_dist.npy")
    _write_xvg(
        per_type_xvg,
        d_bins,
        per_type_dict,
        "Per-type coordination distances",
        "r (nm)",
        "P(r)",
    )
    _write_distribution_npy(per_type_npy, d_bins, per_type_dict)
    print(f"  Per-type dist   → {per_type_xvg}")
    print(f"  Coord NPY       → {per_type_npy}")

    mean_dist_path = Path(f"{output_prefix}_mean_distances.dat")
    with mean_dist_path.open("w", encoding="utf-8") as handle:
        handle.write("# site  ligand_type  mean_nm  std_nm  mean_A  std_A  n_ligands\n")
        for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
            type_groups = {}
            for lig_idx, ligand_name in enumerate(ligand_names):
                type_groups.setdefault(ligand_name, []).append(lig_idx)
            for ligand_type, indices in type_groups.items():
                all_dists = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
                handle.write(
                    f"{site_idx}  {ligand_type:>15s}  {np.mean(all_dists):.6f}  {np.std(all_dists):.6f}"
                    f"  {np.mean(all_dists) * 10:.4f}  {np.std(all_dists) * 10:.4f}  {len(indices)}\n"
                )
    print(f"  Mean distances  → {mean_dist_path}")

    sampled_manifest = None
    if args.sample_frames is not None:
        try:
            sample_label, sample_values = _resolve_sample_series(
                sample_series,
                args.sample_series,
            )
            sampled_analysis_indices = _select_uniform_frame_indices(
                sample_values,
                args.sample_frames,
                observable_label=sample_label,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

        sample_output_dir = (
            Path(args.sample_output_dir)
            if args.sample_output_dir
            else Path(f"{output_prefix}_samples")
        )
        sampled_manifest = _export_sampled_structures(
            input_path,
            sampled_analysis_indices,
            trajectory_frame_indices,
            time_arr,
            args.time_unit,
            sample_output_dir,
            sample_label,
            sample_series,
            source_kind="h5" if is_h5 else "mda",
            template_h5=args.sample_template_h5 if is_h5 else None,
            topology_path=args.topology if is_gromacs else None,
        )
        print(f"  Sample series   → {sample_label}")
        print(f"  Sampled starts  → {sample_output_dir}")
        print(f"  Sample manifest → {sample_output_dir / 'manifest.json'}")

    if args.plot or args.save_plot:
        _plot_q4(
            time_arr,
            q4_arr,
            dist_arr,
            sites,
            q_bins,
            q_dist_dict,
            d_bins,
            per_type_dict,
            args.save_plot,
            args.plot,
            time_unit=args.time_unit,
            style=args.plot_style,
        )

    print()
    print("Done.")
    return 0


def _run_q5_analysis(args) -> int:
    """Execute the trigonal-bipyramidal tau5 / coordination-distance analysis."""
    if not args.file:
        raise SystemExit("ERROR: Q5 analysis requires -f / --file.")
    input_path = Path(args.file)
    output_prefix = Path(args.output) if args.output else Path(input_path.stem)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    is_gromacs = suffix in {".xtc", ".trr", ".gro"}
    is_h5 = suffix in {".h5", ".hdf5"}

    if is_gromacs and not args.topology:
        raise SystemExit(
            "ERROR: GROMACS trajectory requires a topology file (-s / --topology)."
        )

    sites = []
    if args.site5:
        for site in args.site5:
            metal_idx = site[0]
            ligand_indices = list(site[1:])
            sites.append(
                (
                    metal_idx,
                    ligand_indices,
                    f"atom{metal_idx}",
                    [f"atom{i}" for i in ligand_indices],
                )
            )

    if args.metal and args.ligands:
        if not is_gromacs:
            raise SystemExit(
                "ERROR: --metal/--ligands name-based selection requires a GROMACS "
                "trajectory with topology (-s)."
            )
        mid, lids, mname, lnames = _select_atoms_mda(
            args.topology,
            input_path,
            args.metal,
            args.ligands,
            expected_ligands=5,
        )
        sites.append((mid, lids, mname, lnames))

    if not sites:
        raise SystemExit(
            "ERROR: No five-coordinate site specified. Use --site5 or --metal/--ligands."
        )

    if is_gromacs:
        frame_iter = _iter_mda_frames(
            args.topology,
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    elif is_h5:
        frame_iter = _iter_h5_frames(
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    else:
        raise SystemExit(f"ERROR: Unsupported trajectory format '{suffix}'.")

    print("=" * 65)
    print("  diffmd-analyze — Trigonal Bipyramidal Coordination (tau5)")
    print("=" * 65)
    print(f"  Trajectory:   {input_path}")
    if args.topology:
        print(f"  Topology:     {args.topology}")
    for site_idx, (mid, lids, metal_name, ligand_names) in enumerate(sites):
        print(
            f"  Site {site_idx}: metal {metal_name} (idx {mid})  "
            f"→ ligands {', '.join(ligand_names)} (idx {', '.join(map(str, lids))})"
        )
    print("-" * 65)

    n_sites = len(sites)
    time_list = []
    frame_index_list = []
    q5_data = [[] for _ in range(n_sites)]
    dist_data = [[[] for _ in range(5)] for _ in range(n_sites)]

    n_frames = 0
    for frame_idx, time_ns, pos, box in frame_iter:
        time_list.append(time_ns)
        frame_index_list.append(frame_idx)
        for site_idx, (mid, lids, _mname, _lnames) in enumerate(sites):
            center = pos[mid]
            neighbors = pos[lids]
            q_val = trigonal_bipyramidal_tau5(center, neighbors, box)
            q5_data[site_idx].append(q_val)
            dists = site_distances(center, neighbors, box)
            for lig_idx in range(5):
                dist_data[site_idx][lig_idx].append(dists[lig_idx])
        n_frames += 1

    if n_frames == 0:
        raise SystemExit("ERROR: No valid frames found in trajectory.")

    time_arr = np.asarray(time_list) * _TIME_FACTORS[args.time_unit]
    trajectory_frame_indices = np.asarray(frame_index_list, dtype=np.int64)
    q5_arr = [np.asarray(values, dtype=np.float64) for values in q5_data]
    dist_arr = [
        [np.asarray(values, dtype=np.float64) for values in site]
        for site in dist_data
    ]

    sample_series = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        sample_series[f"q5_site{site_idx}"] = q5_arr[site_idx]
        sample_series[f"tau5_site{site_idx}"] = q5_arr[site_idx]
        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            sample_series[f"d_lig{lig_idx}_site{site_idx}_nm"] = dist_arr[site_idx][lig_idx]
            type_groups.setdefault(ligand_name, []).append(lig_idx)
        for ligand_type, indices in type_groups.items():
            stacked = np.vstack([dist_arr[site_idx][idx] for idx in indices])
            sample_series[f"site{site_idx}_{ligand_type}"] = np.mean(stacked, axis=0)

    print(
        f"  Processed {n_frames} frames ({time_arr[0]:.3f} – {time_arr[-1]:.3f} {args.time_unit})"
    )
    print()

    for site_idx, (_mid, _lids, metal_name, ligand_names) in enumerate(sites):
        print(f"  Site {site_idx}: {metal_name}")
        print(
            f"    tau5: {np.mean(q5_arr[site_idx]):.4f} ± {np.std(q5_arr[site_idx]):.4f}"
            f"  (min {np.min(q5_arr[site_idx]):.4f}, max {np.max(q5_arr[site_idx]):.4f})"
        )

        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            type_groups.setdefault(ligand_name, []).append(lig_idx)

        for ligand_type, indices in type_groups.items():
            all_dists_nm = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
            mean_d = np.mean(all_dists_nm)
            std_d = np.std(all_dists_nm)
            print(
                f"    Dist {ligand_type} ({len(indices)} ligands):  "
                f"{mean_d:.4f} ± {std_d:.4f} nm  "
                f"({mean_d * 10:.3f} ± {std_d * 10:.3f} A)"
            )
        print()

    ts_path = Path(f"{output_prefix}_q5_timeseries.dat")
    header_parts = [f"time_{args.time_unit}"]
    columns = [time_arr]
    for site_idx in range(n_sites):
        header_parts.append(f"tau5_site{site_idx}")
        columns.append(q5_arr[site_idx])
        for lig_idx in range(5):
            header_parts.append(f"d_lig{lig_idx}_site{site_idx}_nm")
            columns.append(dist_arr[site_idx][lig_idx])
    np.savetxt(ts_path, np.column_stack(columns), header="  ".join(header_parts), fmt="%.6f")
    print(f"  Time series     → {ts_path}")

    q_bins = np.linspace(args.q5_range[0], args.q5_range[1], args.nbins)
    q_dist_dict = {}
    for site_idx in range(n_sites):
        q_dist_dict[f"site{site_idx}"] = _compute_kde(
            q5_arr[site_idx],
            q_bins,
            bw_factor=args.bw,
            normalize=args.normalize,
        )
    q_xvg = Path(f"{output_prefix}_q5_dist.xvg")
    q_npy = Path(f"{output_prefix}_q5_dist.npy")
    _write_xvg(q_xvg, q_bins, q_dist_dict, "tau5 distribution", "tau5", "P(tau5)")
    _write_distribution_npy(q_npy, q_bins, q_dist_dict)
    print(f"  Q5 distribution → {q_xvg}")
    print(f"  Q5 NPY          → {q_npy}")

    all_dists_flat = np.concatenate(
        [dist_arr[site_idx][lig_idx] for site_idx in range(n_sites) for lig_idx in range(5)]
    )
    if args.dist_range:
        d_lo, d_hi = args.dist_range
    else:
        d_lo = max(0.0, float(np.min(all_dists_flat)) - 0.05)
        d_hi = float(np.max(all_dists_flat)) + 0.05
    d_bins = np.linspace(d_lo, d_hi, args.nbins)

    per_lig_dict = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        for lig_idx, ligand_name in enumerate(ligand_names):
            label = f"site{site_idx}_{ligand_name}_lig{lig_idx}"
            per_lig_dict[label] = _compute_kde(
                dist_arr[site_idx][lig_idx],
                d_bins,
                bw_factor=args.bw,
                normalize=args.normalize,
            )
    per_lig_xvg = Path(f"{output_prefix}_coord_dist_per_ligand_q5.xvg")
    _write_xvg(
        per_lig_xvg,
        d_bins,
        per_lig_dict,
        "Per-ligand five-coordinate distances",
        "r (nm)",
        "P(r)",
    )
    print(f"  Per-ligand dist → {per_lig_xvg}")

    per_type_dict = {}
    for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
        type_groups = {}
        for lig_idx, ligand_name in enumerate(ligand_names):
            type_groups.setdefault(ligand_name, []).append(lig_idx)
        for ligand_type, indices in type_groups.items():
            all_dists = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
            label = f"site{site_idx}_{ligand_type}"
            per_type_dict[label] = _compute_kde(
                all_dists,
                d_bins,
                bw_factor=args.bw,
                normalize=args.normalize,
            )
    per_type_xvg = Path(f"{output_prefix}_coord_dist_q5.xvg")
    per_type_npy = Path(f"{output_prefix}_coord_dist_q5.npy")
    _write_xvg(
        per_type_xvg,
        d_bins,
        per_type_dict,
        "Per-type five-coordinate distances",
        "r (nm)",
        "P(r)",
    )
    _write_distribution_npy(per_type_npy, d_bins, per_type_dict)
    print(f"  Per-type dist   → {per_type_xvg}")
    print(f"  Coord NPY       → {per_type_npy}")

    mean_dist_path = Path(f"{output_prefix}_mean_distances_q5.dat")
    with mean_dist_path.open("w", encoding="utf-8") as handle:
        handle.write("# site  ligand_type  mean_nm  std_nm  mean_A  std_A  n_ligands\n")
        for site_idx, (_mid, _lids, _mname, ligand_names) in enumerate(sites):
            type_groups = {}
            for lig_idx, ligand_name in enumerate(ligand_names):
                type_groups.setdefault(ligand_name, []).append(lig_idx)
            for ligand_type, indices in type_groups.items():
                all_dists = np.concatenate([dist_arr[site_idx][idx] for idx in indices])
                handle.write(
                    f"{site_idx}  {ligand_type:>15s}  {np.mean(all_dists):.6f}  {np.std(all_dists):.6f}"
                    f"  {np.mean(all_dists) * 10:.4f}  {np.std(all_dists) * 10:.4f}  {len(indices)}\n"
                )
    print(f"  Mean distances  → {mean_dist_path}")

    sampled_manifest = None
    if args.sample_frames is not None:
        try:
            sample_label, sample_values = _resolve_sample_series(
                sample_series,
                args.sample_series,
            )
            sampled_analysis_indices = _select_uniform_frame_indices(
                sample_values,
                args.sample_frames,
                observable_label=sample_label,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

        sample_output_dir = (
            Path(args.sample_output_dir)
            if args.sample_output_dir
            else Path(f"{output_prefix}_samples")
        )
        sampled_manifest = _export_sampled_structures(
            input_path,
            sampled_analysis_indices,
            trajectory_frame_indices,
            time_arr,
            args.time_unit,
            sample_output_dir,
            sample_label,
            sample_series,
            source_kind="h5" if is_h5 else "mda",
            template_h5=args.sample_template_h5 if is_h5 else None,
            topology_path=args.topology if is_gromacs else None,
        )
        print(f"  Sample series   → {sample_label}")
        print(f"  Sampled starts  → {sample_output_dir}")
        print(f"  Sample manifest → {sample_output_dir / 'manifest.json'}")

    if args.plot or args.save_plot:
        _plot_q4(
            time_arr,
            q5_arr,
            dist_arr,
            sites,
            q_bins,
            q_dist_dict,
            d_bins,
            per_type_dict,
            args.save_plot,
            args.plot,
            time_unit=args.time_unit,
            style=args.plot_style,
            section="q5",
            value_label="tau5",
            timeseries_title="Trigonal Bipyramidal tau5",
            distribution_title="tau5 Distribution",
        )

    print()
    print("Done.")
    return 0


# ---------------------------------------------------------------------------
# Radius of gyration analysis
# ---------------------------------------------------------------------------

def _save_rg_timeseries(
    path: Path,
    time_arr: np.ndarray,
    rg_matrix: np.ndarray,
    time_unit: str,
) -> None:
    header_parts = [f"time_{time_unit}"]
    columns = [time_arr]
    for chain_idx in range(rg_matrix.shape[1]):
        header_parts.append(f"rg_chain{chain_idx}_nm")
        columns.append(rg_matrix[:, chain_idx])
    header_parts.append("rg_mean_nm")
    columns.append(np.mean(rg_matrix, axis=1))
    np.savetxt(path, np.column_stack(columns), header="  ".join(header_parts), fmt="%.6f")


def _save_apl_timeseries(
    path: Path,
    time_arr: np.ndarray,
    apl_series: np.ndarray,
    time_unit: str,
) -> None:
    np.savetxt(
        path,
        np.column_stack([time_arr, apl_series]),
        header=f"time_{time_unit}  apl_nm2",
        fmt="%.6f",
    )


def _default_rg_grid(samples: np.ndarray, nbins: int) -> np.ndarray:
    s_min = float(np.min(samples))
    s_max = float(np.max(samples))
    span = s_max - s_min
    pad = max(0.05 * span, 0.02)
    lo = max(0.0, s_min - pad)
    hi = s_max + pad
    if hi <= lo:
        hi = lo + 0.1
    return np.linspace(lo, hi, nbins)


def _run_distribution_analysis(args) -> int:
    """Compare saved distribution files directly, without a trajectory."""
    sources: list[tuple[str, str]] = []
    if args.reference:
        sources.extend((label, path) for label, path in args.reference)
    if args.series:
        sources.extend((label, path) for label, path in args.series)

    if len(sources) < 2:
        raise SystemExit(
            "ERROR: Distribution comparison requires at least two saved inputs via "
            "--reference and/or --series."
        )

    default_prefix = Path(Path(sources[0][1]).stem)
    output_prefix = Path(args.output) if args.output else default_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.compare_label:
        reference_label = args.compare_label
    elif args.reference:
        reference_label = args.reference[0][0]
    else:
        reference_label = sources[0][0]

    loaded: dict[str, dict[str, np.ndarray | list[str] | None]] = {}
    source_paths: dict[str, str] = {}
    for label, path in sources:
        if label in loaded:
            raise SystemExit(f"ERROR: Duplicate series label '{label}'.")
        loaded[label] = _load_distribution_family(path)
        source_paths[label] = str(path)

    grid = None
    reference_family = loaded.get(reference_label)
    if reference_family is not None and reference_family["grid"] is not None:
        grid = np.asarray(reference_family["grid"], dtype=np.float64)

    if grid is None:
        for family in loaded.values():
            family_grid = family["grid"]
            if family_grid is not None:
                grid = np.asarray(family_grid, dtype=np.float64)
                break

    pdf_counts = {
        label: int(np.asarray(family["pdfs"], dtype=np.float64).shape[0])
        for label, family in loaded.items()
    }
    n_bins_set = {
        int(np.asarray(family["pdfs"], dtype=np.float64).shape[1])
        for family in loaded.values()
    }

    if grid is None:
        if len(n_bins_set) != 1:
            raise SystemExit(
                "ERROR: Saved distributions without embedded grids must all use the same "
                "number of bins and provide --dist-range MIN MAX."
            )
        if args.dist_range is None:
            raise SystemExit(
                "ERROR: Saved distributions without embedded grids require --dist-range MIN MAX."
            )
        n_bins = next(iter(n_bins_set))
        grid = np.linspace(args.dist_range[0], args.dist_range[1], n_bins)

    for label, family in loaded.items():
        pdfs = np.asarray(family["pdfs"], dtype=np.float64)
        family_grid = family["grid"]
        if family_grid is None:
            if pdfs.shape[1] != grid.shape[0]:
                raise SystemExit(
                    f"ERROR: Series '{label}' has {pdfs.shape[1]} bins but expected {grid.shape[0]}."
                )
            continue

        family_grid = np.asarray(family_grid, dtype=np.float64)
        loaded[label]["pdfs"] = np.vstack(
            [_resample_pdf_to_grid(pdf, family_grid, grid) for pdf in pdfs]
        )

    if args.dist_index is not None:
        active_indices = [args.dist_index]
        for label, count in pdf_counts.items():
            if args.dist_index < 0 or args.dist_index >= count:
                raise SystemExit(
                    f"ERROR: dist_index={args.dist_index} is out of range for series '{label}'."
                )
    else:
        unique_counts = set(pdf_counts.values())
        if len(unique_counts) != 1:
            raise SystemExit(
                "ERROR: Comparing all saved distributions requires every series to contain "
                "the same number of PDFs. Use --dist-index to select one row explicitly."
            )
        active_indices = list(range(next(iter(unique_counts))))

    if reference_label not in loaded:
        raise SystemExit(
            f"ERROR: compare label '{reference_label}' does not match any series label."
        )

    label_source = loaded[reference_label].get("labels") or []
    if len(label_source) > max(active_indices):
        distribution_labels = [str(label_source[idx]) for idx in active_indices]
    else:
        distribution_labels = [f"dist{idx}" for idx in active_indices]

    print("=" * 65)
    print("  diffmd-analyze — Saved Distribution Comparison")
    print("=" * 65)
    for label, path in sources:
        print(f"  {label:>12}: {path}")
    print(f"  Baseline:     {reference_label}")
    print(f"  PDFs:         {', '.join(distribution_labels)}")
    print("-" * 65)

    summary = {
        "mode": "distribution",
        "reference_label": reference_label,
        "distribution_indices": [int(idx) for idx in active_indices],
        "distribution_labels": distribution_labels,
        "grid": {
            "count": int(grid.size),
            "min": float(grid[0]),
            "max": float(grid[-1]),
            "bin_size": float(grid[1] - grid[0]) if grid.size > 1 else None,
        },
        "series": {},
    }

    reference_pdfs = np.asarray(loaded[reference_label]["pdfs"], dtype=np.float64)
    for label, family in loaded.items():
        pdfs = np.asarray(family["pdfs"], dtype=np.float64)
        dist_summary = {}
        kl_vals = []
        w1_vals = []
        rmse_vals = []
        for dist_idx, dist_label in zip(active_indices, distribution_labels):
            pdf = np.asarray(pdfs[dist_idx], dtype=np.float64)
            ref_pdf = np.asarray(reference_pdfs[dist_idx], dtype=np.float64)
            info = _pdf_summary(grid, pdf)
            info["kl_to_reference"] = _kl_forward(pdf, ref_pdf)
            info["w1_to_reference"] = _wasserstein_1d(pdf, ref_pdf)
            info["rmse_to_reference"] = _rmse(pdf, ref_pdf)
            dist_summary[dist_label] = info
            kl_vals.append(info["kl_to_reference"])
            w1_vals.append(info["w1_to_reference"])
            rmse_vals.append(info["rmse_to_reference"])

        summary["series"][label] = {
            "path": source_paths[label],
            "n_distributions": int(pdfs.shape[0]),
            "aggregate": {
                "mean_kl_to_reference": float(np.mean(kl_vals)),
                "mean_w1_to_reference": float(np.mean(w1_vals)),
                "mean_rmse_to_reference": float(np.mean(rmse_vals)),
            },
            "distributions": dist_summary,
        }

    summary_path = Path(f"{output_prefix}_distribution_compare.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Summary       → {summary_path}")

    if len(active_indices) == 1 and (args.save_plot or args.plot):
        plot_path = Path(args.save_plot) if args.save_plot else Path(
            f"{output_prefix}_distribution_compare.png"
        )
        plot_series = {
            label: np.asarray(family["pdfs"], dtype=np.float64)[active_indices[0]]
            for label, family in loaded.items()
        }
        _plot_distribution_comparison(
            grid,
            plot_series,
            title=args.title or f"Saved distribution comparison ({distribution_labels[0]})",
            xlabel="Value",
            save_path=plot_path,
            show_plot=args.plot,
            style=args.plot_style,
        )
    elif args.save_plot or args.plot:
        print(
            "  NOTE: multi-distribution comparison writes JSON metrics for all rows; "
            "use --dist-index to plot one row at a time."
        )

    print()
    print("Series summary:")
    for label, info in summary["series"].items():
        aggregate = info["aggregate"]
        print(
            f"  {label:>22} | mean_KL={aggregate['mean_kl_to_reference']:.6f}"
            f" mean_W1={aggregate['mean_w1_to_reference']:.6f}"
            f" mean_RMSE={aggregate['mean_rmse_to_reference']:.6f}"
        )

    print()
    print("Done.")
    return 0


def _run_density_apl_analysis(args) -> int:
    """Execute loss-compatible membrane density/APL analysis on Diff-MD H5 files."""
    if not args.file:
        raise SystemExit("ERROR: Density/APL analysis requires -f / --file.")
    if args.normalize:
        raise SystemExit(
            "ERROR: --normalize is not supported in --density-apl mode because "
            "density_and_apl uses raw number-density profiles, not normalized PDFs."
        )
    if args.n_lipids is None or args.n_lipids < 1:
        raise SystemExit("ERROR: Density/APL analysis requires --n-lipids >= 1.")
    if args.com_type is None:
        raise SystemExit("ERROR: Density/APL analysis requires --com-type.")

    input_path = Path(args.file)
    output_prefix = Path(args.output) if args.output else Path(input_path.stem)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix not in {".h5", ".hdf5"}:
        raise SystemExit(
            "ERROR: --density-apl currently supports Diff-MD H5 trajectories only, "
            "because it needs the native types/names metadata used by the loss."
        )

    particle_types, type_ids, type_labels, type_counts = _load_h5_density_metadata(input_path)
    try:
        com_type_id = _resolve_type_identifier(args.com_type, type_ids, type_labels)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    grid = _find_reference_grid(args.reference, 0)
    if grid is None:
        if args.dist_range:
            grid = np.linspace(args.dist_range[0], args.dist_range[1], args.nbins)
        else:
            grid = _default_density_grid_from_h5(
                input_path,
                args.nbins,
                start=args.start,
                stop=args.stop,
                stride=args.stride,
            )
    frame_iter = _iter_h5_frames(
        input_path,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )

    density_sum = None
    time_list = []
    frame_index_list = []
    apl_frames = []

    for frame_idx, time_ns, pos, box in frame_iter:
        if box is None:
            raise SystemExit(
                "ERROR: Density/APL analysis requires box information in every frame."
            )
        z_pos = np.mod(np.asarray(pos[:, 2], dtype=np.float64), float(box[2]))
        tail_z = z_pos[particle_types == com_type_id]
        if tail_z.size == 0:
            raise SystemExit(
                f"ERROR: com_type '{args.com_type}' resolved to id {com_type_id} but no particles matched."
            )

        com = _compute_membrane_com(tail_z, float(box[2]))
        centered_z = _center_z_positions(z_pos, com, float(box[2]))
        bandwidth = float(args.bw * (grid[1] - grid[0]))
        frame_density, apl = _frame_density_and_apl(
            centered_z,
            particle_types,
            type_ids,
            type_counts,
            grid,
            bandwidth,
            np.asarray(box, dtype=np.float64),
            args.n_lipids,
        )

        density_sum = frame_density if density_sum is None else density_sum + frame_density
        apl_frames.append(apl)
        time_list.append(time_ns)
        frame_index_list.append(frame_idx)

    if density_sum is None or not apl_frames:
        raise SystemExit("ERROR: No valid frames found in trajectory.")

    time_arr = np.asarray(time_list, dtype=np.float64) * _TIME_FACTORS[args.time_unit]
    trajectory_frame_indices = np.asarray(frame_index_list, dtype=np.int64)
    apl_series = np.asarray(apl_frames, dtype=np.float64)
    replay_density = density_sum / float(len(apl_series))
    sample_series = {"apl_nm2": apl_series}

    family_series = {
        "replay_fixed": {
            "grid": np.asarray(grid, dtype=np.float64),
            "pdfs": replay_density,
            "labels": list(type_labels),
            "path": str(input_path),
        }
    }
    if args.reference:
        for label, ref_path in args.reference:
            if label in family_series:
                raise SystemExit(f"ERROR: Duplicate series label '{label}'.")
            family = _load_distribution_family(ref_path)
            ref_grid = family["grid"]
            if ref_grid is not None:
                ref_grid = np.asarray(ref_grid, dtype=np.float64)
                if ref_grid.shape != grid.shape or not np.allclose(ref_grid, grid):
                    raise SystemExit(
                        f"ERROR: Reference '{ref_path}' uses a different grid from the analysis grid."
                    )
            pdfs = np.asarray(family["pdfs"], dtype=np.float64)
            if pdfs.shape[1] != grid.shape[0]:
                raise SystemExit(
                    f"ERROR: Reference '{ref_path}' has {pdfs.shape[1]} bins but expected {grid.shape[0]}."
                )
            family_series[label] = {
                "grid": np.asarray(grid, dtype=np.float64),
                "pdfs": pdfs,
                "labels": list(family.get("labels") or []),
                "path": str(ref_path),
            }

    active_count = replay_density.shape[0]
    if args.dist_index is not None:
        if args.dist_index < 0 or args.dist_index >= active_count:
            raise SystemExit(
                f"ERROR: dist_index={args.dist_index} is out of range for {active_count} density rows."
            )
        active_indices = [args.dist_index]
    else:
        active_indices = list(range(active_count))

    for label, family in family_series.items():
        pdfs = np.asarray(family["pdfs"], dtype=np.float64)
        if pdfs.shape[0] < max(active_indices) + 1:
            raise SystemExit(
                f"ERROR: Series '{label}' only contains {pdfs.shape[0]} density rows, "
                f"cannot access indices {active_indices}."
            )
        if args.dist_index is None and pdfs.shape[0] != active_count:
            raise SystemExit(
                f"ERROR: Series '{label}' contains {pdfs.shape[0]} density rows but "
                f"the trajectory contains {active_count} type rows."
            )

    reference_label = None
    if args.reference:
        default_reference = args.reference[0][0]
        reference_label = args.compare_label or default_reference
        if reference_label not in family_series:
            raise SystemExit(
                f"ERROR: compare label '{reference_label}' does not match any series label."
            )

    ref_labels = []
    if reference_label is not None:
        ref_labels = list(family_series[reference_label].get("labels") or [])
    if len(ref_labels) > max(active_indices, default=-1):
        density_labels = [str(ref_labels[idx]) for idx in active_indices]
    else:
        density_labels = [type_labels[idx] for idx in active_indices]

    apl_path = Path(f"{output_prefix}_apl_timeseries.dat")
    density_xvg = Path(f"{output_prefix}_density.xvg")
    density_npy = Path(f"{output_prefix}_density.npy")
    summary_path = (
        Path(f"{output_prefix}_density_compare.json")
        if reference_label is not None
        else Path(f"{output_prefix}_density_apl_summary.json")
    )

    _save_apl_timeseries(apl_path, time_arr, apl_series, args.time_unit)
    _write_xvg(
        density_xvg,
        grid,
        {label: replay_density[idx] for idx, label in enumerate(type_labels)},
        "Lateral density profile",
        "Relative position from center (nm)",
        "Number density (nm^-3)",
    )
    _write_distribution_npy(
        density_npy,
        grid,
        {label: replay_density[idx] for idx, label in enumerate(type_labels)},
    )

    summary = {
        "mode": "density_and_apl",
        "trajectory": str(input_path),
        "com_type": str(args.com_type),
        "com_type_id": int(com_type_id),
        "n_lipids": int(args.n_lipids),
        "type_ids": [int(type_id) for type_id in type_ids],
        "type_labels": list(type_labels),
        "reference_label": reference_label,
        "grid": {
            "count": int(grid.size),
            "min": float(grid[0]),
            "max": float(grid[-1]),
            "bin_size": float(grid[1] - grid[0]),
        },
        "apl": {
            "count": int(apl_series.size),
            "mean": float(np.mean(apl_series)),
            "std": float(np.std(apl_series)),
            "min": float(np.min(apl_series)),
            "max": float(np.max(apl_series)),
        },
        "series": {},
    }
    if args.apl_reference is not None:
        summary["apl"]["reference"] = float(args.apl_reference)
        summary["apl"]["abs_error_to_reference"] = float(abs(np.mean(apl_series) - args.apl_reference))

    reference_pdfs = None
    if reference_label is not None:
        reference_pdfs = np.asarray(family_series[reference_label]["pdfs"], dtype=np.float64)

    for label, family in family_series.items():
        pdfs = np.asarray(family["pdfs"], dtype=np.float64)
        dist_summary = {}
        kl_vals = []
        w1_vals = []
        rmse_vals = []
        for density_idx, density_label in zip(active_indices, density_labels):
            pdf = np.asarray(pdfs[density_idx], dtype=np.float64)
            info = _pdf_summary(grid, pdf)
            if reference_pdfs is not None:
                ref_pdf = np.asarray(reference_pdfs[density_idx], dtype=np.float64)
                info["kl_to_reference"] = _kl_forward(pdf, ref_pdf)
                info["w1_to_reference"] = _wasserstein_1d(pdf, ref_pdf)
                info["rmse_to_reference"] = _rmse(pdf, ref_pdf)
                kl_vals.append(info["kl_to_reference"])
                w1_vals.append(info["w1_to_reference"])
                rmse_vals.append(info["rmse_to_reference"])
            dist_summary[density_label] = info

        aggregate = {}
        if reference_pdfs is not None and kl_vals:
            aggregate = {
                "mean_kl_to_reference": float(np.mean(kl_vals)),
                "mean_w1_to_reference": float(np.mean(w1_vals)),
                "mean_rmse_to_reference": float(np.mean(rmse_vals)),
            }
        summary["series"][label] = {
            "path": family.get("path"),
            "n_distributions": int(pdfs.shape[0]),
            "aggregate": aggregate,
            "distributions": dist_summary,
        }

    sampled_manifest = None
    if args.sample_frames is not None:
        try:
            sample_label, sample_values = _resolve_sample_series(
                sample_series,
                args.sample_series,
                default_label="apl_nm2",
            )
            sampled_analysis_indices = _select_uniform_frame_indices(
                sample_values,
                args.sample_frames,
                observable_label=sample_label,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

        sample_output_dir = (
            Path(args.sample_output_dir)
            if args.sample_output_dir
            else Path(f"{output_prefix}_apl_samples")
        )
        sampled_manifest = _export_sampled_structures(
            input_path,
            sampled_analysis_indices,
            trajectory_frame_indices,
            time_arr,
            args.time_unit,
            sample_output_dir,
            sample_label,
            sample_series,
            source_kind="h5",
            template_h5=args.sample_template_h5,
        )
        summary["sampled_frames"] = sampled_manifest

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 65)
    print("  diffmd-analyze — Membrane Density + Area per Lipid")
    print("=" * 65)
    print(f"  Trajectory:   {input_path}")
    print(f"  COM type:     {args.com_type} (id {com_type_id})")
    print(f"  Type order:   {', '.join(type_labels)}")
    print(f"  n_lipids:     {args.n_lipids}")
    print("-" * 65)
    print(
        f"  Processed {apl_series.size} frames ({time_arr[0]:.3f} – {time_arr[-1]:.3f} {args.time_unit})"
    )
    print(f"  Mean APL:     {np.mean(apl_series):.6f} ± {np.std(apl_series):.6f} nm^2")
    print(f"  APL timeseries→ {apl_path}")
    print(f"  Density XVG   → {density_xvg}")
    print(f"  Density NPY   → {density_npy}")
    if sampled_manifest is not None:
        print(f"  Sample series → {sampled_manifest['sample_series_label']}")
        print(f"  Sampled starts→ {sampled_manifest['output_dir']}")
    print(f"  Summary       → {summary_path}")

    plot_path = None
    if args.save_plot:
        plot_path = Path(args.save_plot)
    elif reference_label is not None and len(active_indices) == 1:
        plot_path = Path(f"{output_prefix}_density_compare.png")

    if plot_path is not None or args.plot:
        if len(active_indices) != 1:
            print(
                "  NOTE: multi-row density comparison writes JSON metrics for all rows; "
                "use --dist-index to plot one row at a time."
            )
        else:
            idx = active_indices[0]
            plot_series = {
                label: np.asarray(family["pdfs"], dtype=np.float64)[idx]
                for label, family in family_series.items()
            }
            _plot_distribution_comparison(
                grid,
                plot_series,
                title=args.title or f"Density comparison ({density_labels[0]})",
                xlabel="Relative position from center (nm)",
                save_path=plot_path,
                show_plot=args.plot,
                style=args.plot_style,
            )

    print()
    print("Series summary:")
    for label, info in summary["series"].items():
        aggregate = info["aggregate"]
        if aggregate:
            print(
                f"  {label:>22} | mean_KL={aggregate['mean_kl_to_reference']:.6f}"
                f" mean_W1={aggregate['mean_w1_to_reference']:.6f}"
                f" mean_RMSE={aggregate['mean_rmse_to_reference']:.6f}"
            )
        else:
            print(f"  {label:>22} | replay/reference family loaded")

    print()
    print("Done.")
    return 0


def _run_rg_analysis(args) -> int:
    """Execute the radius-of-gyration distribution analysis."""
    if not args.file:
        raise SystemExit("ERROR: Rg analysis requires -f / --file.")
    input_path = Path(args.file)
    dist_index = 0 if args.dist_index is None else args.dist_index
    output_prefix = Path(args.output) if args.output else Path(input_path.stem)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    is_gromacs = suffix in {".xtc", ".trr", ".gro"}
    is_h5 = suffix in {".h5", ".hdf5"}

    if is_gromacs and not args.topology:
        raise SystemExit(
            "ERROR: GROMACS trajectory requires a topology file (-s / --topology)."
        )

    if is_h5:
        if not args.resname:
            raise SystemExit(
                "ERROR: H5 Rg analysis requires --resname to select the chain atoms."
            )
        chain_indices, chain_masses, selection_label = _load_h5_rg_selection(
            input_path,
            args.resname,
            args.n_chains,
        )
        frame_iter = _iter_h5_frames(
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    elif is_gromacs:
        if not args.selection:
            raise SystemExit(
                "ERROR: GROMACS Rg analysis requires --selection to pick the chain atoms."
            )
        chain_indices, chain_masses, selection_label = _select_rg_atoms_mda(
            args.topology,
            input_path,
            args.selection,
            args.n_chains,
        )
        frame_iter = _iter_mda_frames(
            args.topology,
            input_path,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
    else:
        raise SystemExit(f"ERROR: Unsupported trajectory format '{suffix}'.")

    print("=" * 65)
    print("  diffmd-analyze — Radius of Gyration Distribution")
    print("=" * 65)
    print(f"  Trajectory:   {input_path}")
    if args.topology:
        print(f"  Topology:     {args.topology}")
    print(f"  Selection:    {selection_label}")
    print(f"  Chains:       {args.n_chains}")
    print("-" * 65)

    time_list = []
    frame_index_list = []
    rg_frames = []

    for frame_idx, time_ns, pos, box in frame_iter:
        chains_pos = pos[chain_indices]
        chains_pos = _unwrap_chain_pbc(chains_pos, box)
        rg_vals = _frame_rg(chains_pos, chain_masses)
        time_list.append(time_ns)
        frame_index_list.append(frame_idx)
        rg_frames.append(rg_vals)

    if not rg_frames:
        raise SystemExit("ERROR: No valid frames found in trajectory.")

    time_arr = np.asarray(time_list, dtype=np.float64) * _TIME_FACTORS[args.time_unit]
    trajectory_frame_indices = np.asarray(frame_index_list, dtype=np.int64)
    rg_matrix = np.asarray(rg_frames, dtype=np.float64)
    rg_samples = rg_matrix.reshape(-1)
    frame_rg_mean = np.mean(rg_matrix, axis=1)
    sample_series = {
        f"rg_chain{chain_idx}_nm": rg_matrix[:, chain_idx]
        for chain_idx in range(rg_matrix.shape[1])
    }
    sample_series["rg_mean_nm"] = frame_rg_mean

    grid = _find_reference_grid(args.reference, dist_index)
    if grid is None:
        if args.dist_range:
            grid = np.linspace(args.dist_range[0], args.dist_range[1], args.nbins)
        else:
            grid = _default_rg_grid(rg_samples, args.nbins)
    else:
        grid = np.asarray(grid, dtype=np.float64)

    replay_pdf = _compute_kde(
        rg_samples,
        grid,
        bw_factor=args.bw,
        normalize=args.normalize,
    )

    series = {"replay_fixed": replay_pdf}
    if args.reference:
        for label, ref_path in args.reference:
            if label in series:
                raise SystemExit(f"ERROR: Duplicate series label '{label}'.")
            series[label] = _load_reference_pdf(ref_path, dist_index, expected_grid=grid)

    if args.legacy_adaptive:
        series["replay_legacy_adaptive"] = _legacy_adaptive_kde(
            rg_samples,
            grid,
            bw_factor=args.bw,
            normalize=args.normalize,
        )

    reference_label = None
    if args.reference:
        default_reference = args.reference[0][0]
        reference_label = args.compare_label or default_reference
        if reference_label not in series:
            raise SystemExit(
                f"ERROR: compare label '{reference_label}' does not match any series label."
            )

    ts_path = Path(f"{output_prefix}_rg_timeseries.dat")
    rg_xvg = Path(f"{output_prefix}_rg_dist.xvg")
    rg_npy = Path(f"{output_prefix}_rg_dist.npy")
    _save_rg_timeseries(ts_path, time_arr, rg_matrix, args.time_unit)
    _write_xvg(
        rg_xvg,
        grid,
        {"replay_fixed": replay_pdf},
        "Radius of gyration distribution",
        "Rg (nm)",
        "P(Rg)",
    )
    _write_distribution_npy(rg_npy, grid, {"replay_fixed": replay_pdf})

    print(
        f"  Processed {rg_matrix.shape[0]} frames and {rg_samples.size} chain samples "
        f"({time_arr[0]:.3f} – {time_arr[-1]:.3f} {args.time_unit})"
    )
    print(
        f"  Rg mean/std:   {np.mean(rg_samples):.6f} ± {np.std(rg_samples):.6f} nm"
    )
    print(f"  Time series    → {ts_path}")
    print(f"  Rg distribution→ {rg_xvg}")
    print(f"  Rg NPY         → {rg_npy}")

    sampled_manifest = None
    sample_points = None
    if args.sample_frames is not None:
        try:
            sample_label, sample_values = _resolve_sample_series(
                sample_series,
                args.sample_series,
                default_label="rg_mean_nm",
            )
            sampled_analysis_indices = _select_uniform_frame_indices(
                sample_values,
                args.sample_frames,
                observable_label=sample_label,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

        sample_output_dir = (
            Path(args.sample_output_dir)
            if args.sample_output_dir
            else Path(f"{output_prefix}_rg_samples")
        )
        sampled_manifest = _export_sampled_structures(
            input_path,
            sampled_analysis_indices,
            trajectory_frame_indices,
            time_arr,
            args.time_unit,
            sample_output_dir,
            sample_label,
            sample_series,
            source_kind="h5" if is_h5 else "mda",
            template_h5=args.sample_template_h5 if is_h5 else None,
            topology_path=args.topology if is_gromacs else None,
            extra_sample_fields=lambda analysis_idx: {
                "rg_mean_nm": float(frame_rg_mean[analysis_idx]),
                "rg_chain_nm": [
                    float(x) for x in np.asarray(rg_matrix[analysis_idx], dtype=np.float64)
                ],
            },
        )
        sample_points = ("replay_fixed", sample_values[sampled_analysis_indices])
        print(f"  Sample series  → {sample_label}")
        print(f"  Sampled starts → {sample_output_dir}")
        print(f"  Sample manifest→ {sample_output_dir / 'manifest.json'}")

    summary = {
        "trajectory": str(input_path),
        "topology": str(args.topology) if args.topology else None,
        "selection": selection_label,
        "n_chains": int(args.n_chains),
        "dist_index": int(dist_index),
        "width_ratio": float(args.bw),
        "bin_size": float(grid[1] - grid[0]),
        "absolute_bandwidth": float(args.bw * (grid[1] - grid[0])),
        "reference_label": reference_label,
        "rg_samples": {
            "count": int(rg_samples.size),
            "mean": float(np.mean(rg_samples)),
            "std": float(np.std(rg_samples)),
            "min": float(np.min(rg_samples)),
            "max": float(np.max(rg_samples)),
        },
        "frame_rg_mean": {
            "count": int(frame_rg_mean.size),
            "mean": float(np.mean(frame_rg_mean)),
            "std": float(np.std(frame_rg_mean)),
            "min": float(np.min(frame_rg_mean)),
            "max": float(np.max(frame_rg_mean)),
        },
        "series": {},
    }
    if sampled_manifest is not None:
        summary["sampled_frames"] = sampled_manifest

    for label, pdf in series.items():
        info = _pdf_summary(grid, pdf)
        if reference_label is not None:
            ref_pdf = series[reference_label]
            info["kl_to_reference"] = _kl_forward(pdf, ref_pdf)
            info["w1_to_reference"] = _wasserstein_1d(pdf, ref_pdf)
            info["rmse_to_reference"] = _rmse(pdf, ref_pdf)
        summary["series"][label] = info

    summary_path = (
        Path(f"{output_prefix}_rg_compare.json")
        if reference_label is not None
        else Path(f"{output_prefix}_rg_summary.json")
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Summary        → {summary_path}")

    plot_path = None
    if args.save_plot:
        plot_path = Path(args.save_plot)
    elif reference_label is not None or args.legacy_adaptive:
        plot_path = Path(f"{output_prefix}_rg_compare.png")

    if plot_path is not None or args.plot:
        _plot_distribution_comparison(
            grid,
            series,
            title=args.title or "Radius of gyration distribution comparison",
            xlabel="Radius of gyration (nm)",
            save_path=plot_path,
            show_plot=args.plot,
            sample_points=sample_points,
            style=args.plot_style,
        )

    print()
    print("Series summary:")
    for label, info in summary["series"].items():
        line = (
            f"  {label:>22} | mean={str(info['mean']):>8} std={str(info['std']):>8}"
        )
        if reference_label is not None:
            line += (
                f" KL={info['kl_to_reference']:.6f}"
                f" W1={info['w1_to_reference']:.6f}"
                f" RMSE={info['rmse_to_reference']:.6f}"
            )
        print(line)

    print()
    print("Done.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser / CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structural analysis tools for Diff-MD and GROMACS trajectories.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
              # Loss-compatible membrane density profile + APL from a Diff-MD H5
              diffmd-analyze --density-apl -f RUN_100ns.h5 \\
                  --com-type C1 --n-lipids 72 \\
                  --reference aa density_test.xvg -o dopc_density

              # Q4 for zinc with 4 cysteine-SG ligands (GROMACS)
              diffmd-analyze -q4 -f traj.xtc -s topol.pdb \\
                  --metal "name ZN" --ligands "(resname CYZ and name SG)"

              # Q4 by atom indices (works with both H5 and GROMACS)
              diffmd-analyze -q4 -f traj.h5 --site 0 1 2 3 4

              # Addison tau5 for a five-coordinate trigonal-bipyramidal site
              diffmd-analyze --q5 -f traj.h5 --site5 0 1 2 3 4 5

              # Rg PDF from a Diff-MD H5 trajectory, saved in loss-ready format
              diffmd-analyze --rg -f simulation.h5 --resname PCP --n-chains 1 \\
                  -o rg_analysis --nbins 120 --bw 3.0

              # Uniformly sample N start structures across the empirical Rg CDF
              diffmd-analyze --rg -f simulation.h5 --resname PCP --n-chains 1 \\
                  --sample-frames 8 -o rg_sampled --plot

              # Sample representative frames from a coordination-distance series
              diffmd-analyze -q4 -f traj_comp.xtc -s rubre_eq.pdb \\
                  --metal "name Co" --ligands "(resname CYZ and name SG)" \\
                  --sample-series site0_CYZ:SG --sample-frames 32 -o co_cys_sampled

              # Camera-ready comparison with direct Botticelli overrides
              diffmd-analyze --dist \\
                  --reference aa density_test.xvg \\
                  --series epoch138 DOPC_density_step138.npy \\
                  --compare-label aa --botticelli --hide-integral-panel \\
                  --save-plot dopc_density_camera_ready.png

            Notes
            -----
              - Coordinates are internally in nm (matching Diff-MD convention).
              - All KDE outputs use fixed-width Gaussian kernels with
                bandwidth = bw * bin_width, matching the current Diff-MD losses.
              - Tetrahedral outputs remain directly usable as reference inputs
                for coordination_tetrahedral_dist.
              - Q5 mode reports Addison tau5: (largest angle - second-largest
                angle) / 60, where ideal trigonal bipyramidal = 1 and ideal
                square pyramidal = 0.
              - Rg outputs are written as both XVG and NPY arrays with layout:
                    reference[0]    -> bin centres
                    reference[1:]   -> one PDF row per distribution
            """
        ),
    )

    parser.add_argument(
        "-q4",
        "--tetrahedral",
        action="store_true",
        help="Compute tetrahedral order parameter Q4 for coordination sites.",
    )
    parser.add_argument(
        "--q5",
        "--trigonal-bipyramidal",
        action="store_true",
        dest="trigonal_bipyramidal",
        help="Compute Addison tau5 for five-coordinate trigonal-bipyramidal sites.",
    )
    parser.add_argument(
        "--density-apl",
        "--density-and-apl",
        action="store_true",
        dest="density_and_apl",
        help="Compute loss-compatible membrane density profiles and area-per-lipid from a Diff-MD H5 trajectory.",
    )
    parser.add_argument(
        "--rg",
        "--radius-of-gyration",
        action="store_true",
        dest="radius_of_gyration",
        help="Compute radius-of-gyration distributions and optional comparisons.",
    )
    parser.add_argument(
        "--dist",
        "--distribution",
        action="store_true",
        dest="distribution",
        help="Compare saved .xvg/.npy distributions directly, without a trajectory.",
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Input trajectory: .h5 (Diff-MD) or .xtc/.trr/.gro (GROMACS).",
    )
    parser.add_argument(
        "-s",
        "--topology",
        help="Topology for GROMACS trajectories: .gro or .pdb (required for .xtc/.trr).",
    )

    parser.add_argument(
        "--site",
        action="append",
        nargs=5,
        type=int,
        metavar=("METAL", "LIG1", "LIG2", "LIG3", "LIG4"),
        help="Coordination site by 0-based atom indices: metal lig1 lig2 lig3 lig4. Repeatable.",
    )
    parser.add_argument(
        "--site5",
        action="append",
        nargs=6,
        type=int,
        metavar=("METAL", "LIG1", "LIG2", "LIG3", "LIG4", "LIG5"),
        help="Five-coordinate site by 0-based atom indices for --q5: metal lig1 lig2 lig3 lig4 lig5. Repeatable.",
    )
    parser.add_argument(
        "--metal",
        help="Metal atom selection string for MDAnalysis (e.g. 'name ZN').",
    )
    parser.add_argument(
        "--ligands",
        help="Ligand atom selection string for MDAnalysis for tetrahedral mode.",
    )

    parser.add_argument(
        "--resname",
        help="Residue name for H5 Rg analysis (used to build chain indices).",
    )
    parser.add_argument(
        "--com-type",
        help="Type label or integer type id used to center membrane density/APL analysis.",
    )
    parser.add_argument(
        "--selection",
        help="MDAnalysis atom selection for GROMACS Rg analysis.",
    )
    parser.add_argument(
        "--n-lipids",
        type=int,
        default=None,
        help="Total number of lipids (both leaflets) for density/APL analysis.",
    )
    parser.add_argument(
        "--apl-reference",
        type=float,
        default=None,
        help="Optional target area-per-lipid value in nm^2, reported in the summary JSON.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=1,
        help="Number of chains in the Rg selection (default: 1).",
    )
    parser.add_argument(
        "--reference",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Baseline distribution source; accepts .npy or .xvg. Repeatable.",
    )
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Additional saved distribution source for --dist mode; accepts .npy or .xvg.",
    )
    parser.add_argument(
        "--compare-label",
        help="Series label used as the reference baseline for KL/W1/RMSE (default: first --reference).",
    )
    parser.add_argument(
        "--dist-index",
        type=int,
        default=None,
        help="Distribution row index for multi-column inputs. Omit in --dist mode to compare all rows.",
    )
    parser.add_argument(
        "--legacy-adaptive",
        action="store_true",
        help="In Rg mode also evaluate the historical adaptive scipy gaussian_kde behavior.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help=(
            "Select N unique frames spread uniformly across the empirical CDF of a "
            "frame-level scalar series. In --rg the default series is rg_mean_nm; in "
            "--density-apl it is apl_nm2; in -q4/--q5 use --sample-series to choose a coordination or "
            "distance series. Diff-MD H5 inputs export restart-ready single-frame H5s, "
            "while GROMACS trajectories export single-frame .gro snapshots."
        ),
    )
    parser.add_argument(
        "--sample-series",
        help=(
            "Frame-level scalar series used by --sample-frames. Examples: rg_mean_nm, "
            "rg_chain0_nm, apl_nm2, q4_site0, q5_site0, tau5_site0, "
            "d_lig0_site0_nm, site0_CYZ:SG."
        ),
    )
    parser.add_argument(
        "--sample-output-dir",
        help=(
            "Directory for sampled structures and manifest.json."
        ),
    )
    parser.add_argument(
        "--sample-template-h5",
        help=(
            "Optional finish/input H5 used only when exporting sampled structures from "
            "a Diff-MD H5 trajectory that lacks flat restart metadata such as names, "
            "resnames, or molecules."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file prefix (default: input file stem).",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame index (default: 0).")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1).")

    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="Number of bins for KDE distributions (default: 100).",
    )
    parser.add_argument(
        "--bw",
        type=float,
        default=1.0,
        help="Fixed KDE width ratio: bandwidth = bw * bin_width (default: 1.0).",
    )
    parser.add_argument(
        "--dist-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Distribution range in nm for distance / Rg distributions (default: auto or reference grid).",
    )
    parser.add_argument(
        "--q-range",
        nargs=2,
        type=float,
        default=[0.0, 1.0],
        metavar=("MIN", "MAX"),
        help="Q4 range for tetrahedral distributions (default: 0.0 1.0).",
    )
    parser.add_argument(
        "--q5-range",
        nargs=2,
        type=float,
        default=[0.0, 1.0],
        metavar=("MIN", "MAX"),
        help="tau5 range for Q5/trigonal-bipyramidal distributions (default: 0.0 1.0).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize KDE curves so they integrate to 1.0.",
    )

    parser.add_argument(
        "-tu",
        "--time-unit",
        choices=["fs", "ps", "ns"],
        default="ns",
        dest="time_unit",
        help="Time unit for plots and time series output (default: ns).",
    )
    parser.add_argument(
        "--title",
        help="Optional title for comparison plots.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Suppress figure and panel titles in saved/displayed plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display diagnostic plots (requires matplotlib).",
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="Save diagnostic plots to file.",
    )
    parser.add_argument(
        "--botticelli",
        action="store_true",
        help=(
            "Enable paper-style plotting defaults with a curated palette, serif typography, "
            "and higher-resolution figure settings."
        ),
    )
    parser.add_argument(
        "--botticelli-config",
        metavar="PATH",
        help=(
            "Optional TOML or JSON file with plot customization for series labels, colors, "
            "linestyles, markers, figure size, axis limits, tick formatting, annotations, "
            "panel titles, legends, and layout."
        ),
    )
    parser.add_argument(
        "--figure-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Override figure size in inches for Botticelli/custom plots.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        help="Override output DPI for Botticelli/custom plots.",
    )
    parser.add_argument(
        "--legend-loc",
        help="Legend location override for Botticelli/custom plots.",
    )
    parser.add_argument(
        "--legend-label",
        action="append",
        metavar="LABEL=DISPLAY",
        help="Rename one plotted series in the legend. Repeatable.",
    )
    parser.add_argument(
        "--series-color",
        action="append",
        metavar="LABEL=COLOR",
        help="Set one series color. Repeatable.",
    )
    parser.add_argument(
        "--series-linestyle",
        action="append",
        metavar="LABEL=STYLE",
        help="Set one series linestyle. Repeatable.",
    )
    parser.add_argument(
        "--series-marker",
        action="append",
        metavar="LABEL=MARKER",
        help="Set one series marker symbol. Repeatable.",
    )
    parser.add_argument(
        "--series-linewidth",
        action="append",
        metavar="LABEL=VALUE",
        help="Set one series line width. Repeatable.",
    )
    parser.add_argument(
        "--series-alpha",
        action="append",
        metavar="LABEL=VALUE",
        help="Set one series alpha/opacity. Repeatable.",
    )
    parser.add_argument(
        "--hide-integral-panel",
        action="store_true",
        help="Hide the CDF/integral panel in comparison plots.",
    )
    parser.add_argument(
        "--font-family",
        help="Override font family for Botticelli/custom plots.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        help="Override base font size for Botticelli/custom plots.",
    )
    parser.add_argument(
        "--background-color",
        help="Override figure background color for Botticelli/custom plots.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.plot_style = _build_botticelli_style(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    selected_modes = (
        int(bool(args.tetrahedral))
        + int(bool(args.trigonal_bipyramidal))
        + int(bool(args.density_and_apl))
        + int(bool(args.radius_of_gyration))
        + int(bool(args.distribution))
    )
    if selected_modes != 1:
        parser.print_help()
        print(
            "\nERROR: Select exactly one analysis mode: use -q4 for tetrahedral "
            "analysis, --q5 for five-coordinate tau5 analysis, --density-apl "
            "for membrane density/APL analysis, --rg for radius-of-gyration "
            "analysis, or --dist for saved-distribution comparison."
        )
        return 1

    if args.tetrahedral:
        return _run_q4_analysis(args)
    if args.trigonal_bipyramidal:
        return _run_q5_analysis(args)
    if args.density_and_apl:
        return _run_density_apl_analysis(args)
    if args.distribution:
        return _run_distribution_analysis(args)
    return _run_rg_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())