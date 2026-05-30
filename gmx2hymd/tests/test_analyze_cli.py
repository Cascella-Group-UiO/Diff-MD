import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from gmx2HyMD import analyze_cli


def _gaussian_pdf(grid: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    pdf = np.exp(-0.5 * ((grid - mean) / sigma) ** 2)
    return pdf / np.trapezoid(pdf, grid)


def _write_test_rg_h5(path: Path) -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.9, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions, dtype=np.float32)
    box = np.repeat(np.eye(3, dtype=np.float32)[None, :, :] * 4.0, positions.shape[0], axis=0)

    with h5py.File(path, "w") as handle:
        handle.attrs["box"] = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
        handle.create_dataset("indices", data=np.arange(4, dtype=np.int32))
        handle.create_dataset("types", data=np.zeros(4, dtype=np.int32))
        handle.create_dataset("names", data=np.asarray([b"A", b"B", b"C", b"D"], dtype="S16"))
        handle.create_dataset("masses", data=np.ones(4, dtype=np.float32))
        handle.create_dataset("molecules", data=np.ones(4, dtype=np.int32))
        handle.create_dataset("resnames", data=np.asarray([b"PCP", b"PCP", b"PCP", b"PCP"], dtype="S10"))
        handle.create_dataset("charge", data=np.zeros(4, dtype=np.float32))

        particles = handle.create_group("particles").create_group("all")

        position_group = particles.create_group("position")
        position_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        position_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        position_group.create_dataset("value", data=positions, dtype=np.float32)

        velocity_group = particles.create_group("velocity")
        velocity_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        velocity_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        velocity_group.create_dataset("value", data=velocities, dtype=np.float32)

        box_group = particles.create_group("box").create_group("edges")
        box_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        box_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        box_group.create_dataset("value", data=box, dtype=np.float32)


def test_distribution_analysis_resamples_embedded_grids(tmp_path: Path) -> None:
    target_grid = np.linspace(0.5, 1.795, 260)
    target_pdf = _gaussian_pdf(target_grid, mean=0.99, sigma=0.04)
    target_path = tmp_path / "target_pdf.npy"
    np.save(target_path, np.vstack([target_grid, target_pdf]))

    replay_grid = np.linspace(0.800383, 1.159404, 260)
    replay_pdf = _gaussian_pdf(replay_grid, mean=1.03, sigma=0.02)
    replay_path = tmp_path / "rg_analysis_rg_dist.xvg"
    analyze_cli._write_xvg(
        replay_path,
        replay_grid,
        {"replay_fixed": replay_pdf},
        "Radius of gyration distribution",
        "Rg (nm)",
        "P(Rg)",
    )

    legacy_pdf = _gaussian_pdf(target_grid, mean=1.07, sigma=0.05)
    legacy_path = tmp_path / "step22.npy"
    np.save(legacy_path, legacy_pdf)

    args = Namespace(
        reference=[
            ("aa", str(target_path)),
            ("CG", str(replay_path)),
            ("step22", str(legacy_path)),
        ],
        series=None,
        output=str(tmp_path / "conf"),
        compare_label="aa",
        dist_index=None,
        dist_range=None,
        save_plot=None,
        plot=False,
        title=None,
    )

    result = analyze_cli._run_distribution_analysis(args)

    assert result == 0

    summary_path = tmp_path / "conf_distribution_compare.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["reference_label"] == "aa"
    assert summary["grid"]["count"] == 260
    assert summary["grid"]["min"] == float(target_grid[0])
    assert summary["grid"]["max"] == float(target_grid[-1])
    assert set(summary["series"]) == {"aa", "CG", "step22"}
    assert summary["series"]["CG"]["aggregate"]["mean_rmse_to_reference"] > 0.0


def test_rg_analysis_exports_uniformly_sampled_restart_h5s(tmp_path: Path) -> None:
    traj_path = tmp_path / "trajectory.h5"
    _write_test_rg_h5(traj_path)

    args = Namespace(
        file=str(traj_path),
        topology=None,
        resname="PCP",
        selection=None,
        n_chains=1,
        reference=None,
        compare_label=None,
        dist_index=None,
        output=str(tmp_path / "rg_analysis"),
        start=0,
        stop=None,
        stride=1,
        nbins=64,
        bw=1.0,
        dist_range=None,
        normalize=False,
        time_unit="ns",
        title=None,
        plot=False,
        save_plot=None,
        legacy_adaptive=False,
        sample_frames=3,
        sample_series=None,
        sample_output_dir=str(tmp_path / "rg_samples"),
        sample_template_h5=None,
    )

    result = analyze_cli._run_rg_analysis(args)

    assert result == 0

    summary = json.loads((tmp_path / "rg_analysis_rg_summary.json").read_text(encoding="utf-8"))
    sampled = summary["sampled_frames"]
    assert sampled["n_samples"] == 3
    assert [entry["frame_index"] for entry in sampled["samples"]] == [0, 2, 4]

    manifest_path = tmp_path / "rg_samples" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_mode"] == "uniform_empirical_cdf_frame_series"

    exported_h5 = tmp_path / "rg_samples" / "sample_001_frame_000002_input.h5"
    assert exported_h5.exists()
    with h5py.File(exported_h5, "r") as handle:
        assert handle["coordinates"].shape == (1, 4, 3)
        assert handle["velocities"].shape == (1, 4, 3)
        assert np.allclose(handle.attrs["box"], np.asarray([4.0, 4.0, 4.0], dtype=np.float32))
        assert handle["resnames"].shape == (4,)
        assert handle["particles/all/position/value"].shape == (1, 4, 3)


def _write_test_density_h5(path: Path) -> None:
    positions = np.asarray(
        [
            [[0.2, 0.3, 0.2], [1.2, 1.3, 3.8]],
            [[0.4, 0.5, 6.2], [2.4, 2.5, -0.3]],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions, dtype=np.float32)
    boxes = np.asarray(
        [
            np.diag([2.0, 2.0, 4.0]),
            np.diag([3.0, 3.0, 6.0]),
        ],
        dtype=np.float32,
    )

    with h5py.File(path, "w") as handle:
        handle.attrs["box"] = np.asarray([2.0, 2.0, 4.0], dtype=np.float32)
        handle.create_dataset("indices", data=np.arange(2, dtype=np.int32))
        handle.create_dataset("types", data=np.zeros(2, dtype=np.int32))
        handle.create_dataset("names", data=np.asarray([b"C1", b"C1"], dtype="S16"))
        handle.create_dataset("masses", data=np.ones(2, dtype=np.float32))
        handle.create_dataset("molecules", data=np.ones(2, dtype=np.int32))
        handle.create_dataset("resnames", data=np.asarray([b"DOPC", b"DOPC"], dtype="S10"))
        handle.create_dataset("charge", data=np.zeros(2, dtype=np.float32))

        particles = handle.create_group("particles").create_group("all")

        position_group = particles.create_group("position")
        position_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        position_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        position_group.create_dataset("value", data=positions, dtype=np.float32)

        velocity_group = particles.create_group("velocity")
        velocity_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        velocity_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        velocity_group.create_dataset("value", data=velocities, dtype=np.float32)

        box_group = particles.create_group("box").create_group("edges")
        box_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        box_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        box_group.create_dataset("value", data=boxes, dtype=np.float32)


def test_density_apl_uses_full_selected_box_range_and_samples_frames(tmp_path: Path) -> None:
    traj_path = tmp_path / "density.h5"
    _write_test_density_h5(traj_path)

    args = Namespace(
        file=str(traj_path),
        output=str(tmp_path / "density_analysis"),
        reference=None,
        compare_label=None,
        dist_index=None,
        start=0,
        stop=None,
        stride=1,
        nbins=20,
        bw=1.0,
        dist_range=None,
        normalize=False,
        time_unit="ns",
        title=None,
        plot=False,
        save_plot=None,
        plot_style=None,
        n_lipids=2,
        com_type="C1",
        apl_reference=None,
        sample_frames=2,
        sample_series=None,
        sample_output_dir=str(tmp_path / "density_samples"),
        sample_template_h5=None,
    )

    result = analyze_cli._run_density_apl_analysis(args)

    assert result == 0

    summary = json.loads((tmp_path / "density_analysis_density_apl_summary.json").read_text(encoding="utf-8"))
    assert summary["grid"]["count"] == 20
    assert summary["grid"]["min"] == -2.85
    assert summary["grid"]["max"] == 2.85
    assert summary["apl"]["mean"] == 6.5

    manifest = json.loads((tmp_path / "density_samples" / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["frame_index"] for entry in manifest["samples"]] == [0, 1]


def test_coordination_geometry_reference_values() -> None:
    center = np.zeros(3, dtype=np.float64)
    tetra_neighbors = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert analyze_cli.tetrahedral_q(center, tetra_neighbors) == pytest.approx(1.0)

    tbp_neighbors = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert analyze_cli.trigonal_bipyramidal_tau5(center, tbp_neighbors) == pytest.approx(1.0)

    square_pyramid_neighbors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert analyze_cli.trigonal_bipyramidal_tau5(center, square_pyramid_neighbors) == pytest.approx(0.0)

    box = np.asarray([4.0, 4.0, 4.0], dtype=np.float64)
    wrapped_center = np.asarray([3.8, 2.0, 2.0], dtype=np.float64)
    wrapped_neighbors = np.mod(wrapped_center + tbp_neighbors, box)
    assert analyze_cli.trigonal_bipyramidal_tau5(wrapped_center, wrapped_neighbors, box) == pytest.approx(1.0)


def _write_test_q5_h5(path: Path) -> None:
    center = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    tbp_neighbors = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    positions = []
    for scale in (1.0, 1.05, 0.95):
        frame = np.vstack([center, center + scale * tbp_neighbors]).astype(np.float32)
        positions.append(frame)
    positions = np.asarray(positions, dtype=np.float32)
    boxes = np.repeat(np.eye(3, dtype=np.float32)[None, :, :] * 6.0, positions.shape[0], axis=0)

    with h5py.File(path, "w") as handle:
        particles = handle.create_group("particles").create_group("all")
        position_group = particles.create_group("position")
        position_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        position_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        position_group.create_dataset("value", data=positions, dtype=np.float32)
        box_group = particles.create_group("box").create_group("edges")
        box_group.create_dataset("step", data=np.arange(positions.shape[0], dtype=np.int32))
        box_group.create_dataset("time", data=np.arange(positions.shape[0], dtype=np.float32) * 1000.0)
        box_group.create_dataset("value", data=boxes, dtype=np.float32)


def test_q5_analysis_writes_tau5_outputs(tmp_path: Path) -> None:
    traj_path = tmp_path / "q5.h5"
    _write_test_q5_h5(traj_path)

    args = Namespace(
        file=str(traj_path),
        topology=None,
        site5=[(0, 1, 2, 3, 4, 5)],
        metal=None,
        ligands=None,
        output=str(tmp_path / "q5_analysis"),
        start=0,
        stop=None,
        stride=1,
        nbins=32,
        bw=1.0,
        dist_range=None,
        q5_range=[0.0, 1.0],
        normalize=False,
        time_unit="ns",
        plot=False,
        save_plot=None,
        plot_style=None,
        sample_frames=None,
        sample_series=None,
        sample_output_dir=None,
        sample_template_h5=None,
    )

    result = analyze_cli._run_q5_analysis(args)
    assert result == 0

    timeseries = np.loadtxt(tmp_path / "q5_analysis_q5_timeseries.dat")
    assert timeseries.shape == (3, 7)
    assert np.allclose(timeseries[:, 1], 1.0, atol=1e-6)
    q5_dist = np.load(tmp_path / "q5_analysis_q5_dist.npy")
    coord_dist = np.load(tmp_path / "q5_analysis_coord_dist_q5.npy")
    assert q5_dist.shape == (2, 32)
    assert coord_dist.shape == (6, 32)
    assert (tmp_path / "q5_analysis_coord_dist_per_ligand_q5.xvg").exists()
    assert (tmp_path / "q5_analysis_mean_distances_q5.dat").exists()