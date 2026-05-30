import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import textwrap

import numpy as np


DEFAULT_COLUMNS = [
    "step",
    "time_fs",
    "temp_K",
    "E_total_kJmol",
    "E_potential_kJmol",
    "E_kin_kJmol",
    "E_LJ_kJmol",
    "E_elec_kJmol",
    "E_bond_kJmol",
    "E_angle_kJmol",
    "E_torsional_kJmol",
    "E_improper_torsional_kJmol",
]
PRESSURE_COLUMN = "P_bar"

PLOT_CHOICES = ("combined", "separate")
TIME_UNIT_TO_FS = {
    "fs": 1.0,
    "ps": 1_000.0,
    "ns": 1_000_000.0,
}


def _parse_header_columns(log_file: Path) -> list[str] | None:
    with log_file.open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            if "step" in stripped and "|" in stripped:
                return [part.strip().lstrip("# ") for part in stripped.split("|")]
    return None


def _resolve_columns(n_cols: int, header_columns: list[str] | None) -> list[str]:
    if header_columns and len(header_columns) == n_cols:
        return header_columns
    if n_cols == len(DEFAULT_COLUMNS):
        return DEFAULT_COLUMNS.copy()
    if n_cols == len(DEFAULT_COLUMNS) + 1:
        return DEFAULT_COLUMNS + [PRESSURE_COLUMN]
    raise ValueError(
        f"Unsupported number of columns in energy log: {n_cols}."
    )


def _load_energy_log(log_file: Path) -> tuple[np.ndarray, list[str]]:
    header_columns = _parse_header_columns(log_file)
    data = np.loadtxt(log_file, comments="#", ndmin=2)
    if data.size == 0:
        raise ValueError("The energy log is empty.")
    columns = _resolve_columns(data.shape[1], header_columns)
    return data, columns


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Reinstall gmx2diffmd or install matplotlib in the active environment."
        ) from exc
    return plt


def _parse_begin(begin: list[str] | None) -> float | None:
    if begin is None:
        return None
    value = float(begin[0])
    unit = begin[1].lower()
    scale = {"fs": 1.0, "ps": 1000.0, "ns": 1_000_000.0}.get(unit)
    if scale is None:
        raise ValueError(f"Unknown time unit '{unit}'. Use fs, ps, or ns.")
    return value * scale


def _normalize_select(select_args: list[str] | None) -> list[str] | None:
    if not select_args:
        return None
    selected = []
    for item in select_args:
        selected.extend(part.strip() for part in item.split(",") if part.strip())
    return selected or None


def _resolve_selected_names(
    selected: list[str] | None,
    analyzable_columns: list[str],
) -> list[str]:
    if selected is None:
        return analyzable_columns

    resolved = []
    invalid = []
    for item in selected:
        if item.isdecimal():
            selection_idx = int(item)
            if 1 <= selection_idx <= len(analyzable_columns):
                resolved.append(analyzable_columns[selection_idx - 1])
            else:
                invalid.append(item)
            continue
        if item in analyzable_columns:
            resolved.append(item)
        else:
            invalid.append(item)

    if invalid:
        raise ValueError(
            "Unknown column selection: "
            + ", ".join(invalid)
            + f". Use names or numbers 1-{len(analyzable_columns)} from --list-columns."
        )
    return resolved


def _filter_data(data: np.ndarray, columns: list[str], begin_fs: float | None) -> np.ndarray:
    if begin_fs is None:
        return data
    time_idx = columns.index("time_fs")
    filtered = data[data[:, time_idx] >= begin_fs]
    if len(filtered) == 0:
        raise ValueError("No data left after applying the requested start time.")
    return filtered


def _write_xvg(output_path: Path, time_data: np.ndarray, properties: dict[str, np.ndarray]) -> None:
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("# Created by diff-MD energy analyzer\n")
        outfile.write(f"@    title \"{output_path.stem}\"\n")
        outfile.write('@    xaxis  label "Time (fs)"\n')
        outfile.write('@    yaxis  label "Value"\n')
        outfile.write("@TYPE xy\n")
        for idx, prop_name in enumerate(properties):
            outfile.write(f'@ s{idx} legend "{prop_name}"\n')

        for frame_idx, time_fs in enumerate(time_data):
            values = [f"{time_fs:14.5f}"]
            values.extend(f"{properties[name][frame_idx]:14.5f}" for name in properties)
            outfile.write(" ".join(values) + "\n")


def _convert_time_axis(time_data_fs: np.ndarray, time_unit: str) -> tuple[np.ndarray, str]:
    factor = TIME_UNIT_TO_FS[time_unit]
    return time_data_fs / factor, time_unit


def _plot_properties(
    time_data_fs: np.ndarray,
    properties: dict[str, np.ndarray],
    time_unit: str,
    plot_mode: str,
    save_plot: str | None,
    show_plot: bool,
    title: str | None,
) -> None:
    plt = _load_matplotlib()
    time_axis, time_label_unit = _convert_time_axis(time_data_fs, time_unit)
    title = title or "Diff-MD Energy Analysis"

    if plot_mode == "separate":
        fig, axes = plt.subplots(len(properties), 1, figsize=(11, 3.5 * len(properties)), sharex=True)
        if len(properties) == 1:
            axes = [axes]
        for ax, (prop_name, values) in zip(axes, properties.items()):
            ax.plot(time_axis, values, linewidth=1.4, label=prop_name)
            ax.set_ylabel(prop_name)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
        axes[-1].set_xlabel(f"Time ({time_label_unit})")
        fig.suptitle(title)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        for prop_name, values in properties.items():
            ax.plot(time_axis, values, linewidth=1.4, label=prop_name)
        ax.set_xlabel(f"Time ({time_label_unit})")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()

    if save_plot:
        fig.savefig(save_plot, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_plot}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _simulation_format(columns: list[str]) -> str:
    return "NPT" if PRESSURE_COLUMN in columns else "NVT"


def _format_drift(values: np.ndarray, time_data_fs: np.ndarray, prop_name: str) -> str:
    if prop_name not in {"E_total_kJmol", "E_potential_kJmol"} or len(time_data_fs) <= 1:
        return "-"

    simulated_time = time_data_fs[-1] - time_data_fs[0]
    slope_fs, _ = np.polyfit(time_data_fs, values, 1)
    if simulated_time >= 1_000_000:
        drift = slope_fs * 1_000_000
        drift_unit = "ns"
    elif simulated_time >= 1_000:
        drift = slope_fs * 1_000
        drift_unit = "ps"
    else:
        drift = slope_fs
        drift_unit = "fs"
    return f"{drift:10.4f} kJ/mol/{drift_unit}"


def _print_summary(
    log_file: Path,
    columns: list[str],
    data: np.ndarray,
    properties: dict[str, np.ndarray],
    begin_fs: float | None,
) -> None:
    time_idx = columns.index("time_fs")
    time_data_fs = data[:, time_idx]
    start_time = time_data_fs[0]
    end_time = time_data_fs[-1]
    window_fs = end_time - start_time if len(time_data_fs) > 1 else 0.0

    print("=" * 108)
    print("Diff-MD Energy Analysis")
    print("=" * 108)
    print(f"File            : {log_file}")
    print(f"Detected format : {_simulation_format(columns)}")
    print(f"Frames used     : {len(data)}")
    print(f"Start time      : {start_time:.3f} fs")
    print(f"End time        : {end_time:.3f} fs")
    print(f"Window          : {window_fs:.3f} fs | {window_fs / 1_000.0:.6f} ps | {window_fs / 1_000_000.0:.9f} ns")
    if begin_fs is not None:
        print(f"Begin filter    : >= {begin_fs:.3f} fs")
    print(f"Selection       : {', '.join(properties)}")
    print("-" * 108)
    print(
        f"{'Observable':<28} {'Mean':>14} {'Std.Dev':>14} {'Min':>14} {'Max':>14} {'Drift':>20}"
    )
    print("-" * 108)
    for prop_name, values in properties.items():
        drift = _format_drift(values, time_data_fs, prop_name)
        print(
            f"{prop_name:<28} {np.mean(values):14.6f} {np.std(values):14.6f} {np.min(values):14.6f} {np.max(values):14.6f} {drift:>20}"
        )
    print("=" * 108)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
                description="Analyze Diff-MD energy.log files, including NPT runs with P_bar, and optionally plot/export the selected observables.",
                formatter_class=RawDescriptionHelpFormatter,
                epilog=textwrap.dedent(
                        """
                        Examples:
                            diffmd-energy -f energy.log
                            diffmd-energy -f energy.log -b 100 ps -s E_total_kJmol E_potential_kJmol
                            diffmd-energy -f energy.log -s 2 3
                            diffmd-energy -f energy.log -s P_bar,E_total_kJmol --plot --plot-mode separate
                            diffmd-energy -f energy.log --plot --save-plot energy.png -x energy.xvg
                            diffmd-energy -f energy.log --list-columns

                        Notes:
                            - Legacy NVT logs without P_bar and newer NPT logs with P_bar are both supported.
                            - Column selections accept names or 1-based numbers from --list-columns.
                            - Drift is reported for E_total_kJmol and E_potential_kJmol when at least two frames are available.
                            - Plotting requires matplotlib and can either be shown interactively or saved to file.
                        """
                ),
    )
    parser.add_argument("-f", "--file", required=True, help="Input energy.log file")
    parser.add_argument(
        "-b",
        "--begin",
        nargs=2,
        metavar=("VALUE", "UNIT"),
        help="Discard frames before this time, e.g. -b 100 ps",
    )
    parser.add_argument(
        "-s",
        "--select",
        nargs="+",
        help="Columns to summarize. Accepts names or 1-based --list-columns numbers separated by spaces or commas.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print the detected columns and exit.",
    )
    parser.add_argument(
        "-x",
        "--xvg",
        help="Optional XVG output path for the selected properties.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a matplotlib plot for the selected properties.",
    )
    parser.add_argument(
        "--save-plot",
        help="Save the plot to an image file. Can be used with or without --plot.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=PLOT_CHOICES,
        default="combined",
        help="Plot all properties on one axis or split them into separate subplots.",
    )
    parser.add_argument(
        "--plot-title",
        help="Custom title for the generated plot.",
    )
    parser.add_argument(
        "--time-unit",
        choices=tuple(TIME_UNIT_TO_FS),
        default="fs",
        help="Time unit used on the plot x-axis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_file = Path(args.file)
    data, columns = _load_energy_log(log_file)

    if args.list_columns:
        print("Detected analyzable columns:")
        for idx, column in enumerate(columns[2:], start=1):
            print(f"  {idx}: {column}")
        print("Metadata columns:")
        for column in columns[:2]:
            print(f"  - {column}")
        return 0

    begin_fs = _parse_begin(args.begin)
    data = _filter_data(data, columns, begin_fs)

    analyzable_columns = columns[2:]
    selected_names = _resolve_selected_names(
        _normalize_select(args.select), analyzable_columns
    )

    time_idx = columns.index("time_fs")
    time_data = data[:, time_idx]
    properties = {name: data[:, columns.index(name)] for name in selected_names}

    _print_summary(log_file, columns, data, properties, begin_fs)

    if args.xvg:
        output_path = Path(args.xvg)
        _write_xvg(output_path, time_data, properties)
        print(f"Saved XVG output to {output_path}")

    if args.plot or args.save_plot:
        _plot_properties(
            time_data,
            properties,
            time_unit=args.time_unit,
            plot_mode=args.plot_mode,
            save_plot=args.save_plot,
            show_plot=args.plot,
            title=args.plot_title,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())