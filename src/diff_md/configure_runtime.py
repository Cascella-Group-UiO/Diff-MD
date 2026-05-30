import os
import random
import sys
import time
from argparse import ArgumentParser

from .logger import Logger


def _makedirs_mpi_safe(path: str, retries: int = 5, delay: float = 0.5) -> None:
    """Create *path* (and parents) tolerating MPI race conditions.

    On parallel file systems (Lustre, GPFS) several MPI ranks may call
    ``os.makedirs`` simultaneously.  Even with ``exist_ok=True`` this can
    raise ``FileExistsError`` because the metadata-cache lag makes
    ``os.path.isdir()`` return ``False`` even though the directory was
    just created by another rank.  We retry with a short sleep to let the
    filesystem metadata propagate.
    """
    for attempt in range(retries):
        try:
            os.makedirs(path, exist_ok=True)
            return
        except FileExistsError:
            if os.path.isdir(path):
                return
            if attempt < retries - 1:
                time.sleep(delay)
    # Final attempt — let it raise if it still fails.
    os.makedirs(path, exist_ok=True)


def get_arguments(ap, required):
    ap.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    ap.add_argument(
        "-d", "--destdir", default=".", help="Write output to specified directory"
    )
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Set output file name (default: 'sim')",
        default="sim",
    )
    ap.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Set the jax.random PRNG seed",
    )
    ap.add_argument(
        "--no-charges",
        action="store_true",
        help="Set charges to zero",
    )
    required.add_argument(
        "-c",
        "--config",
        dest="config",
        help="Input simulation parameters file (toml)",
        required=True,
    )
    required.add_argument(
        "-p",
        "--topol",
        dest="topol",
        help="Input topology file (toml)",
        required=True,
    )
    required.add_argument(
        "-f",
        "--file",
        dest="coord",
        help="Input coordinate file (h5)",
        required=True,
    )

    args = ap.parse_args(sys.argv[2:])
    args.logfile = f"{args.output}.log"
    args.prog = ap.prog

    _makedirs_mpi_safe(args.destdir)

    if args.seed is None:
        args.seed = random.randint(0, 100_000)

    # Setup logger
    Logger.setup(
        log_file=f"{args.destdir}/{args.logfile}",
        verbose=args.verbose,
    )
    return args


def mdrun_runtime():
    ap = ArgumentParser(prog="diff_md mdrun")

    ap.add_argument(
        "--disable-field",
        action="store_true",
        help="Disable field forces",
    )
    ap.add_argument(
        "--disable-bonds",
        action="store_true",
        help="Disable two-particle bond forces",
    )
    ap.add_argument(
        "--disable-angle-bonds",
        action="store_true",
        help="Disable three-particle angle bond forces",
    )
    ap.add_argument(
        "--disable-dihedrals",
        action="store_true",
        help="Disable four-particle dihedral forces",
    )
    ap.add_argument(
        "--disable-dipole",
        action="store_true",
        help="Disable BB dipole calculation",
    )
    ap.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision positions/velocities",
    )
    ap.add_argument(
        "--double-output",
        action="store_true",
        help="Use double precision in output h5md",
    )
    ap.add_argument(
        "--dump-per-particle",
        action="store_true",
        help="Log energy values per particle, not total",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Dump forces to h5md output",
    )
    ap.add_argument(
        "--velocity-output",
        action="store_true",
        help="Dump velocities to h5md output",
    )
    ap.add_argument(
        "-m",
        "--db",
        dest="database",
        help="Training model file (toml) from which to read the values for the chi interactions",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output trajectory (exact restart with step/time continuity)",
    )

    required = ap.add_argument_group("required arguments")
    args = get_arguments(ap, required)
    return args


def optimize_runtime():
    ap = ArgumentParser(prog="diff_md optimize")
    ap.add_argument(
        "--debug",
        help="Run the program in debug mode, performing a single test simulation",
        action="store_true",
    )
    ap.add_argument(
        "--restart",
        help="Restart training from a checkpoint state file in the given directory.",
    )
    ap.add_argument(
        "--restart-state",
        choices=("optimizer", "kinematic", "exact"),
        default="optimizer",
        help="Restart mode: 'optimizer' restores only epoch, parameters, and optimizer state; 'kinematic' also restores coordinates, velocities, box dimensions, and PRNG keys while recomputing thermodynamic observables; 'exact' additionally restores the carried equilibration state. No silent cross-mode fallback: if the checkpoint lacks the required state the run raises.",
    )
    ap.add_argument(
        "--double-precision",
        action="store_true",
        default=False,
        help="Enable JAX float64 (jax_enable_x64) for all MD and gradient computations. "
             "Can also be set via 'double_precision = true' in the [nn] section of the "
             "training TOML.",
    )

    required = ap.add_argument_group("required arguments")
    required.add_argument(
        "-m",
        "--model",
        help="Training setup and model (toml)",
    )

    args = get_arguments(ap, required)

    # Don't parse the database again in the optimize branch
    args.database = None
    return args
