import os
import random
import sys
from argparse import ArgumentParser

from .logger import Logger


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

    os.makedirs(args.destdir, exist_ok=True)

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
