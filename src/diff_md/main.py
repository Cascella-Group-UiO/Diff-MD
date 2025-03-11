import argparse
import sys

from mpi4py import MPI

from .configure_runtime import mdrun_runtime, optimize_runtime
from .mdrun import main as call_mdrun
from .optimize import main as call_optimize


class Parser:
    def __init__(self, comm):
        self.comm = comm
        parser = argparse.ArgumentParser(
            # description="Diff-MD",
            usage="""diff_md <command> [<args>]

The available commands are:
   mdrun      Run a MD simulation
   optimize   Run a differentiable MD simulation to optimize target parameters
""",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def mdrun(self):
        args = mdrun_runtime()
        call_mdrun(args)

    def optimize(self):
        args = optimize_runtime()
        call_optimize(args, self.comm)


def main():
    comm = MPI.COMM_WORLD
    Parser(comm)
