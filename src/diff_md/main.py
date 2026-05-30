import argparse
import os
import sys

# Work around hwloc OpenGL probe hanging on WSL2 / headless systems.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

# Prevent mpi4py from calling MPI_Init() on import.
# MPI will only be initialized explicitly in the `optimize` subcommand.
import mpi4py
mpi4py.rc.initialize = False


class Parser:
    def __init__(self):
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
        from .configure_runtime import mdrun_runtime
        from .mdrun import main as call_mdrun

        args = mdrun_runtime()
        call_mdrun(args)

    def optimize(self):
        from mpi4py import MPI

        MPI.Init_thread(MPI.THREAD_MULTIPLE)
        comm = MPI.COMM_WORLD

        # Pin each MPI rank to its own GPU before JAX is imported.
        # Multi-node safe: we need the *node-local* rank (0..GPUs-1),
        # not the global MPI rank which keeps growing across nodes.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            local_rank = _get_local_rank(comm)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

        # Disable XLA GPU autotuning and CUDA command buffers BEFORE JAX
        # is imported.  On TYKKY/Singularity containers:
        #  - autotune benchmark kernels crash silently, poison CUDA context
        #  - CUDA command buffers (CUDA Graphs) capture can crash during
        #    lax.scan body capture
        # Both cause CUDA_ERROR_LAUNCH_FAILED at the first BufferFromHostBuffer.
        _xf = os.environ.get("XLA_FLAGS", "")
        _extra = []
        if "--xla_gpu_autotune_level" not in _xf:
            _extra.append("--xla_gpu_autotune_level=0")
        if "--xla_gpu_enable_command_buffer" not in _xf:
            _extra.append("--xla_gpu_enable_command_buffer=")
        if _extra:
            os.environ["XLA_FLAGS"] = (_xf + " " + " ".join(_extra)).strip()

        from .configure_runtime import optimize_runtime
        from .optimize import main as call_optimize

        args = optimize_runtime()
        call_optimize(args, comm)


def _get_local_rank(comm):
    """Return the node-local MPI rank (0 .. n_local-1).

    Tries SLURM / OpenMPI / MPICH environment variables first (zero-cost),
    then falls back to the portable MPI-3 Split_type(COMM_TYPE_SHARED)
    which works on any MPI implementation and any number of nodes.
    """
    for var in ("SLURM_LOCALID",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "MPI_LOCALRANKID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)

    # Portable MPI-3 fallback: split communicator by shared memory domain
    # (= same physical node).  The resulting rank is guaranteed 0..n_local-1.
    node_comm = comm.Split_type(comm.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    node_comm.Free()
    return local_rank


def main():
    Parser()
