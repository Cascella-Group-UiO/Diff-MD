import logging
import sys


def _get_rank_size():
    """Return (rank, size), falling back to (0, 1) when MPI is not initialized."""
    try:
        from mpi4py import MPI
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            return comm.Get_rank(), comm.Get_size()
    except Exception:
        pass
    return 0, 1


class MPIFilterRoot(logging.Filter):
    def filter(self, record):
        if record.funcName == "<module>":
            record.funcName = "main"
        rank, size = _get_rank_size()
        if rank == 0:
            record.rank = rank
            record.size = size
            return True
        else:
            return False


class MPIFilterAll(logging.Filter):
    def filter(self, record):
        if record.funcName == "<module>":
            record.funcName = "main"
        rank, size = _get_rank_size()
        record.rank = rank
        record.size = size
        return True


class Logger:
    level = None
    log_file = None
    format = " %(levelname)-8s [%(filename)s:%(lineno)d] <%(funcName)s> %(message)s"  # noqa: E501
    date_format = "%(asctime)s"
    formatter = logging.Formatter(fmt=date_format + format)
    rank0 = logging.getLogger("DiffMD.rank_0")
    all_ranks = logging.getLogger("DiffMD.all_ranks")

    @classmethod
    def setup(cls, default_level=logging.INFO, log_file=None, verbose=False):
        cls.level = default_level
        cls.log_file = log_file

        level = default_level

        # TODO: define custom log levels, because DEBUG seems weird to use
        if verbose:
            level = logging.DEBUG

        cls.rank0.setLevel(level)
        cls.all_ranks.setLevel(level)

        cls.rank0.propagate = False
        cls.all_ranks.propagate = False

        # Make setup idempotent: avoid duplicated output when setup is called
        # more than once in the same interpreter session.
        cls.rank0.handlers.clear()
        cls.all_ranks.handlers.clear()
        cls.rank0.filters.clear()
        cls.all_ranks.filters.clear()

        cls.rank0.addFilter(MPIFilterRoot())
        cls.all_ranks.addFilter(MPIFilterAll())

        if not log_file:
            return

        if log_file:
            cls.log_file_handler = logging.FileHandler(log_file)
            # cls.log_file_handler.setLevel(level)  # Should always log to file
            cls.log_file_handler.setFormatter(cls.formatter)
            cls.rank0.addHandler(cls.log_file_handler)
            cls.all_ranks.addHandler(cls.log_file_handler)

        cls.log_to_stdout = True
        cls.stdout_handler = logging.StreamHandler()
        cls.stdout_handler.setLevel(level)
        cls.stdout_handler.setStream(sys.stdout)
        cls.stdout_handler.setFormatter(cls.formatter)
        cls.rank0.addHandler(cls.stdout_handler)
        cls.all_ranks.addHandler(cls.stdout_handler)


def format_timedelta(timedelta):
    days = timedelta.days
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    microseconds = timedelta.microseconds
    ret_str = ""
    if days != 0:
        ret_str += f"{days} days "
    ret_str += f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"
    return ret_str
