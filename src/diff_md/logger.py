import logging
import sys

from mpi4py import MPI


class MPIFilterRoot(logging.Filter):
    def filter(self, record):
        if record.funcName == "<module>":
            record.funcName = "main"
        if MPI.COMM_WORLD.Get_rank() == 0:
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return True
        else:
            return False


class MPIFilterAll(logging.Filter):
    def filter(self, record):
        if record.funcName == "<module>":
            record.funcName = "main"
        record.rank = MPI.COMM_WORLD.Get_rank()
        record.size = MPI.COMM_WORLD.Get_size()
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
