import argparse
import logging
import logging.config
import os

from mpi4py import MPI

from dolfiny.utils import ANSI


class MPIRankFilter(logging.Filter):
    """Only allow log records through on MPI rank 0."""

    def filter(self, record):
        return MPI.COMM_WORLD.rank == 0


def _configure_dolfiny_logging():
    """Auto-configure logging for dolfiny, bypassing pytest and pre-configured loggers.

    This runs at import time. To suppress entirely, set the environment variable
    ``DOLFINY_LOG=0`` before importing dolfiny, or configure the root
    logger (any handler) before the import.

    Options (via CLI or environment variable):
        --log-level / DOLFINY_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        --log-prefix / DOLFINY_LOG_PREFIX=1: Enable timestamp and logger name prefix (default: off)
    """
    # Opt-out: respect explicit suppression via environment variable
    if os.environ.get("DOLFINY_LOG", "1") == "0":
        return

    # Parse logging arguments safely (CLI > env var > default)
    _valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-level", default=None, choices=_valid_levels)
    parser.add_argument("--log-prefix", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    # Resolve: CLI wins, then env var, then default
    env_level = os.environ.get("DOLFINY_LOG_LEVEL", "").upper()
    if env_level and env_level not in _valid_levels:
        raise ValueError(f"Invalid DOLFINY_LOG_LEVEL: {env_level}. Must be one of {_valid_levels}")
    log_level = args.log_level or env_level or "INFO"
    log_prefix = args.log_prefix or os.environ.get("DOLFINY_LOG_PREFIX", "0") == "1"

    if log_prefix:
        fmt = ANSI.bright_black + "%(asctime)s | %(name)s | " + ANSI.reset + "%(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(message)s"
        datefmt = None

    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "rank0": {
                "()": MPIRankFilter,
            }
        },
        "formatters": {
            "simple": {
                "format": fmt,
                "datefmt": datefmt,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                # no "stream" key — defaults to sys.stderr
                "formatter": "simple",
                "filters": ["rank0"],
            }
        },
        # Target ONLY the dolfiny logger
        "loggers": {
            "dolfiny": {
                "level": log_level,
                "handlers": ["console"],
            }
        },
    }

    logging.config.dictConfig(logger_config)


_configure_dolfiny_logging()
