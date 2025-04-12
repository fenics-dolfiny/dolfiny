from mpi4py import MPI


def pprint(str="", end="", flush=True, comm=MPI.COMM_WORLD):
    """Parallel print for MPI processes. Only rank==0 prints and flushes.

    Parameters
    ----------
    str: optional
        String to be printed
    end: optional
        Line ending
    flush: optional
        Flag for flushing output to stdout
    comm: optional
        MPI communicator

    """

    if comm.rank == 0:
        print(str, end, flush=flush)


def attributes_to_dict(c, invert=False):
    """Generate dictionary of class attributes.

    Parameters
    ----------
    s: Class
    invert: optional
        Invert key-value pair in dictionary

    """
    d = {}

    for k, v in vars(c).items():
        if not callable(v) and not k.startswith("__"):
            if invert:
                d[v] = k
            else:
                d[k] = v

    return d


def prefixify(n: int, prefixes=[" ", "k", "m", "b"]) -> str:
    """Convert given integer number to [sign] + 3 digits + (metric) prefix.

    Parameters
    ----------
    n: integer
    prefixes: optional
        List of (metric) prefix characters

    """
    # https://stackoverflow.com/a/74099536
    i = int(0.30102999566398114 * (int(n).bit_length() - 1)) + 1
    e = i - (10**i > n)
    e //= 3
    return f"{n // 10 ** (3 * e):>3d}{prefixes[e]}"
