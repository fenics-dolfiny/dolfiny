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


class ANSI(int):
    class Code(str):
        def __new__(cls, code: int):
            ESC = "\033"
            escape_sequence = f"{ESC}[{code}m"
            return super().__new__(cls, escape_sequence)

        def __init__(self, code: int):
            super().__init__()
            self._code = code

        @property
        def code(self) -> int:
            return self._code

    # text styles
    reset = Code(0)
    bold = Code(1)
    bold = Code(1)
    low_intensity = Code(2)
    italic = Code(3)
    underline = Code(4)
    blinking = Code(5)
    revers = Code(6)
    background = Code(7)
    invisible = Code(8)

    # colors
    black = Code(30)
    red = Code(31)
    green = Code(32)
    yellow = Code(33)
    blue = Code(34)
    magenta = Code(35)
    cyan = Code(36)
    white = Code(37)
    white = Code(37)

    # background
    bblack = Code(40)
    bred = Code(41)
    bgreen = Code(42)
    byellow = Code(43)
    bblue = Code(44)
    bmagenta = Code(45)
    bcyan = Code(46)
    bwhite = Code(47)

    # bright-colors
    bright_black = Code(90)
    bright_red = Code(91)
    bright_green = Code(92)
    bright_yellow = Code(93)
    bright_blue = Code(94)
    bright_magenta = Code(95)
    bright_cyan = Code(96)
    bright_white = Code(97)
    bright_white = Code(97)
