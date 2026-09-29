"""C++ extensions for dolfiny."""

# Force MPI_Init
import mpi4py.MPI  # noqa


def get_include() -> str:
    """Directory holding the C++ headers installed alongside the _cpp extension."""
    import pathlib

    from dolfiny.cpp import _cpp

    return str(pathlib.Path(_cpp.__file__).parent)
