from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class PyDolfiny(PythonPackage):
    """High-level convenience wrappers for DOLFINx."""

    homepage = "https://dolfiny.uni.lu/"
    url = "https://github.com/fenics-dolfiny/dolfiny/"
    git = "https://github.com/fenics-dolfiny/dolfiny.git"

    maintainers("anzil", "michalhabera", "schnellerhase")

    version("main", branch="main")

    # dependencies
    depends_on("cxx", type=("build", "run"))
    conflicts("%gcc@:12", msg="dolfiny requires GCC 13 or newer (C++23 <stdfloat>)")

    depends_on("py-scikit-build-core+pyproject@1:", when="@main", type="build")
    depends_on("py-nanobind@2:", when="@main:", type="build")
    depends_on("cmake@4:", type="build")

    depends_on("py-fenics-dolfinx+petsc4py+slepc4py@main", when="@main", type=("build", "link", "run"))

    depends_on("py-matplotlib", type="run")
    depends_on("py-pygraphviz", type="run")
    depends_on("py-scipy", type="run")
    depends_on("py-sympy", type="run")
    depends_on("py-pyvista@0.45:", type="run")
    depends_on("py-numba", type="run")
    depends_on("py-ufl-units@main", type="run")
    depends_on("py-gmsh", type="run")
    depends_on("py-ipython", type="run")
