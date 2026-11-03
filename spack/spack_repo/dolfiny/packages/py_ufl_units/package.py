from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class PyUflUnits(PythonPackage):
    """Physical units and dimensional analysis for UFL expressions and forms."""

    homepage = "https://github.com/fenics-dolfiny/ufl-units"
    git = "https://github.com/fenics-dolfiny/ufl-units.git"

    maintainers("anzil", "michalhabera", "schnellerhase")

    license("MIT")

    version("main", branch="main")

    depends_on("python@3.12:", type=("build", "run"))
    depends_on("py-setuptools@77:", type="build")

    depends_on("py-fenics-ufl", type=("build", "run"))
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-sympy", type=("build", "run"))
