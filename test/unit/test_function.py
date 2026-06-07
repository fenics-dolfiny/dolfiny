from mpi4py import MPI

import dolfinx

import numpy as np
import numpy.typing as npt
import pytest

from dolfiny.function import evaluate, unroll_dofs


@pytest.mark.parametrize("shape", [1, 2, 3])
def test_unroll_dofs(shape: int) -> None:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n := 10, n)
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (shape,)))
    dofs = unroll_dofs(np.array([0, 1, 2], dtype=np.int32), V.dofmap.bs)

    assert np.allclose(dofs, np.arange(0, V.dofmap.bs * 3))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_evaluate(dtype: npt.DTypeLike) -> None:
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n := 10, n, n, dtype=dtype)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    f = dolfinx.fem.Function(V, dtype=dtype)

    def expr(x):
        return np.sin(x[0] * x[1] * x[2])

    f.interpolate(expr)

    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64)
    values = evaluate(f, x)

    assert np.allclose(values[:, 0], expr(x))
