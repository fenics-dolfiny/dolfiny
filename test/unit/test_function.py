from mpi4py import MPI

import dolfinx

import numpy as np
import pytest

from dolfiny.function import unroll_dofs


@pytest.mark.parametrize("shape", [1, 2, 3])
def test_unroll_dofs(shape: int) -> None:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n := 10, n)
    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (shape,)))
    dofs = unroll_dofs(np.array([0, 1, 2], dtype=np.int32), V.dofmap.bs)

    assert np.allclose(dofs, np.arange(0, V.dofmap.bs * 3))
