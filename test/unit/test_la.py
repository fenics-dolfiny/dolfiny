from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
import scipy.sparse

import dolfiny
from dolfiny.la import negative_part, positive_part

skip_in_parallel = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial."
)


@skip_in_parallel
def test_scipy_to_petsc():
    A = scipy.sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    A_petsc = dolfiny.la.scipy_to_petsc(A)
    assert np.isclose(A_petsc.getValue(0, 1), 2.0)


@skip_in_parallel
def test_petsc_to_scipy():
    A = PETSc.Mat().createAIJ(size=(2, 2))
    A.setUp()
    A.setValuesCSR((0, 2, 4), (0, 1, 0, 1), (1.0, 2.0, 3.0, 4.0))
    A.assemble()

    A_scipy = dolfiny.la.petsc_to_scipy(A)
    assert np.isclose(A_scipy[0, 1], 2.0)


def test_positive_negative_dense():
    comm = MPI.COMM_WORLD
    A = PETSc.Mat().createDense(((2, PETSc.DECIDE), (2, PETSc.DECIDE)))
    offset = comm.rank * 2
    A[0 + offset, 0 + offset] = 1
    A[0 + offset, 1 + offset] = 2
    A[1 + offset, 0 + offset] = -3
    A[1 + offset, 1 + offset] = -4
    A.assemble()

    A_p = A.copy()
    positive_part(A_p)
    assert np.isclose(A_p[0 + offset, 0 + offset], 1)
    assert np.isclose(A_p[0 + offset, 1 + offset], 2)
    assert np.isclose(A_p[1 + offset, 0 + offset], 0)
    assert np.isclose(A_p[1 + offset, 1 + offset], 0)

    negative_part(A)
    assert np.isclose(A[0 + offset, 0 + offset], 0)
    assert np.isclose(A[0 + offset, 1 + offset], 0)
    assert np.isclose(A[1 + offset, 0 + offset], 3)
    assert np.isclose(A[1 + offset, 1 + offset], 4)


@skip_in_parallel
def test_positive_negative_seqaij():
    A = PETSc.Mat().createAIJ((2, 2), nnz=2)
    A[0, 0] = 1
    A[0, 1] = 2
    A[1, 0] = -3
    A[1, 1] = -4
    A.assemble()

    A_p = A.copy()
    positive_part(A_p)
    assert np.isclose(A_p[0, 0], 1)
    assert np.isclose(A_p[0, 1], 2)
    assert np.isclose(A_p[1, 0], 0)
    assert np.isclose(A_p[1, 1], 0)

    negative_part(A)
    assert np.isclose(A[0, 0], 0)
    assert np.isclose(A[0, 1], 0)
    assert np.isclose(A[1, 0], 3)
    assert np.isclose(A[1, 1], 4)
