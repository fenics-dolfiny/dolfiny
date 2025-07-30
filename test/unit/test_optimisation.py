from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
from numpy import typing as npt


def svanberg_cantilever_beam(tao: PETSc.TAO) -> tuple[npt.NDArray[np.float64], float]:  # type: ignore
    """
    Cantilever beam from ref. https://doi.org/10.1002/nme.1620240207.

    min  C₁ (x₁ + ... + x₅)
     x
    s.t. 0 < x
         0 ≤ C₂ - 61/x₁^3 - 37/x₂^3 - 19/x₃^3 - 7/x₄^3 - 1/x₅^3

    where
        C₁ = 0.0624
        C₂ = 1.0

    optimal point and objective value are
        x̄    = (6.016, 5.309, 4.494, 3.502, 2.153)
        f(x̄) = 1.340

    """
    C_1 = 0.0624
    C_2 = 1.0

    def objective(tao, x: PETSc.Vec) -> float:  # type: ignore
        return C_1 * x.sum()  # type: ignore

    def gradient(tao, x: PETSc.Vec, J: PETSc.Vec) -> None:  # type: ignore
        J.set(C_1)
        J.assemble()

    def constraint(tao, x: PETSc.Vec, c: PETSc.Vec) -> None:  # type: ignore
        c[0] = C_2
        c[0] -= 61.0 / x[0] ** 3
        c[0] -= 37.0 / x[1] ** 3
        c[0] -= 19.0 / x[2] ** 3
        c[0] -= 7.0 / x[3] ** 3
        c[0] -= 1.0 / x[4] ** 3
        c.assemble()

    def constraint_jacobian(tao, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:  # type: ignore
        J[0, 0] = 61.0 * 3 / x[0] ** 4
        J[0, 1] = 37.0 * 3 / x[1] ** 4
        J[0, 2] = 19.0 * 3 / x[2] ** 4
        J[0, 3] = 7.0 * 3 / x[3] ** 4
        J[0, 4] = 1.0 * 3 / x[4] ** 4
        J.assemble()

    x = PETSc.Vec().createSeq(5)  # type: ignore
    x.set(1.0)
    tao.setSolution(x)

    tao.setObjective(objective)
    tao.setGradient(gradient, x.copy())

    c = PETSc.Vec().createSeq(1)  # type: ignore
    tao.setInequalityConstraints(constraint, c)

    J_c = PETSc.Mat().createDense((1, 5))  # type: ignore
    J_c.assemble()
    tao.setJacobianInequality(constraint_jacobian, J_c)

    lb = x.copy()
    lb.set(1e-12)
    ub = x.copy()
    ub.set(PETSc.INFINITY)  # type: ignore
    tao.setVariableBounds(lb, ub)

    return np.array([6.016, 5.309, 4.494, 3.502, 2.153]), 1.340


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Sequential only.")
def test_svanberg_cantilever_beam():
    comm = MPI.COMM_SELF

    opts = PETSc.Options()
    opts["tao_type"] = "almm"

    tao = PETSc.TAO().create(comm=comm)
    tao.setFromOptions()

    x, f = svanberg_cantilever_beam(tao)
    tao.solve()

    assert tao.getConvergedReason() > 0
    assert np.allclose(tao.getSolution().getArray(), x, atol=1e-3)
    assert np.allclose(tao.getObjectiveValue(), f, atol=1e-3)
