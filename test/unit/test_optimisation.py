import math

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
from numpy import typing as npt


def svanberg_cantilever_beam(tao: PETSc.TAO) -> tuple[npt.NDArray[np.float64], float, float]:  # type: ignore
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

    return np.array([6.016, 5.309, 4.494, 3.502, 2.153]), 1.340, 1e-3


def svanberg_two_bar_truss(tao: PETSc.TAO) -> tuple[npt.NDArray[np.float64], float, float]:  # type: ignore
    """
    2-bar truss from ref. https://doi.org/10.1002/nme.1620240207.

    min  C₁ x₁ √(1 + x₂^2)
     x
    s.t. 0.2 ≤ x₁ ≤ 4.0
         0.1 ≤ x₂ ≤ 1.6
         0 ≤ 1 - C₂ √(1 + x₂^2) (8/x₁ + 1/x₁x₂)
         0 ≤ 1 - C₂ √(1 + x₂^2) (8/x₁ - 1/x₁x₂)

    where
        C₁ = 1.0
        C₂ = 0.124

    optimal point and objective value are
        x̄    = (1.41, 0.38)
        f(x̄) = 1.51

    """
    C_1 = 1.0
    C_2 = 0.124

    def objective(tao: PETSc.TAO, x: PETSc.Vec) -> float:  # type: ignore
        return C_1 * x[0] * math.sqrt(1 + x[1] ** 2)  # type: ignore

    def gradient(tao, x: PETSc.Vec, J: PETSc.Vec) -> None:  # type: ignore
        J.zeroEntries()
        J[0] = C_1 * math.sqrt(1 + x[1] ** 2)
        J[1] = C_1 * x[0] * x[1] / math.sqrt(1 + x[1] ** 2)
        J.assemble()

    def constraint(tao: PETSc.TAO, x: PETSc.Vec, c: PETSc.Vec) -> None:  # type: ignore
        c[0] = 1 - C_2 * math.sqrt(1 + x[1] ** 2) * (8.0 / x[0] + 1.0 / (x[0] * x[1]))
        c[1] = 1 - C_2 * math.sqrt(1 + x[1] ** 2) * (8.0 / x[0] - 1.0 / (x[0] * x[1]))
        c.assemble()

    def constraint_jacobian(tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:  # type: ignore
        J[0, 0] = C_2 * (8 * x[1] + 1) * math.sqrt(x[1] ** 2 + 1) / (x[0] ** 2 * x[1])
        J[0, 1] = C_2 * (1 - 8 * x[1] ** 3) / (x[0] * x[1] ** 2 * math.sqrt(1 + x[1] ** 2))
        J[1, 0] = C_2 * (8 * x[1] - 1) * math.sqrt(x[1] ** 2 + 1) / (x[0] ** 2 * x[1])
        J[1, 1] = -C_2 * (8 * x[1] ** 3 + 1) / (x[0] * x[1] ** 2 * math.sqrt(1 + x[1] ** 2))
        J.assemble()

    x = PETSc.Vec().createSeq(2)  # type: ignore
    x[0] = 1.5
    x[1] = 0.5
    tao.setSolution(x)

    tao.setObjective(objective)
    tao.setGradient(gradient, x.copy())

    c = PETSc.Vec().createSeq(2)  # type: ignore
    tao.setInequalityConstraints(constraint, c)

    J_c = PETSc.Mat().createDense((2, 2))  # type: ignore
    tao.setJacobianInequality(constraint_jacobian, J_c)

    lb = x.copy()
    lb[0] = 0.2
    lb[1] = 0.1
    ub = x.copy()
    ub[0] = 4.0
    ub[1] = 1.6
    tao.setVariableBounds(lb, ub)

    return np.array([1.41, 0.38]), 1.51, 1e-2


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Sequential only.")
@pytest.mark.parametrize("problem", [svanberg_cantilever_beam, svanberg_two_bar_truss])
def test_almm(problem):
    opts = PETSc.Options()
    opts["tao_type"] = "almm"

    tao = PETSc.TAO().create()
    tao.setFromOptions()

    x, f, atol = problem(tao)
    tao.solve()

    assert tao.getConvergedReason() > 0
    assert np.allclose(tao.getSolution().getArray(), x, atol=atol)
    assert np.allclose(tao.getObjectiveValue(), f, atol=atol)
