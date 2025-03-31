from petsc4py import PETSc

import basix
import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny


def test_poisson(V1):
    W = V1

    u = dolfinx.fem.Function(W)
    f = 1.0

    F = 1/2 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx - ufl.inner(f, u) * ufl.dx

    opts = PETSc.Options("poisson")
    opts.setValue("tao_type", "nls")

    problem = dolfiny.snesblockproblem.TAOBlockProblem(F, [u], prefix="monolithic")
    (sol,) = problem.solve()
