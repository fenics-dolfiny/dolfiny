from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import pytest

from dolfiny.continuation import Crisfield
from dolfiny.expression import assemble
from dolfiny.snesproblem import SNESProblem


@pytest.mark.parametrize("lam0", [0, -0.01, 0.01])
@pytest.mark.parametrize("ds", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("psi", [0.5, 1.0, 2.0])
def test_crisfield_with_linear_scalar_problem(lam0, ds, psi):
    """F(u, λ) = u - λ on a single DOF function space."""

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, 1)
    V = dolfinx.fem.functionspace(mesh, ("DP", 0))
    u = dolfinx.fem.Function(V)
    lam = dolfinx.fem.Constant(mesh, 0.0)

    δu = ufl.TestFunction(V)

    f = δu * (u - lam) * ufl.dx

    problem = SNESProblem([f], [u])
    problem.verbose = dict(snes=False, ksp=False)

    cont = Crisfield(problem, lam)
    cont.initialise(ds=ds, λ=lam0, psi=psi)

    cont.solve_step(ds=ds, zero_x_predictor=True)

    # solution equals parameter (residual zero).
    res = assemble(u - lam, ufl.dx)
    assert np.isclose(res, 0.0, atol=1e-8)

    # arc-length constraint: |Δx|^2 + Δλ^2 * |dFdλ|^2 == ds^2
    dFdλ_inner = cont.inner(cont.dFdλ, cont.dFdλ)
    lhs = cont.inner(cont.Δx, cont.Δx) + cont.Δλ.value.item() ** 2 * cont.psi**2 * dFdλ_inner
    assert np.isclose(lhs, cont.ds**2, atol=1e-8)
