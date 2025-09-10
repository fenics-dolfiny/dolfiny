from petsc4py import PETSc

import basix
import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny


def test_monolithic(V1, V2, squaremesh_5):
    mesh = squaremesh_5

    We = basix.ufl.mixed_element([V1.ufl_element(), V2.ufl_element()])
    W = dolfinx.fem.functionspace(mesh, We)

    u = dolfinx.fem.Function(W)
    u0, u1 = ufl.split(u)

    Phi = (ufl.sin(u0) - 0.5) ** 2 * ufl.dx(mesh) + (4.0 * u0 - u1) ** 2 * ufl.dx(mesh)

    v = ufl.TestFunction(W)
    F = ufl.derivative(Phi, u, v)

    opts = PETSc.Options("monolithic")

    opts.setValue("snes_type", "newtonls")
    opts.setValue("snes_linesearch_type", "basic")

    opts.setValue("snes_rtol", 1.0e-10)
    opts.setValue("snes_max_it", 20)

    opts.setValue("ksp_type", "preonly")
    opts.setValue("pc_type", "lu")
    opts.setValue("pc_factor_mat_solver_type", "mumps")

    problem = dolfiny.snesproblem.SNESProblem([F], [u], prefix="monolithic")
    (sol,) = problem.solve()

    u0, u1 = sol.split()
    u0 = u0.collapse()
    u1 = u1.collapse()

    assert np.isclose((u0.x.petsc_vec - np.arcsin(0.5)).norm(), 0.0)
    assert np.isclose((u1.x.petsc_vec - 4.0 * np.arcsin(0.5)).norm(), 0.0)


@pytest.mark.parametrize("nest", [True, False])
def test_block(V1, V2, squaremesh_5, nest):
    mesh = squaremesh_5

    u0 = dolfinx.fem.Function(V1, name="u0")
    u1 = dolfinx.fem.Function(V2, name="u1")

    v0 = ufl.TestFunction(V1)
    v1 = ufl.TestFunction(V2)

    Phi = (ufl.sin(u0) - 0.5) ** 2 * ufl.dx(mesh) + (4.0 * u0 - u1) ** 2 * ufl.dx(mesh)

    F0 = ufl.derivative(Phi, u0, v0)
    F1 = ufl.derivative(Phi, u1, v1)

    F = [F0, F1]
    u = [u0, u1]

    opts = PETSc.Options("block")

    opts.setValue("snes_type", "newtonls")
    opts.setValue("snes_rtol", 1.0e-08)
    opts.setValue("snes_max_it", 12)

    if nest:
        opts.setValue("ksp_type", "cg")
        opts.setValue("pc_type", "fieldsplit")
        opts.setValue("fieldsplit_pc_type", "lu")
        opts.setValue("ksp_rtol", 1.0e-08)
        opts.setValue("ksp_atol", 1.0e-12)
    else:
        opts.setValue("ksp_type", "preonly")
        opts.setValue("pc_type", "lu")
        opts.setValue("pc_factor_mat_solver_type", "mumps")

    problem = dolfiny.snesproblem.SNESProblem(F, u, nest=nest, prefix="block")
    sol = problem.solve()

    assert problem.snes.getConvergedReason() > 0
    assert np.isclose((sol[0].x.petsc_vec - np.arcsin(0.5)).norm(), 0.0)
    assert np.isclose((sol[1].x.petsc_vec - 4.0 * np.arcsin(0.5)).norm(), 0.0)
