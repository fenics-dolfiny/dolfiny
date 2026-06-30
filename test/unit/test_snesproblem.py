from typing import Any

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


def test_submesh(squaremesh_5) -> None:
    mesh = squaremesh_5
    tdim = mesh.topology.dim

    V = dolfinx.fem.functionspace(mesh, ("P", 1))

    boundary = dolfinx.mesh.locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.full(x[0].shape, True)
    )
    boundary_dof = dolfinx.fem.locate_dofs_topological(V, tdim - 1, boundary)
    bcs = [dolfinx.fem.dirichletbc(np.float64(0.0), boundary_dof, V)]

    def R(u: dolfinx.fem.Function, f: Any) -> ufl.classes.Expr:
        dx = ufl.dx(mesh)
        v = ufl.TestFunction(V)
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - f * v * dx  # type: ignore

    # ufl symbolic conditional
    u = dolfinx.fem.Function(V)
    x = ufl.SpatialCoordinate(mesh)
    eps = 1e-12
    f_symbolic = ufl.conditional(x[0] < 0.4 + eps, 1.0, 0)
    problem = dolfiny.snesproblem.SNESProblem([R(u, f_symbolic)], [u], bcs)
    (u,) = problem.solve()

    left = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] < 0.4 + eps)
    mesh_left, entity_maps, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, left)

    # submesh solve
    V_l = dolfinx.fem.functionspace(mesh_left, ("P", 1))
    f = dolfinx.fem.Function(V_l)
    f.interpolate(lambda x: np.full(x[0].shape, 1.0))

    u_sub = dolfinx.fem.Function(V)
    problem = dolfiny.snesproblem.SNESProblem(
        [R(u_sub, f)], [u_sub], bcs, entity_maps=[entity_maps]
    )
    (u_sub,) = problem.solve()

    assert np.allclose(u.x.array, u_sub.x.array)
