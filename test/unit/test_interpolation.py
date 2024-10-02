import time

from mpi4py import MPI

import basix
import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny

mesh3d = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
mesh2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1000, 1000)

He1 = basix.ufl.quadrature_element("tetrahedron", degree=1)
He4 = basix.ufl.quadrature_element("tetrahedron", degree=4)


@pytest.mark.parametrize(
    "element",
    [
        basix.ufl.element("P", "tetrahedron", 1, shape=(3, 3)),
        basix.ufl.element("DP", "tetrahedron", 0, shape=(3, 3)),
        basix.ufl.element("DP", "tetrahedron", 0, shape=(3, 3), symmetry=True),
        basix.ufl.element("DP", "tetrahedron", 1, shape=(3, 3), symmetry=True),
        basix.ufl.element("DP", "tetrahedron", 2, shape=(3, 3), symmetry=True),
        basix.ufl.blocked_element(He1, shape=(3, 3), symmetry=True),
        basix.ufl.blocked_element(He4, shape=(3, 3), symmetry=True),
        basix.ufl.element("Regge", "tetrahedron", 1),
    ],
)
def test_expr_matrix(element):
    V = dolfinx.fem.functionspace(mesh3d, element)

    x = ufl.SpatialCoordinate(V.mesh)
    u_expr = ufl.grad(ufl.grad(ufl.sinh(x[0]) + x[1] ** 3 + x[0] * x[2]))

    # Interpolate
    ui = dolfinx.fem.Function(V)
    dolfiny.interpolation.interpolate(u_expr, ui)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project(u_expr, up)

    # Compare
    assert np.isclose(dolfinx.la.norm(ui.x), dolfinx.la.norm(up.x), rtol=1.0e-4)


@pytest.mark.parametrize(
    "element",
    [
        basix.ufl.element("P", "tetrahedron", 1, shape=(3,)),
        basix.ufl.element("DP", "tetrahedron", 1, shape=(3,)),
        basix.ufl.blocked_element(He1, shape=(3,)),
        basix.ufl.blocked_element(He4, shape=(3,)),
        basix.ufl.element("N1E", "tetrahedron", 1),
        basix.ufl.element("N2E", "tetrahedron", 1),
    ],
)
def test_expr_vector(element):
    V = dolfinx.fem.functionspace(mesh3d, element)

    x = ufl.SpatialCoordinate(V.mesh)
    u_expr = ufl.grad(ufl.sinh(x[0]) + x[1] ** 3)

    # Interpolate
    ui = dolfinx.fem.Function(V)
    dolfiny.interpolation.interpolate(u_expr, ui)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project(u_expr, up)

    # Compare
    assert np.isclose(dolfinx.la.norm(ui.x), dolfinx.la.norm(up.x), rtol=1.0e-2)


@pytest.mark.parametrize(
    "element",
    [
        basix.ufl.element("P", "triangle", degree=1),
        basix.ufl.element("DP", "triangle", degree=1),
        basix.ufl.quadrature_element("triangle", degree=0),
        basix.ufl.quadrature_element("triangle", degree=1),
    ],
)
def test_linear_combination(element):
    V = dolfinx.fem.functionspace(mesh2d, element)

    u1 = dolfinx.fem.Function(V)
    u2 = dolfinx.fem.Function(V)
    u3 = dolfinx.fem.Function(V)

    f = dolfinx.fem.Function(V)
    g = dolfinx.fem.Function(V)

    print(f"\nndofs = {f.x.petsc_vec.getSize()}")

    with u1.x.petsc_vec.localForm() as local:
        local.set(1.0)
    with u2.x.petsc_vec.localForm() as local:
        local.set(2.0)
    with u3.x.petsc_vec.localForm() as local:
        local.set(3.0)

    c1 = dolfinx.fem.Constant(V.mesh, 3.14)
    c2 = dolfinx.fem.Constant(V.mesh, 0.1)

    expr = (c1 + c2) * (2.0 * (3.0 * u1 + u2) - u3 / 3.0 + 4 * (u1 + c2 * u3))

    t0 = time.time()
    dolfiny.interpolation.interpolate(expr, f)
    print(f"Linear combination {time.time() - t0:.4f}")

    t0 = time.time()
    dolfiny.interpolation.interpolate(
        2 + expr - 2, g
    )  # FIXME: provide no-op ufl.Expr of dolfinx.fem.Function
    print(f"Expression interp. {time.time() - t0:.4f}")

    assert np.allclose(f.x.array, g.x.array)


@pytest.mark.parametrize(
    "element0",
    [
        basix.ufl.element("P", "triangle", degree=1),
        basix.ufl.element("DP", "triangle", degree=1),
    ],
)
@pytest.mark.parametrize(
    "element1",
    [
        basix.ufl.element("P", "triangle", degree=3),
        basix.ufl.element("DP", "triangle", degree=3),
        basix.ufl.quadrature_element("triangle", degree=3),
    ],
)
def test_function_expression_scalar(element0, element1):
    V0 = dolfinx.fem.functionspace(mesh2d, element0)
    V1 = dolfinx.fem.functionspace(mesh2d, element1)

    u0 = dolfinx.fem.Function(V0)
    u0.interpolate(lambda x: np.cos(x[0]) + x[1] ** 2)

    u1 = dolfinx.fem.Function(V1)
    u2 = dolfinx.fem.Function(V1)

    print(f"\nndofs V0 = {u0.x.petsc_vec.getSize()}, ndofs V1 = {u1.x.petsc_vec.getSize()}")

    t0 = time.time()
    dolfiny.interpolation.interpolate(u0, u1)
    print(f"Interpolate dolfinx.fem.Function {time.time() - t0:.4f}")

    t0 = time.time()
    dolfiny.interpolation.interpolate(
        2 + u0 - 2, u2
    )  # FIXME: provide no-op ufl.Expr of dolfinx.fem.Function
    print(f"Interpolate ufl.Expr             {time.time() - t0:.4f}")

    assert np.allclose(u1.x.array, u2.x.array)


@pytest.mark.parametrize(
    "element0, element1",
    [
        (
            basix.ufl.blocked_element(He1, shape=(3,)),
            basix.ufl.element("DP", "tetrahedron", 0, shape=(3,)),
        )
    ],
)
def test_function_expression_blocked(element0, element1):
    V0 = dolfinx.fem.functionspace(mesh3d, element0)
    V1 = dolfinx.fem.functionspace(mesh3d, element1)

    u0 = dolfinx.fem.Function(V0)
    u0.interpolate(lambda x: np.array([np.cos(x[0]), x[1] ** 2, 3 * x[2]]))

    u1 = dolfinx.fem.Function(V1)
    u2 = dolfinx.fem.Function(V1)

    print(f"\nndofs V0 = {u0.x.petsc_vec.getSize()}, ndofs V1 = {u1.x.petsc_vec.getSize()}")

    t0 = time.time()
    dolfiny.interpolation.interpolate(u0, u1)
    print(f"Interpolate dolfinx.fem.Function {time.time() - t0:.4f}")

    t0 = time.time()
    dolfiny.interpolation.interpolate(
        2 * u0 / 2, u2
    )  # FIXME: provide no-op ufl.Expr of dolfinx.fem.Function
    print(f"Interpolate ufl.Expr             {time.time() - t0:.4f}")

    assert np.allclose(u1.x.array, u2.x.array)
