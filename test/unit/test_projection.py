from mpi4py import MPI

import basix
import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny

mesh3d = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 20, 20, 20)
mesh2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 80, 80)


def expr2d_scalar(x):
    return x[0] ** 3 * x[1] ** 2 + x[1] ** 3 * x[0] ** 2 + x[0] ** 2 * x[1] ** 2 + 1.0


def expr3d_scalar(x):
    return x[0] ** 3 * x[1] ** 2 + x[1] ** 3 * x[0] ** 2 + x[2] ** 2 * x[1] ** 2 + 1.0


@pytest.mark.parametrize("mesh, expr", [(mesh2d, expr2d_scalar), (mesh3d, expr3d_scalar)])
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 3),
        ("DP", 1),
        ("DP", 3),
    ],
)
def test_project_scalar(mesh, expr, family, order):
    e = basix.ufl.element(family, mesh.basix_cell(), order)
    V = dolfinx.fem.functionspace(mesh, e)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    u_expr = expr(x)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project(u_expr, up)

    # Provide reference
    ui = dolfinx.fem.Function(V)
    ex = dolfinx.fem.Expression(u_expr, V.element.interpolation_points)
    ui.interpolate(ex)

    # Compare
    assert np.allclose(ui.x.array, up.x.array, rtol=1.0e-2)


@pytest.mark.parametrize("mesh, expr", [(mesh2d, expr2d_scalar), (mesh3d, expr3d_scalar)])
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 2),
        ("DP", 1),
        ("DP", 2),
    ],
)
def test_project_vector(mesh, expr, family, order):
    e = basix.ufl.element(family, mesh.basix_cell(), order, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, e)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    u_expr = ufl.grad(expr(x)) + ufl.as_vector([1] * mesh.geometry.dim)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project(u_expr, up)

    # Provide reference
    ui = dolfinx.fem.Function(V)
    ex = dolfinx.fem.Expression(u_expr, V.element.interpolation_points)
    ui.interpolate(ex)

    # Compare
    assert np.allclose(ui.x.array, up.x.array, rtol=1.0e-2)


@pytest.mark.parametrize(
    "mesh, expr",
    [
        (mesh2d, expr2d_scalar),
        (mesh3d, expr3d_scalar),
    ],
)
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 2),
        ("RT", 2),
        ("RT", 3),
    ],
)
def test_project_codimension_vector_normal(mesh, expr, family, order):
    e = basix.ufl.element(family, mesh.basix_cell(), order, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, e)

    # Tag exterior facets
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mt_id = 42
    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, exterior_facets, mt_id)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    u_expr = ufl.grad(expr(x)) + ufl.as_vector([1] * mesh.geometry.dim)

    # normal
    projector = lambda u: ufl.dot(u, n)  # noqa: E731
    p_expression = projector(u_expr)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project_codimension(p_expression, up, projector, mt, mt_id)

    # Compute error on codimension
    norm_diff = ufl.sqrt(
        ufl.inner(projector(up) - projector(u_expr), projector(up) - projector(u_expr))
    )
    norm_expr = ufl.sqrt(ufl.inner(projector(u_expr), projector(u_expr)))
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    norm_error_relative = dolfiny.expression.assemble(norm_diff, ds)
    norm_error_relative /= dolfiny.expression.assemble(norm_expr, ds)

    # Compare
    assert norm_error_relative < 1.0e-2


@pytest.mark.parametrize(
    "mesh, expr",
    [
        (mesh2d, expr2d_scalar),
        (mesh3d, expr3d_scalar),
    ],
)
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 2),
        ("N1E", 2),
        ("N1E", 3),
        ("N2E", 2),
        ("N2E", 3),
    ],
)
def test_project_codimension_vector_tangential(mesh, expr, family, order):
    e = basix.ufl.element(family, mesh.basix_cell(), order, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, e)

    # Tag exterior facets
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mt_id = 42
    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, exterior_facets, mt_id)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    u_expr = ufl.grad(expr(x)) + ufl.as_vector([1] * mesh.geometry.dim)

    # tangential
    projector = lambda u: (ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)) * u  # noqa: E731
    p_expression = projector(u_expr)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project_codimension(p_expression, up, projector, mt, mt_id)

    # Compute error on codimension
    norm_diff = ufl.sqrt(
        ufl.inner(projector(up) - projector(u_expr), projector(up) - projector(u_expr))
    )
    norm_expr = ufl.sqrt(ufl.inner(projector(u_expr), projector(u_expr)))
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    norm_error_relative = dolfiny.expression.assemble(norm_diff, ds)
    norm_error_relative /= dolfiny.expression.assemble(norm_expr, ds)

    # Compare
    assert norm_error_relative < 1.0e-2


@pytest.mark.parametrize("mesh, expr", [(mesh2d, expr2d_scalar), (mesh3d, expr3d_scalar)])
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 2),
        ("HHJ", 1),
        ("HHJ", 2),
    ],
)
def test_project_codimension_matrix_normalnormal(mesh, expr, family, order):
    if family == "HHJ" and mesh.geometry.dim == 3:
        # pass, unless implemented
        return

    if family == "HHJ":
        e = basix.ufl.element(family, mesh.basix_cell(), order)
    else:
        e = basix.ufl.element(
            family,
            mesh.basix_cell(),
            order,
            shape=(mesh.geometry.dim, mesh.geometry.dim),
            symmetry=True,
        )
    V = dolfinx.fem.functionspace(mesh, e)

    # Tag exterior facets
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mt_id = 42
    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, exterior_facets, mt_id)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    u_expr = ufl.grad(ufl.grad(expr(x))) + ufl.Identity(mesh.geometry.dim)

    # normal-normal
    projector = lambda u: ufl.dot(ufl.dot(u, n), n)  # noqa: E731
    p_expression = projector(u_expr)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project_codimension(p_expression, up, projector, mt, mt_id)

    # Compute error on codimension
    norm_diff = ufl.sqrt(
        ufl.inner(projector(up) - projector(u_expr), projector(up) - projector(u_expr))
    )
    norm_expr = ufl.sqrt(ufl.inner(projector(u_expr), projector(u_expr)))
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    norm_error_relative = dolfiny.expression.assemble(norm_diff, ds)
    norm_error_relative /= dolfiny.expression.assemble(norm_expr, ds)

    # Compare
    assert norm_error_relative < 1.0e-2


@pytest.mark.parametrize("mesh, expr", [(mesh2d, expr2d_scalar), (mesh3d, expr3d_scalar)])
@pytest.mark.parametrize(
    "family, order",
    [
        ("P", 1),
        ("P", 2),
        ("Regge", 1),
        ("Regge", 2),
    ],
)
def test_project_codimension_matrix_normaltangential(mesh, expr, family, order):
    if family == "Regge":
        e = basix.ufl.element(family, mesh.basix_cell(), order)
    else:
        e = basix.ufl.element(
            family,
            mesh.basix_cell(),
            order,
            shape=(mesh.geometry.dim, mesh.geometry.dim),
            symmetry=True,
        )
    V = dolfinx.fem.functionspace(mesh, e)

    # Tag exterior facets
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mt_id = 42
    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, exterior_facets, mt_id)

    # Expression
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    u_expr = ufl.grad(ufl.grad(expr(x))) + ufl.Identity(mesh.geometry.dim)

    # normal-tangential
    projector = lambda u: (ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)) * ufl.dot(u, n)  # noqa: E731
    p_expression = projector(u_expr)

    # Project
    up = dolfinx.fem.Function(V)
    dolfiny.projection.project_codimension(p_expression, up, projector, mt, mt_id)

    # Compute error on codimension
    norm_diff = ufl.sqrt(
        ufl.inner(projector(up) - projector(u_expr), projector(up) - projector(u_expr))
    )
    norm_expr = ufl.sqrt(ufl.inner(projector(u_expr), projector(u_expr)))
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    norm_error_relative = dolfiny.expression.assemble(norm_diff, ds)
    norm_error_relative /= dolfiny.expression.assemble(norm_expr, ds)

    # Compare
    assert norm_error_relative < 1.0e-2
