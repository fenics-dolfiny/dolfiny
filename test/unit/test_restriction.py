import os

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny

skip_in_parallel = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial."
)


def test_coupled_poisson():
    # dS integrals in parallel require shared_facet ghost mode
    if MPI.COMM_WORLD.size == 1:
        ghost_mode = dolfinx.mesh.GhostMode.none
    else:
        ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16, ghost_mode=ghost_mode)

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    left_half = dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim, lambda x: np.less_equal(x[0], 0.5)
    )
    right_half = dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim, lambda x: np.greater_equal(x[0], 0.5)
    )

    left_values = np.full(left_half.shape, 1, dtype=np.intc)
    right_values = np.full(right_half.shape, 2, dtype=np.intc)

    indices = np.hstack((left_half, right_half))
    values = np.hstack((left_values, right_values))
    indices, pos = np.unique(indices, return_index=True)
    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, indices, values[pos])

    interface_facets = dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.5)
    )
    indices = np.unique(interface_facets)
    mt_interface = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, indices, 1)

    U0 = dolfinx.fem.functionspace(mesh, ("P", 1))
    U1 = dolfinx.fem.functionspace(mesh, ("P", 2))
    L = dolfinx.fem.functionspace(mesh, ("P", 1))

    v0 = ufl.TestFunction(U0)
    v1 = ufl.TestFunction(U1)
    m = ufl.TestFunction(L)

    u0_bc = dolfinx.fem.Function(U0)
    u1_bc = dolfinx.fem.Function(U1)

    dx = ufl.Measure("dx", subdomain_data=mt, domain=mesh)
    dS = ufl.Measure("dS", subdomain_data=mt_interface, domain=mesh)

    w0 = dolfinx.fem.Function(U0, name="w0")
    w1 = dolfinx.fem.Function(U1, name="w1")
    lam = dolfinx.fem.Function(L, name="l")

    b0 = v0 * dx(1)
    b1 = v1 * dx(2)
    b2 = dolfinx.fem.Function(L)("+") * m("+") * dS(1)

    F0 = ufl.inner(ufl.grad(w0), ufl.grad(v0)) * dx(1) + lam("-") * v0("-") * dS(1) - b0
    F1 = ufl.inner(ufl.grad(w1), ufl.grad(v1)) * dx(2) - lam("+") * v1("+") * dS(1) - b1
    F2 = w0("-") * m("-") * dS(1) - w1("+") * m("+") * dS(1) - b2

    bcdofsU0 = dolfinx.fem.locate_dofs_geometrical(U0, lambda x: np.isclose(x[0], 0.0))
    bcdofsU1 = dolfinx.fem.locate_dofs_geometrical(U1, lambda x: np.isclose(x[0], 1.0))
    bcs = [dolfinx.fem.dirichletbc(u0_bc, bcdofsU0), dolfinx.fem.dirichletbc(u1_bc, bcdofsU1)]

    rdofsU0 = dolfinx.fem.locate_dofs_topological(U0, mesh.topology.dim, left_half, remote=False)
    rdofsU1 = dolfinx.fem.locate_dofs_topological(U1, mesh.topology.dim, right_half, remote=False)
    rdofsL = dolfinx.fem.locate_dofs_topological(
        L, mesh.topology.dim - 1, interface_facets, remote=False
    )

    r = dolfiny.restriction.Restriction([U0, U1, L], [rdofsU0, rdofsU1, rdofsL])

    opts = PETSc.Options("poisson")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 5
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts["mat_mumps_icntl_24"] = 1

    problem = dolfiny.snesproblem.SNESProblem(
        [F0, F1, F2], [w0, w1, lam], bcs=bcs, restriction=r, prefix="poisson"
    )
    s0, s1, _s2 = problem.solve()

    assert problem.snes.getConvergedReason() > 0
    assert problem.snes.getIterationNumber() == 1

    # Evaluate the solution -0.5*x*(x-1) at x=0.5
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    p = np.array([0.5, 0.5, 0.0], dtype=np.float64)

    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, p)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, p).array

    if len(cells) > 0:
        value_s0 = s0.eval(p, cells[0])
        value_s1 = s1.eval(p, cells[0])

        assert np.isclose(value_s0[0], 0.125, rtol=1.0e-4)
        assert np.isclose(value_s1[0], 0.125, rtol=1.0e-4)


def test_sloped_stokes():
    path = os.path.dirname(os.path.realpath(__file__))

    # Read mesh, subdomains and boundaries
    with dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, os.path.join(path, "data", "sloped_triangle_mesh.xdmf"), "r"
    ) as infile:
        mesh = infile.read_mesh(name="Grid")
        mesh.topology.create_connectivity(1, mesh.topology.dim)

    with dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, os.path.join(path, "data", "sloped_line_mvc.xdmf"), "r"
    ) as infile:
        boundaries = infile.read_meshtags(mesh, name="Grid")

    V = dolfinx.fem.functionspace(mesh, ("P", 2, (mesh.geometry.dim,)))
    P = dolfinx.fem.functionspace(mesh, ("P", 1))
    L = dolfinx.fem.functionspace(mesh, ("P", 1))

    u = dolfinx.fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    p = dolfinx.fem.Function(P, name="p")
    q = ufl.TestFunction(P)

    lam = dolfinx.fem.Function(L, name="l")
    m = ufl.TestFunction(L)

    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", subdomain_data=boundaries, domain=mesh)

    F0 = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh)
        - p * ufl.div(v) * ufl.dx(mesh)
        + lam * ufl.inner(v, n) * ds(1)
    )
    F1 = ufl.div(u) * q * ufl.dx(mesh)
    F2 = ufl.inner(u, n) * m * ds(1)

    u_bc = dolfinx.fem.Function(V)
    bc_facets = np.where(boundaries.values == 4)[0]
    bcdofsV = dolfinx.fem.locate_dofs_topological(V, 1, boundaries.indices[bc_facets])

    u_bc_top = dolfinx.fem.Function(V)
    bctop_facets = np.where(boundaries.values == 3)[0]
    bcdofstopV = dolfinx.fem.locate_dofs_topological(V, 1, boundaries.indices[bctop_facets])

    def utop(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 1.0
        values[1] = 0.0
        return values

    u_bc_top.interpolate(utop)

    # Find common dofs at the top corners
    intersect = np.intersect1d(bcdofsV, bcdofstopV)

    # Remove for consistency
    bcdofstopV = np.setdiff1d(bcdofstopV, intersect)

    bcs = [dolfinx.fem.dirichletbc(u_bc, bcdofsV)]
    bcs.append(dolfinx.fem.dirichletbc(u_bc_top, bcdofstopV))

    r_facets = np.where(boundaries.values == 1)[0]
    rdofsL = dolfinx.fem.locate_dofs_topological(L, 1, boundaries.indices[r_facets])

    Vsize = V.dofmap.index_map_bs * (V.dofmap.index_map.size_local)
    Psize = P.dofmap.index_map_bs * (P.dofmap.index_map.size_local)

    rdofsV = np.arange(Vsize, dtype=np.int32)
    rdofsP = np.arange(Psize, dtype=np.int32)

    r = dolfiny.restriction.Restriction([V, P, L], [rdofsV, rdofsP, rdofsL])

    opts = PETSc.Options("stokes")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 10
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts["mat_mumps_icntl_24"] = 1

    problem = dolfiny.snesproblem.SNESProblem(
        [F0, F1, F2], [u, p, lam], bcs=bcs, restriction=r, prefix="stokes"
    )
    problem.solve()

    assert problem.snes.getConvergedReason() > 0
    assert problem.snes.getIterationNumber() == 1


def test_pipes_stokes():
    import gmsh

    gmsh.initialize(interruptible=False)
    gmsh.model.add("test_pipes")
    geo = gmsh.model.geo

    p0 = geo.addPoint(0.0, 0.0, 0.0)
    p1 = geo.addPoint(1.0, 0.0, 0.0)
    p2 = geo.addPoint(0.0, -1.0, 0.0)
    p3 = geo.addPoint(1.0, -1.0, 0.0)
    p4 = geo.addPoint(1.0, -2.0, 0.0)
    p5 = geo.addPoint(2.0, -2.0, 0.0)
    p6 = geo.addPoint(2.0, -1.0, 0.0)

    l0 = geo.addLine(p0, p2)
    l1 = geo.addCircleArc(p2, p3, p4)
    l2 = geo.addLine(p4, p5)
    l3 = geo.addLine(p5, p6)
    l4 = geo.addLine(p6, p3)
    l5 = geo.addLine(p3, p1)
    l6 = geo.addLine(p1, p0)

    cl = geo.addCurveLoop([l0, l1, l2, l3, l4, l5, l6])
    s0 = geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Bottom
    bottom_tag = gmsh.model.addPhysicalGroup(1, [l0, l1, l2])
    gmsh.model.setPhysicalName(1, bottom_tag, "bottom")
    # Up
    up_tag = gmsh.model.addPhysicalGroup(1, [l4, l5])
    gmsh.model.setPhysicalName(1, up_tag, "up")
    # Inflow
    inflow_tag = gmsh.model.addPhysicalGroup(1, [l6])
    gmsh.model.setPhysicalName(1, inflow_tag, "inflow")

    # Surface (if not added as PG, only 1D entities are created)
    gmsh.model.addPhysicalGroup(2, [s0])

    geo.synchronize()
    gmsh.model.mesh.generate()

    gmsh.model.mesh.refine()
    gmsh.model.mesh.refine()

    gmsh.model.mesh.generate()

    mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
    mesh = mesh_data.mesh

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as out:
        out.write_mesh(mesh)

    V = dolfinx.fem.functionspace(mesh, ("P", 2, (mesh.geometry.dim,)))
    P = dolfinx.fem.functionspace(mesh, ("P", 1))
    L = dolfinx.fem.functionspace(mesh, ("P", 1))

    u = dolfinx.fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    p = dolfinx.fem.Function(P, name="p")
    q = ufl.TestFunction(P)

    lam = dolfinx.fem.Function(L, name="l")
    m = ufl.TestFunction(L)

    ds = ufl.Measure("ds", subdomain_data=mesh_data.facet_tags, domain=mesh)
    n = ufl.FacetNormal(mesh)

    F0 = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh)
        - p * ufl.div(v) * ufl.dx(mesh)
        + lam * ufl.inner(v, n) * ds(mesh_data.physical_groups["bottom"].tag)
    )
    F1 = ufl.div(u) * q * ufl.dx(mesh) + dolfinx.fem.Constant(mesh, 0.0) * p * q * ufl.dx(mesh)
    F2 = ufl.inner(u, n) * m * ds(mesh_data.physical_groups["bottom"].tag)

    u_inflow = dolfinx.fem.Function(V)
    inflowdofsV = dolfiny.mesh.locate_dofs_topological(
        V, mesh_data.facet_tags, mesh_data.physical_groups["inflow"].tag
    )

    u_up = dolfinx.fem.Function(V)
    updofsV = dolfiny.mesh.locate_dofs_topological(
        V, mesh_data.facet_tags, mesh_data.physical_groups["up"].tag
    )

    def uinflow(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 0.0
        values[1] = -1.0
        return values

    u_inflow.interpolate(uinflow)

    p_bc = dolfinx.fem.Function(P)

    bcdofsP = dolfinx.fem.locate_dofs_geometrical(
        P, lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    )

    # Find common dofs at the top corners
    intersect = np.intersect1d(inflowdofsV, updofsV)

    # Remove for consistency
    inflowdofsV = np.setdiff1d(inflowdofsV, intersect)

    bcs = [dolfinx.fem.dirichletbc(u_inflow, inflowdofsV)]
    bcs.append(dolfinx.fem.dirichletbc(u_up, updofsV))
    bcs.append(dolfinx.fem.dirichletbc(p_bc, bcdofsP))

    lagrangedofsL = dolfiny.mesh.locate_dofs_topological(
        L, mesh_data.facet_tags, mesh_data.physical_groups["bottom"].tag
    )

    Vsize = V.dofmap.index_map_bs * V.dofmap.index_map.size_local
    Psize = P.dofmap.index_map_bs * P.dofmap.index_map.size_local

    rdofsV = np.arange(Vsize, dtype=np.int32)
    rdofsP = np.arange(Psize, dtype=np.int32)

    r = dolfiny.restriction.Restriction([V, P, L], [rdofsV, rdofsP, lagrangedofsL])

    opts = PETSc.Options("pipes")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 10
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    problem = dolfiny.snesproblem.SNESProblem(
        [F0, F1, F2], [u, p, lam], bcs=bcs, restriction=r, prefix="pipes"
    )
    s0, _s1, _s2 = problem.solve()

    assert problem.snes.getConvergedReason() > 0
