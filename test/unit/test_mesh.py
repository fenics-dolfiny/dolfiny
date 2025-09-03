from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny


def test_simple_triangle():
    if MPI.COMM_WORLD.rank == 0:
        import gmsh

        gmsh.initialize(interruptible=False)
        gmsh.model.add("test")

        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)
        p2 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
        p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
        p4 = gmsh.model.geo.addPoint(1.0, 0.5, 0.0)

        l0 = gmsh.model.geo.addLine(p0, p1)
        l1 = gmsh.model.geo.addCircleArc(p1, p4, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p0)

        cl0 = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
        s0 = gmsh.model.geo.addPlaneSurface([cl0])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [l0, l2], 2)
        gmsh.model.setPhysicalName(1, 2, "sides")

        gmsh.model.addPhysicalGroup(1, [l1], 3)
        gmsh.model.setPhysicalName(1, 3, "arc")

        gmsh.model.addPhysicalGroup(2, [s0], 4)
        gmsh.model.setPhysicalName(2, 4, "surface")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate()
        gmsh.model.mesh.setOrder(2)

        gmsh_model = gmsh.model

    else:
        gmsh_model = None

    mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, MPI.COMM_WORLD, rank=0, gdim=2)
    mesh = mesh_data.mesh

    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 2
    assert mesh_data.physical_groups["arc"][0] == 1

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)
        mesh.topology.create_connectivity(1, 2)
        file.write_meshtags(mesh_data.cell_tags, mesh.geometry)

    ds = ufl.Measure("ds", subdomain_data=mesh_data.facet_tags, domain=mesh)

    form = dolfinx.fem.form(
        1.0 * ds(mesh_data.physical_groups["sides"].tag)
        + 1.0 * ds(mesh_data.physical_groups["arc"].tag)
    )
    val = dolfinx.fem.assemble_scalar(form)

    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert np.isclose(val, 2.0 + 2.0 * np.pi * 0.5 / 2.0, rtol=1.0e-3)

    dx = ufl.Measure("dx", subdomain_data=mesh_data.cell_tags, domain=mesh)

    form = dolfinx.fem.form(1.0 * dx(mesh_data.physical_groups["surface"].tag))
    val = dolfinx.fem.assemble_scalar(form)

    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert np.isclose(val, 1.0 + np.pi * 0.5**2 / 2.0, rtol=1.0e-3)


@pytest.mark.parametrize("gdim", (2, 3))
def test_truss_x_braced(gdim):
    orig_mesh = (
        dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1, dolfinx.mesh.CellType.quadrilateral)
        if gdim == 2
        else dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, dolfinx.mesh.CellType.hexahedron)
    )
    mesh = dolfiny.mesh.create_truss_x_braced_mesh(orig_mesh)

    assert mesh.geometry.x.shape == orig_mesh.geometry.x.shape  # equal up to reorder
    mesh.topology.create_connectivity(0, 1)
    v_to_e = mesh.topology.connectivity(0, 1)
    for i in range(v_to_e.num_nodes):
        links = v_to_e.links(i)
        assert links.size == (3 if gdim == 2 else 7)  # connected to all vertices up to self
