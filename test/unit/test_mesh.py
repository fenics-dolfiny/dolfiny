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
    mesh = dolfiny.mesh_generation.create_truss_x_braced_mesh(orig_mesh, comm=MPI.COMM_SELF)

    assert mesh.geometry.x.shape == orig_mesh.geometry.x.shape  # equal up to reorder
    mesh.topology.create_connectivity(0, 1)
    v_to_e = mesh.topology.connectivity(0, 1)
    for i in range(mesh.topology.index_map(0).size_local):
        links = v_to_e.links(i)
        assert links.size == (3 if gdim == 2 else 7)  # connected to all vertices up to self


@pytest.mark.parametrize("dim", (1, 2, 3))
def test_tag_box_facets(dim):
    comm = MPI.COMM_WORLD

    box_bounds = [[-1] * dim, [1] * dim]
    if dim == 1:
        mesh = dolfinx.mesh.create_interval(comm, 10, [-1, 1])
    elif dim == 2:
        mesh = dolfinx.mesh.create_rectangle(comm, box_bounds, [10, 10])
    elif dim == 3:
        mesh = dolfinx.mesh.create_box(comm, box_bounds, [10, 10, 10])
    else:
        assert False

    meshtags, mts = dolfiny.mesh.tag_box_facets(mesh, box_bounds)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=meshtags)

    for d in range(dim):
        for side in ("min", "max"):
            name = f"face_x{d}_{side}"
            assert name in mts.keys()
            face_dim, tag = mts[name]
            assert face_dim == dim - 1

            area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(tag)))
            area = comm.allreduce(area)

            assert area == pytest.approx(2 ** (dim - 1))
