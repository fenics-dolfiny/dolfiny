from mpi4py import MPI

import dolfinx

import gmsh
import numpy as np
import pytest

from dolfiny.gmsh import wireframe_mesh


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Sequential only.")
def test_wireframe_mesh_triangle() -> None:
    gmsh.initialize()
    gmsh.clear()

    gmsh.model.add("triangle")

    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0)
    p3 = gmsh.model.geo.addPoint(0, 1, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p1)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 10.0)  # single element
    gmsh.model.mesh.generate(2)

    pg_nodes = gmsh.model.addPhysicalGroup(0, [p1, p2], name="pg_nodes")
    pg_tri = gmsh.model.addPhysicalGroup(2, [surf], name="pg_tri")

    wireframe_mesh(gmsh.model)

    meshdata = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
    mesh = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facet_tags = meshdata.facet_tags

    assert mesh.topology.dim == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.index_map(0).size_local == 3
    assert mesh.topology.index_map(1).size_local == 3
    assert cell_tags is not None
    assert cell_tags.name == "cell_tags"
    assert np.all(cell_tags.values == pg_tri)
    assert facet_tags is not None
    assert facet_tags.name == "facet_tags"
    assert np.count_nonzero(facet_tags.values == pg_nodes) == 2

    gmsh.clear()
    gmsh.finalize()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Sequential only.")
def test_wireframe_mesh_complex() -> None:
    """Test wireframe extraction on a cell complex (mixed dimensional mesh), consisting of
    one line, one triangle and one tetrahedron.

               p5
              /| \
             / |  \
            /  |   \
      p1--p2--p3----p4
               |   /
               |  /
               p6

    Note: p6 is out of the z=0 plane and edge p5-p6 not visible.
    """

    gmsh.initialize()
    gmsh.clear()

    gmsh.model.add("complex")

    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0)
    p3 = gmsh.model.geo.addPoint(2, 0, 0)
    p4 = gmsh.model.geo.addPoint(3, 0, 0)
    p5 = gmsh.model.geo.addPoint(2, 1, 0)
    p6 = gmsh.model.geo.addPoint(2, 0, 1)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p5)
    l4 = gmsh.model.geo.addLine(p2, p5)
    l5 = gmsh.model.geo.addLine(p3, p4)
    l6 = gmsh.model.geo.addLine(p4, p5)
    l7 = gmsh.model.geo.addLine(p3, p6)
    l8 = gmsh.model.geo.addLine(p5, p6)
    l9 = gmsh.model.geo.addLine(p4, p6)

    t1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l3, l4], reorient=True)])
    t2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l5, l6], reorient=True)])
    t3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l7, l8], reorient=True)])
    t4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l7, l9], reorient=True)])
    t5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l6, l8, l9], reorient=True)])

    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([t2, t3, t4, t5])])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 10.0)  # single element
    gmsh.model.mesh.generate(3)

    pg_line = gmsh.model.addPhysicalGroup(1, [l1], name="pg_line")
    pg_tri = gmsh.model.addPhysicalGroup(2, [t1], name="pg_tri")
    pg_tet = gmsh.model.addPhysicalGroup(2, [t4, t5], name="pg_tet")

    wireframe_mesh(gmsh.model)

    meshdata = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
    mesh = meshdata.mesh
    cell_tags = meshdata.cell_tags

    assert mesh.topology.dim == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.index_map(0).size_local == 6
    assert mesh.topology.index_map(1).size_local == 9
    assert cell_tags is not None
    assert cell_tags.name == "cell_tags"
    assert np.count_nonzero(cell_tags.values == pg_line) == 1
    assert np.count_nonzero(cell_tags.values == pg_tri) == 3
    assert np.count_nonzero(cell_tags.values == pg_tet) == 5

    gmsh.clear()
    gmsh.finalize()
