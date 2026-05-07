import xml.etree.ElementTree as ET

from mpi4py import MPI

import dolfinx

import numpy as np

import dolfiny


def test_write_mesh_data(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, n := 3, n, n)
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(1, 3)

    tdim = mesh.topology.dim

    im_c = mesh.topology.index_map(tdim)
    cell_entities = np.arange(im_c.size_local + im_c.num_ghosts, dtype=np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cell_entities, tdim)

    fdim = tdim - 1
    facet_entities = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 0.0)
    )

    rdim = tdim - 2
    ridge_entities = np.array([0], dtype=np.int32)

    pdim = tdim - 3
    peak_entities = np.array([0], dtype=np.int32)

    mesh_data = dolfinx.io.gmsh.MeshData(
        mesh=mesh,
        physical_groups={"cells": {"tag": 1}, "facets": {"tag": 2}},
        cell_tags=cell_tags,
        facet_tags=dolfinx.mesh.meshtags(mesh, fdim, facet_entities, fdim),
        ridge_tags=dolfinx.mesh.meshtags(mesh, rdim, ridge_entities, rdim),
        peak_tags=dolfinx.mesh.meshtags(mesh, pdim, peak_entities, pdim),
    )

    output = tmp_path / "mesh_data.xdmf"
    with dolfiny.io.XDMFFile(MPI.COMM_SELF, str(output), "w") as xdmf:
        xdmf.write_mesh_data(mesh_data)

    root = ET.parse(output).getroot()
    information_nodes = list(root.iter("Information"))
    assert len(information_nodes) == 1
    assert information_nodes[0].attrib == {
        "Name": dolfiny.io.XDMFFile.KEYS_OF_MESHTAGS,
        "Value": "dict_keys(['cells', 'facets'])",
    }

    grid_names = [grid.attrib["Name"] for grid in root.iter("Grid")]
    assert grid_names.count("mesh") == 1
    assert grid_names.count("mesh_tags") == 4
