import logging

from mpi4py import MPI

import basix
import dolfinx.mesh
import ufl
from dolfinx import cpp, default_real_type
from dolfinx.cpp.mesh import CellType
from dolfinx.graph import adjacencylist
from dolfinx.io import distribute_entity_data
from dolfinx.io.gmshio import ufl_mesh
from dolfinx.mesh import create_mesh, meshtags, meshtags_from_entities

import numpy as np


def gmsh_to_dolfin(
    gmsh_model, tdim: int, comm=MPI.COMM_WORLD, partitioner=None, prune_y=False, prune_z=False
):
    """Converts gmsh model to mesh and mesh tags.

    Converts a gmsh model object into `dolfinx.Mesh` and `dolfinx.MeshTags`
    for physical tags.

    Parameters
    ----------
    gmsh_model
    tdim
        Topological dimension of the mesh
    comm: optional
    partitioner: optional
        Use given partitioner.
    prune_y: optional
        Prune y-components. Used to embed a flat geometries into lower dimension.
    prune_z: optional
        Prune z-components. Used to embed a flat geometries into lower dimension.

    Note
    ----
    User must call `geo.synchronize()` and `mesh.generate()` before passing the model into
    this method.

    """
    rank = comm.rank

    logger = logging.getLogger("dolfiny")

    if rank == 0:
        # Map from internal gmsh cell type number to gmsh cell name
        # see https://gitlab.onelab.info/gmsh/gmsh/blob/master/Common/GmshDefines.h#L75
        gmsh_cellname = {
            1: "line",
            2: "triangle",
            3: "quad",
            4: "tetra",
            5: "hexahedron",
            8: "line3",
            9: "triangle6",
            10: "quad9",
            11: "tetra10",
            12: "hexahedron27",
            15: "vertex",
            21: "triangle10",
            26: "line4",
            29: "tetra20",
            36: "quad16",
            #  92: 'hexahedron64',
        }

        gmsh_dolfin = {
            "vertex": (CellType.point, 0),
            "line": (CellType.interval, 1),
            "line3": (CellType.interval, 2),
            "line4": (CellType.interval, 3),
            "triangle": (CellType.triangle, 1),
            "triangle6": (CellType.triangle, 2),
            "triangle10": (CellType.triangle, 3),
            "quad": (CellType.quadrilateral, 1),
            "quad9": (CellType.quadrilateral, 2),
            "quad16": (CellType.quadrilateral, 3),
            "tetra": (CellType.tetrahedron, 1),
            "tetra10": (CellType.tetrahedron, 2),
            "tetra20": (CellType.tetrahedron, 3),
            "hexahedron": (CellType.hexahedron, 1),
            "hexahedron27": (CellType.hexahedron, 2),
            #  "hexahedron64": (CellType.hexahedron, 3),
        }

        # Number of nodes for gmsh cell type
        nodes = {
            "vertex": 1,
            "line": 2,
            "line3": 3,
            "line4": 4,
            "triangle": 3,
            "triangle6": 6,
            "triangle10": 10,
            "tetra": 4,
            "tetra10": 10,
            "tetra20": 20,
            "quad": 4,
            "quad9": 9,
            "quad16": 16,
            "hexahedron": 8,
            "hexahedron27": 27,
            #  'hexahedron64': 64,
        }

        node_tags, coord, param_coords = gmsh_model.mesh.getNodes()

        # Fetch elements for the mesh
        cell_types, cell_tags, cell_node_tags = gmsh_model.mesh.getElements(dim=tdim)

        unused_nodes = np.setdiff1d(node_tags, cell_node_tags)

        node_tags_sorted = np.argsort(node_tags)
        unused_nodes_perm = np.searchsorted(node_tags, unused_nodes, sorter=node_tags_sorted)
        unused_nodes_indices = node_tags_sorted[unused_nodes_perm]

        # Every node has 3 components in gmsh
        dim = 3
        points = np.reshape(coord, (-1, dim))

        # Delete unreferenced nodes
        points = np.delete(points, unused_nodes_indices, axis=0)
        node_tags = np.delete(node_tags, unused_nodes_indices)

        # Prepare a map from node tag to index in coords array
        nmap = np.argsort(node_tags - 1)
        cells = {}

        if len(cell_types) > 1:
            raise RuntimeError("Mixed topology meshes not supported.")

        try:
            cellname = gmsh_cellname[cell_types[0]]
            celltype = cell_types[0]
        except KeyError:
            raise RuntimeError(f"Gmsh cell code {cell_types[0]:d} not supported.")

        try:
            num_nodes = nodes[cellname]
        except KeyError:
            raise RuntimeError(
                f'Cannot determine number of nodes for Gmsh cell type "{cellname:s}".'
            )

        logger.info(f'Processing mesh of gmsh cell name "{cellname:s}"')

        # Shift 1-based numbering and apply node map
        cells[cellname] = nmap[cell_node_tags[0] - 1]
        cells[cellname] = np.reshape(cells[cellname], (-1, num_nodes))

        if prune_z:
            if not np.allclose(points[:, 2], 0.0):
                raise RuntimeError("Non-zero z-component would be pruned.")

            points = points[:, :-1]

        if prune_y:
            if not np.allclose(points[:, 1], 0.0):
                raise RuntimeError("Non-zero y-component would be pruned.")

            if prune_z:
                # In the case we already pruned z-component
                points = points[:, 0]
            else:
                points = points[:, [0, 2]]

        try:
            dolfin_cell_type, order = gmsh_dolfin[cellname]
        except KeyError:
            raise RuntimeError(
                f'Cannot determine dolfin cell type for Gmsh cell type "{cellname:s}".'
            )

        perm = cpp.io.perm_gmsh(dolfin_cell_type, num_nodes)
        logger.info(f"Mesh will be permuted with {perm}")
        cells_cellname: np.ndarray = cells[cellname][:, perm]

        logger.info(f"Constructing mesh for tdim: {tdim:d}, gdim: {points.shape[1]:d}")
        logger.info(f"Number of elements: {cells_cellname.shape[0]:d}")

        cells_shape, pts_shape, celltype = comm.bcast(
            [cells_cellname.shape, points.shape, celltype], root=0
        )
    else:
        cells_shape, pts_shape, celltype = comm.bcast([None, None, None], root=0)
        cells_cellname = np.empty((0, cells_shape[1]))
        points = np.empty((0, pts_shape[1]))

    mesh = create_mesh(
        comm,
        cells_cellname,
        points,
        ufl_mesh(celltype, pts_shape[1], dtype=default_real_type),
        partitioner,
    )
    mts = {}

    # Get physical groups (dimension, tag)
    pgdim_pgtags = comm.bcast(gmsh_model.getPhysicalGroups() if rank == 0 else None, root=0)

    for pgdim, pgtag in pgdim_pgtags:
        # For the current physical tag there could be multiple entities
        # e.g. user tagged bottom and up boundary part with one physical tag
        entity_tags = comm.bcast(
            gmsh_model.getEntitiesForPhysicalGroup(pgdim, pgtag) if rank == 0 else None, root=0
        )
        pg_tag_name = comm.bcast(
            gmsh_model.getPhysicalName(pgdim, pgtag) if rank == 0 else None, root=0
        )

        if pg_tag_name == "":
            pg_tag_name = f"tag_{pgtag:d}"

        if rank == 0:
            _mt_cells = []
            _mt_values = []

            for i, entity_tag in enumerate(entity_tags):
                pgcell_types, pgcell_tags, pgnode_tags = gmsh_model.mesh.getElements(
                    pgdim, entity_tag
                )

                assert len(pgcell_types) == 1
                pgcellname = gmsh_cellname[pgcell_types[0]]
                pgnum_nodes = nodes[pgcellname]

                # Shift 1-based numbering and apply node map
                pgnode_tags[0] = nmap[pgnode_tags[0] - 1]
                _mt_cells.append(pgnode_tags[0].reshape(-1, pgnum_nodes))
                _mt_values.append(np.full(_mt_cells[-1].shape[0], pgtag, dtype=np.int32))

            # Stack all topology and value data. This prepares data
            # for one MVC per (dim, physical tag) instead of multiple MVCs
            _mt_values_ndarray = np.hstack(_mt_values)
            _mt_cells_ndarray = np.vstack(_mt_cells)

            # Fetch the permutation needed for physical group
            pgdolfin_cell_type, pgorder = gmsh_dolfin[pgcellname]
            pgpermutation = cpp.io.perm_gmsh(pgdolfin_cell_type, pgnum_nodes)

            _mt_cells_ndarray = _mt_cells_ndarray[:, pgpermutation]

            logger.info(f"Constructing MVC for tdim: {pgdim:d}")
            logger.info(f"Number of data values: {_mt_values_ndarray.shape[0]:d}")

            mt_cells_shape, pgdim = comm.bcast([_mt_cells_ndarray.shape, pgdim], root=0)
        else:
            mt_cells_shape, pgdim = comm.bcast([None, None], root=0)
            _mt_cells_ndarray = np.empty((0, mt_cells_shape[1]), dtype=np.int64)
            _mt_values_ndarray = np.empty((0,), dtype=np.int32)

        local_entities, local_values = distribute_entity_data(
            mesh, pgdim, _mt_cells_ndarray, _mt_values_ndarray
        )

        # compute full connectivity for convenience
        for d in range(mesh.topology.dim + 1):
            mesh.topology.create_connectivity(pgdim, d)

        mt = meshtags_from_entities(mesh, pgdim, adjacencylist(local_entities), local_values)
        mt.name = pg_tag_name

        mts[pg_tag_name] = mt

    return mesh, mts


def msh_to_gmsh(msh_file, order=1, comm=MPI.COMM_WORLD):
    """Read msh file with gmsh and return the gmsh model.

    Parameters
    ----------
    msh_file:
        The msh file
    order: optional
        Adjust order of gmsh mesh cells
    comm: optional

    Returns
    -------
    gmsh_model:
        The gmsh model
    tdim:
        The highest topological dimension of the mesh entities

    """
    if comm.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.open(msh_file)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate()
        gmsh.model.mesh.setOrder(order)

    tdim = comm.bcast(
        max([dim for dim, _ in gmsh.model.getEntities()]) if comm.rank == 0 else None, root=0
    )

    return gmsh.model if comm.rank == 0 else None, tdim


def locate_dofs_topological(V, meshtags, value, exclude_dofs=None, unroll=False):
    """Identifies dofs of a given function space associated with a given meshtags value.

    Parameters
    ----------
    V: FunctionSpace
    meshtags: MeshTags object
    value: mesh tag value
    exclude_dofs: numpy array of dofs to exclude
    unroll: unroll dofs

    Returns
    -------
    The system dof indices.

    """
    from dolfinx import fem

    from numpy import setdiff1d, where

    from dolfiny import function

    if isinstance(value, list):
        match = []
        for v in value:
            match.extend(where(meshtags.values == v)[0])
    else:
        match = where(meshtags.values == value)[0]

    dofs = fem.locate_dofs_topological(V, meshtags.dim, meshtags.indices[match])

    if exclude_dofs is not None:
        dofs = setdiff1d(dofs, exclude_dofs)

    if unroll:
        dofs = function.unroll_dofs(dofs, V.dofmap.bs)

    return dofs


def locate_dofs_geometrical(V, meshtags, value, exclude_dofs=None, unroll=False):
    """Identifies dofs of a given function space associated with a given meshtags value.

    Parameters
    ----------
    V: FunctionSpace
    meshtags: MeshTags object
    value: mesh tag value
    exclude_dofs: numpy array of dofs to exclude
    unroll: unroll dofs

    Returns
    -------
    The system dof indices.

    """
    from dolfinx import fem

    from numpy import empty, int32, isclose, setdiff1d, where

    from dolfiny import function

    if isinstance(value, list):
        match = []
        for v in value:
            match.extend(where(meshtags.values == v)[0])
    else:
        match = where(meshtags.values == value)[0]

    if isinstance(V, tuple):
        V_ = V[0]
    else:
        V_ = V

    if meshtags.dim != 0:
        raise RuntimeError(f"MeshTags of dimension {meshtags.dim} > 0 are not supported.")

    def marker(x):
        if match.size == 0:
            return False
        else:
            # build vertex-to-node map
            mesh = V_.mesh

            connect_node_vertex = mesh.topology.connectivity(0, 0)
            connect_cell_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)

            vertices_per_cell = mesh.geometry.dofmap.shape[1]
            v2n = empty(connect_node_vertex.num_nodes, dtype=int32)
            c2v = connect_cell_vertex.array.reshape(-1, vertices_per_cell)

            v2n[c2v] = mesh.geometry.dofmap

            local_dof_idx = v2n[meshtags.indices[match]]

            return isclose(x.T, mesh.geometry.x[local_dof_idx]).all(axis=1)

    dofs = fem.locate_dofs_geometrical(V, marker)

    if exclude_dofs is not None:
        dofs = setdiff1d(dofs, exclude_dofs)

    if unroll:
        dofs = function.unroll_dofs(dofs, V_.dofmap.bs)

    return dofs


def merge_meshtags(mesh, mts, dim):
    """Merge multiple MeshTags into one.

    Parameters
    ----------
    mts:
        List of meshtags
    dim:
        Dimension of MeshTags which should be merged. Note it is
        not possible to merge MeshTags with different dimensions into one
        MeshTags object.

    """
    mts = [(mt, name) for name, mt in mts.items() if mt.dim == dim]
    if len(mts) == 0:
        raise RuntimeError(f"Cannot find MeshTags of dimension {dim}")

    indices = np.hstack([mt.indices for mt, name in mts])
    values = np.hstack([mt.values for mt, name in mts])

    keys = {}
    for mt, name in mts:
        comm = mt.topology.comm
        # In some cases this process could receive a MeshTags which are empty
        # We need to return correct "keys" mapping on each process, so this
        # communicates the value from processes which don't have empty meshtags
        if len(mt.values) == 0:
            value = -1
        else:
            if np.max(mt.values) < 0:
                raise RuntimeError("Not expecting negative values for MeshTags")
            value = int(mt.values[0])
        value = comm.allreduce(value, op=MPI.MAX)

        keys[name] = value

    indices, pos = np.unique(indices, return_index=True)
    mt = meshtags(mesh, dim, indices, values[pos])

    return mt, keys


def create_truss_x_braced_mesh(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.Mesh:
    """Create truss (interval) mesh with x-bracings from quad or hex mesh.

    Parameters
    ----------
    mesh:
        Mesh with quadrilateral (tdim=2) or hexahedral (tdim=3) elements.

    Returns
    -------
    Truss mesh with all possible bracings.

    """
    cell_type = mesh.topology.cell_type
    if cell_type not in (dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.CellType.hexahedron):
        raise RuntimeError("Truss x-braced mesh can only be constructed on quads or hexs.")

    if mesh.comm.size > 1:
        # TODO: limited by https://github.com/FEniCS/dolfinx/issues/3733
        raise RuntimeError("Ghost construction of branching meshes not supported in parallel.")

    if mesh.geometry.cmap.degree != 1:
        raise RuntimeError("Only 1st order coordinate elements are supported.")

    top = mesh.topology
    tdim = top.dim
    geo = mesh.geometry
    gdim = geo.dim

    top.create_connectivity(1, 0)
    e_to_v = top.connectivity(1, 0)
    top.create_connectivity(tdim, 0)
    c_to_v = top.connectivity(tdim, 0)

    # baseline are all edges of input mesh
    new_x = geo.x[:, :-1] if gdim == 2 else geo.x
    cells = e_to_v.array.reshape(-1, 2)

    if cell_type is dolfinx.mesh.CellType.quadrilateral:
        # 2----3
        # |    |
        # |    |
        # 0----1
        v = c_to_v.array.reshape(-1, 4)
        cells = np.append(cells, np.stack([v[:, 1], v[:, 2]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 0], v[:, 3]], axis=1), axis=0)
    else:
        #     6----------7
        #    /|         /|
        #   / |        / |
        #  /  |       /  |
        # 4----------5   |
        # |   |      |   |
        # |   2------+---3
        # |  /       |  /
        # | /        | /
        # |/         |/
        # 0----------1
        v = c_to_v.array.reshape(-1, 8)
        cells = np.append(cells, np.stack([v[:, 0], v[:, 3]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 1], v[:, 2]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 2], v[:, 7]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 3], v[:, 6]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 0], v[:, 6]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 2], v[:, 4]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 1], v[:, 7]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 3], v[:, 5]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 0], v[:, 5]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 1], v[:, 4]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 4], v[:, 7]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 5], v[:, 6]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 0], v[:, 7]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 3], v[:, 4]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 1], v[:, 6]], axis=1), axis=0)
        cells = np.append(cells, np.stack([v[:, 2], v[:, 5]], axis=1), axis=0)

        # remove duplicate edges (only exist for hexahedron)
        cells = np.unique(cells, axis=0)

    cells = cells.astype(np.int64)  # promote to global indices

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(gdim,)))
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, new_x, element)
