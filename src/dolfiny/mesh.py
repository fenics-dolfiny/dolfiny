from mpi4py import MPI

import basix
import dolfinx.mesh
import ufl
from dolfinx.mesh import meshtags

import numpy as np


def msh_to_gmsh(msh_file, order=1, comm=MPI.COMM_WORLD):
    """Read msh file with gmsh and return the gmsh model.

    Parameters
    ----------
    msh_file:
        The msh file
    order: optional
        Adjust order of gmsh mesh cells
    comm: optional
        Communicator over which the tdim is broadcasted

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
    """Identify dofs of a given function space associated with a given meshtags value.

    Parameters
    ----------
    V:
        FunctionSpace
    meshtags:
        MeshTags object
    value:
        mesh tag value
    exclude_dofs:
        numpy array of dofs to exclude
    unroll:
        unroll dofs

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
    """Identify dofs of a given function space associated with a given meshtags value.

    Parameters
    ----------
    V:
        FunctionSpace
    meshtags:
        MeshTags object
    value:
        mesh tag value
    exclude_dofs:
        numpy array of dofs to exclude
    unroll:
        unroll dofs

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
    mesh:
        Mesh associated with mesh tags
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
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, element, new_x)
