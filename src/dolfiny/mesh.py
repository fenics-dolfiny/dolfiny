from mpi4py import MPI

import dolfinx
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

    unique_indices, pos = np.unique(indices, return_index=True)

    # Check disjoint
    if unique_indices.size != indices.size:
        raise RuntimeError("Overlapping MeshTags are not supported.")

    mt = meshtags(mesh, dim, unique_indices, values[pos])

    return mt, keys


def tag_box_facets(
    mesh: dolfinx.mesh.Mesh, box_bounds: list[list[float]]
) -> tuple[dolfinx.mesh.MeshTags, dict[str, dolfinx.io.gmsh.PhysicalGroup]]:
    """Tag every facet of a box domain.

    Face in space direction d at the lower bound, where x[d]==box_bounds[0][d], is named face_xd_0.
    Face in space direction d at the upper bound, where x[d]==box_bounds[1][d], is named face_xd_1.
    """
    facet_dim = mesh.topology.dim - 1

    facets = np.empty((0,), dtype=np.int32)
    tags = np.empty((0,), dtype=np.int32)

    mt: dict[str, dolfinx.io.gmsh.PhysicalGroup] = {}

    for d in range(mesh.geometry.dim):
        for k, kx in enumerate((box_bounds[0][d], box_bounds[1][d])):
            facets_face = dolfinx.mesh.locate_entities(
                mesh, facet_dim, lambda x: np.isclose(x[d], kx)
            )

            # only tag locally owned indices
            facets_face = facets_face[facets_face < mesh.topology.index_map(facet_dim).size_local]
            facets_face = np.unique(facets_face)

            name = f"face_x{d}_{'min' if k == 0 else 'max'}"
            tag = 2 * d + k
            mt[name] = dolfinx.io.gmsh.PhysicalGroup(dim=facet_dim, tag=tag)

            facets = np.append(facets, facets_face)
            tags = np.append(tags, np.full_like(facets_face, tag))
    # Sort by facet
    perm = np.argsort(facets)
    facets = facets[perm]
    tags = tags[perm]
    return dolfinx.mesh.meshtags(mesh, facet_dim, facets, tags), mt
