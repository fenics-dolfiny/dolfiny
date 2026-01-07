from mpi4py import MPI

import basix
import dolfinx
import ufl

import numpy as np


def create_truss_x_braced_mesh(
    mesh: dolfinx.mesh.Mesh | None, comm=MPI.COMM_WORLD
) -> dolfinx.mesh.Mesh:
    """Create truss (interval) mesh with x-bracings from quad or hex mesh.

    Parameters
    ----------
    mesh:
        Mesh with quadrilateral (tdim=2) or hexahedral (tdim=3) elements.
    comm:
        Communicator to construct the truss mesh on.

    Note:
        The input mesh needs to be provided on the 0-th process - non parallelized.

    Returns
    -------
    Truss mesh with all possible bracings.

    """
    if mesh is not None and mesh.comm.size > 1:
        raise RuntimeError("Reference mesh must be provided in non-distributed manner.")

    if comm.rank != 0:
        gdim, max_facet_to_cell_links = comm.bcast(0, root=0)
        element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(gdim,)))
        cells = np.zeros((0, 2), dtype=np.int64)
        new_x = np.zeros((0, gdim), dtype=np.float64)
        return dolfinx.mesh.create_mesh(
            comm, cells, element, new_x, max_facet_to_cell_links=max_facet_to_cell_links
        )

    assert mesh is not None

    if mesh.geometry.cmap.degree != 1:
        raise RuntimeError("Only 1st order coordinate elements are supported.")

    cell_type = mesh.topology.cell_type
    if cell_type not in (dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.CellType.hexahedron):
        raise RuntimeError("Truss x-braced mesh can only be constructed on quads or hexs.")

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
        max_facet_to_cell_links = 8
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
        max_facet_to_cell_links = 26
        # 6 axis aligned
        # 12 in plane diagonals
        # 8 'true' diagonals
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
    comm.bcast((gdim, max_facet_to_cell_links), root=0)
    return dolfinx.mesh.create_mesh(
        comm, cells, element, new_x, max_facet_to_cell_links=max_facet_to_cell_links
    )
