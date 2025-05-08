from mpi4py import MPI

import basix
import dolfinx
import ufl

import numpy as np
import pyvista as pv

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("Parallelization not supported.")


def create_truss_x_bracing_mesh(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.Mesh:
    # TODO: assert quads and space dimension
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim - 2)
    f_to_v = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim - 2)

    new_x = mesh.geometry.x[:, :-1]
    new_cells = f_to_v.array.reshape(
        -1, 2
    )  # baseline are all facets/line elements of the original mesh

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 2)
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 2)

    for i in range(c_to_v.num_nodes):
        quad_vertices = c_to_v.links(i)
        # TODO: document - assert?
        # cross = [[quad_vertices[0], quad_vertices[2]], [quad_vertices[1], quad_vertices[3]]]
        cross = [[quad_vertices[1], quad_vertices[2]], [quad_vertices[0], quad_vertices[3]]]
        new_cells = np.append(new_cells, cross, axis=0)

    new_cells = new_cells.astype(np.int64)  # promote to global indices

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(2,)))  # TODO: dtype?
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, new_cells, new_x, element)


n = 10
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0, 0], [2, 1]],
    n * np.array([2, 1]),
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)
mesh = create_truss_x_bracing_mesh(mesh)

fixed_vertices = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0))
load_vertex = dolfinx.mesh.locate_entities(
    mesh, 0, lambda x: np.isclose(x[0], 2) & np.isclose(x[1], 0)
)

grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
plotter = pv.Plotter()

plotter.add_mesh(grid, show_edges=True, color="black")  # render_lines_as_tubes=True

for point in mesh.geometry.x[fixed_vertices]:
    circle = pv.Circle(radius=0.2 / n)
    circle = circle.translate(point)
    plotter.add_mesh(circle, color="red", opacity=0.7)


load = pv.Arrow(
    start=mesh.geometry.x[load_vertex], direction=(0, -1, 0), scale=2 / n
)  # TODO use f later
plotter.add_mesh(load, color="green")

plotter.add_axes()
plotter.view_xy()
plotter.show()
