from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
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

V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
Vs = dolfinx.fem.functionspace(mesh, ("DG", 0))
Vv = dolfinx.fem.functionspace(mesh, ("DG", 0, (2,)))

dx_dX = ufl.Jacobian(mesh)[:, 0]
t_ufl = dx_dX / ufl.sqrt(ufl.inner(dx_dX, dx_dX))
t = dolfinx.fem.Function(Vv, name="Tangent_vector")
t.interpolate(dolfinx.fem.Expression(t_ufl, Vv.element.interpolation_points))

# To apply point loads on the truss nodes, it will be more convenient to define a piecewise linear function `F` and set the corresponding degrees of freedom with the corresponding nodal forces. In this case, we apply vertical downwards concentrated forces of intensity 1 on the bottom nodes which we retrieve from the facet tag `1`. We also retrieve the left and right dofs for applying Dirichlet boundary conditions

# +
F = dolfinx.fem.Function(V)
mesh.topology.create_connectivity(0, 1)
load_dofs = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, load_vertex)
F.x.array[load_dofs] = -1

dofs_fixed = dolfinx.fem.locate_dofs_topological(V, 0, fixed_vertices)
bcs = [dolfinx.fem.dirichletbc(np.zeros(2), dofs_fixed, V)]

du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
u = dolfinx.fem.Function(V, name="Displacement")

E = dolfinx.fem.Constant(mesh, 200e3)
# S = dolfinx.fem.Constant(mesh, 1.0)
S = dolfinx.fem.Function(Vs, name="Cross-section area")
S.x.array[:] = 1.0
# S.x.array[:] = np.random.default_rng(seed=42).random(S.x.array.size)

ε = ufl.dot(ufl.dot(ufl.grad(u), t_ufl), t_ufl)  # strain
N = E * S * ε  # normal_force

F0 = dolfinx.fem.Constant(mesh, np.zeros(2))
J = 1 / 2 * ufl.inner(N, ε) * ufl.dx - ufl.inner(F0, u) * ufl.dx

R = ufl.derivative(J, u, ufl.TestFunction(V))
R = ufl.replace(R, {u: ufl.TrialFunction(V)})
a, L = ufl.lhs(R), ufl.rhs(R)
a_form = dolfinx.fem.form(a)
L_form = dolfinx.fem.form(L)

A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
A.assemble()
b = dolfinx.fem.petsc.create_vector(L_form)

b.array[:] = F.x.array[:]

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

solver.solve(b, u.x.petsc_vec)
u.x.scatter_forward()

V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
N_exp = dolfinx.fem.Expression(N, V0.element.interpolation_points)
N = dolfinx.fem.Function(V0, name="Normal_force")
N.interpolate(N_exp)

plotter = pv.Plotter()
plotter.window_size = [1920, 1080]
grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))

u_3D = np.zeros((V.dofmap.index_map.size_local, 3))
u_3D[:, :2] = u.x.array.reshape(-1, 2)
# u_3D[:, 2] = (
#     1e-3  # slightly offset to avoid overlap in xy view with underlying undeformed mesh
# )
grid.point_data["Deflection"] = u_3D
grid.cell_data[S.name] = S.x.array
plotter.add_mesh(grid, show_edges=True, color="black", opacity=1.0)  # render_lines_as_tubes=True

warped = grid.warp_by_vector("Deflection", factor=2000.0)

plotter.add_mesh(
    warped,
    show_edges=True,
    color="black",
    scalars=S.name,
    # line_width=5.0, # TODO: bug!
    clim=[0, 1],
    scalar_bar_args={"vertical": False},  # TODO: overlapping after zoom
    opacity=0.7,
)

for point in mesh.geometry.x[fixed_vertices]:
    circle = pv.Circle(radius=0.2 / n)
    circle = circle.translate(point)
    plotter.add_mesh(circle, color="red", opacity=1.0)


# TODO: warp vector start
load = pv.Arrow(
    start=mesh.geometry.x[load_vertex], direction=(0, -1, 0), scale=2 / n
)  # TODO use f later
plotter.add_mesh(load, color="green")

plotter.add_axes()
plotter.view_xy()
plotter.show()
