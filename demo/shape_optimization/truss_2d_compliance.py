from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl

import numpy as np
import pyvista as pv

import dolfiny
import dolfiny.taoblockproblem

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


n = 10  # 5
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


F = dolfinx.fem.Function(V)
mesh.topology.create_connectivity(0, 1)
load_dofs = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, load_vertex)
F.x.array[load_dofs] = -1

dofs_fixed = dolfinx.fem.locate_dofs_topological(V, 0, fixed_vertices)
bcs = [dolfinx.fem.dirichletbc(np.zeros(2), dofs_fixed, V)]


u = dolfinx.fem.Function(V, name="Displacement")

E = dolfinx.fem.Constant(mesh, 200e3)
# S = dolfinx.fem.Constant(mesh, 1.0)
S = dolfinx.fem.Function(Vs, name="Cross-section area")
S.x.array[:] = 0.5
# S.x.array[:] = np.random.default_rng(seed=42).random(S.x.array.size)

ε = ufl.dot(ufl.dot(ufl.grad(u), t_ufl), t_ufl)  # strain
N = E * S * ε  # normal_force

# F0 = dolfinx.fem.Constant(mesh, np.zeros(2))
J = 1 / 2 * ufl.inner(N, ε) * ufl.dx  #  - ufl.inner(F, u) * ufl.dx

R = ufl.derivative(J, u, ufl.TestFunction(V))
R = ufl.replace(R, {u: ufl.TrialFunction(V)})
a, L = ufl.lhs(R), ufl.rhs(R)
a_form = dolfinx.fem.form(a)
# L_form = dolfinx.fem.form(L)

f = u.x.petsc_vec.copy()
f[load_dofs] = -1
f.assemble()


A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
A.assemble()

state_solver = PETSc.KSP().create(mesh.comm)
state_solver.setOperators(A)
state_solver.setType(PETSc.KSP.Type.PREONLY)
state_solver.getPC().setType(PETSc.PC.Type.LU)

b = u.x.petsc_vec.copy()
b.zeroEntries()
b[load_dofs] = -1


@dolfiny.taoblockproblem.link_state(S)
def C(tao, x) -> float:
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)
    A.assemble()

    state_solver.solve(b, u.x.petsc_vec)
    return f.dot(u.x.petsc_vec)


# gx = dolfinx.fem.form(ufl.derivative(a, S, ufl.TestFunction(Vs)))
# p = -u
p = dolfinx.fem.Function(V)
gx = dolfinx.fem.form(ufl.derivative(ufl.action(ufl.derivative(J, u), p), S, ufl.TestFunction(Vs)))
Gx = dolfinx.fem.assemble_vector(gx)


@dolfiny.taoblockproblem.link_state(S)
def JC(tao, x, J):
    # state_solver.solve(b, u.x.petsc_vec) # TODO: remove?
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)
    A.assemble()

    state_solver.solve(b, u.x.petsc_vec)

    J.zeroEntries()
    u.x.petsc_vec.copy(p.x.petsc_vec)
    p.x.petsc_vec.scale(-1)

    dolfinx.fem.petsc.assemble_vector(J, gx)


g = [S * ufl.dx <= 50]  # 25
Jg = [[ufl.TestFunction(Vs) * ufl.dx]]
# S.x.array[:] = 1

# print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(g[0].lhs)))
# exit()


opts = PETSc.Options("truss")
opts["tao_type"] = "almm"
opts["tao_almm_type"] = "classic"
opts["tao_almm_subsolver_tao_monitor"] = ""
opts["tao_monitor"] = ""
opts["tao_grtol"] = 0.0
opts["tao_gatol"] = 1e-6
opts["tao_catol"] = 1e-3
problem = dolfiny.taoblockproblem.TAOBlockProblem(
    C, [S], bcs=bcs, J=(JC, S.x.petsc_vec.copy()), h=g, Jh=Jg, prefix="truss"
)

ub = dolfinx.fem.Function(Vs, name="upper_bound")
ub.x.array[:] = 1
# bc.set(ub.x.array, alpha=1)

lb = dolfinx.fem.Function(Vs, name="lower_bound")
lb.x.array[:] = 1e-1
# bc.set(lb.x.array, alpha=1)

# TODO: move into problem
problem.tao.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)
problem.solve([lb])


A.zeroEntries()
dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)
A.assemble()

state_solver.solve(b, u.x.petsc_vec)
print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(g[0].lhs)))
# exit()

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
plotter.add_mesh(grid, show_edges=True, color="black", opacity=0.0)  # render_lines_as_tubes=True

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
