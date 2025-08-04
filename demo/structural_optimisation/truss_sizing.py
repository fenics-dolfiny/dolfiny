# %% [markdown]
# # Truss Sizing Optimisation with 🐬
#
# This demo showcases how a sizing optimisation of a truss structure can be addressed.
# For this example we will optimise an *inverse truss bridge* design.
#
# In particular this demo emphasizes
# 1. implementation of a linear elastic truss model with `ufl`,
# 2. creation of (braced) truss meshes, and
# 3. the interface to PETSc/TAO for otimisation solvers.
#
# %%
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl

import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import pyvista as pv
from numpy import typing as npt

import dolfiny
import dolfiny.taoproblem
from dolfiny.mesh import create_truss_x_braced_mesh

# %% [markdown]
# ## Generation of Truss Meshes
#
# Truss structures make up an interesting example of the concept of *geometric-* and
# *topological dimension*, let those be denoted as $d_g$ and $d_t$, which is also used throughout
# DOLFINx {cite:p}`2023dolfinx`.
#
# The geometric dimension is the dimension of the space in which the mesh $\mathcal{T}$ is embedded,
# $d_g \in \{ 1, 2, 3 \}$ and $\mathcal{T} \subset \mathbb{R}^{d_g}$, and the topological dimension
# is the dimension of the cells of the mesh in the reference configuration, $d_t = \dim (\hat{T})$
# for $\hat{T}$ the reference cell of $T \in \mathcal{T}$.
#
# ```{note}
#   Since a mesh needs to embedded into a bigger space, it holds $d_t \leq d_g$.
# ```
#
# The usual finite element text-book setting applies to the case $d_t = d_g$, for example a triangle
# mesh is only considered in $\mathbb{R}^2$ and not in $\mathbb{R}^3$.
#
# Trusses are made up of lines, or, domain inspecifc, intervals, which are one-dimensional entities
# $d_t=1$ and form planar or volumetric structures, $d_g \in \{ 2, 3 \}$.
# Here, we consider the volumetric case, $d_g=3$.
#
# The bridge dimensions (length, height, width) are $(100,\, 20,\, 10)$ all in meters and we create
# the truss mesh from the edges of a (regular) hexahedral mesh, with trusses of size $h=5$, and
# introduce all possible bracings to it.
# This is what the `create_truss_x_braced_mesh` functionality simplifies.
# Given a hexehedral mesh, it generates the edge skeleton together with all possible bracings.
#
# ```{note}
#   Dolfinx currently does not support the generation of parallel branching meshes.
#   More precisely, the dual graph computation is not capable of detecting cross process boundary
#   edges, compare [dolfinx#3733](https://github.com/FEniCS/dolfinx/issues/3733).
#   Therefore this demo is only capable of sequential execution.
# ```
#
# %%
comm = MPI.COMM_WORLD
if comm.size > 1:
    raise RuntimeError("Parallelization not supported.")

dim = np.array([100, 20, 10], dtype=np.float64)
h = 5
mesh = create_truss_x_braced_mesh(
    dolfinx.mesh.create_box(
        comm,
        [np.zeros_like(dim), dim],
        (dim / h).astype(np.int32),  # type: ignore
        dolfinx.mesh.CellType.hexahedron,
    )
)

# %% [markdown]
# We create one functionspace $V_u = P_1(\mathcal{T})$ for the displacement field $u \in V_u$ and
# one for the cross sectional area $s$ of the trusses, which we model as constant over each element,
# so $V_s = DG_0(\mathcal{T})$.
# %%
V_u = dolfinx.fem.functionspace(mesh, ("CG", 1, (3,)))
u = dolfinx.fem.Function(V_u, name="displacement")

V_s = dolfinx.fem.functionspace(mesh, ("DG", 0))
s = dolfinx.fem.Function(V_s, name="cross-sectional-area")

# %% [markdown]
# ## Boundary Conditions and Load Surface
# For the realisation of an inverse truss bridge, we will fix the trusses on the left and right top
# edges and apply a vertical load, i.e. in $y$-direction, on the bridge deck, which we assume to
# span the whole upper surface/side of the truss mesh, excluding the fixed edges.
# %%
mesh.topology.create_connectivity(0, 1)
vertices_fixed = dolfinx.mesh.locate_entities(
    mesh,
    0,
    lambda x: (np.isclose(x[0], 0) | np.isclose(x[0], dim[0])) & np.isclose(x[1], dim[1]),
)

dofs_fixed = dolfinx.fem.locate_dofs_topological(V_u, 0, vertices_fixed)
bcs = [dolfinx.fem.dirichletbc(np.zeros(3, dtype=np.float64), dofs_fixed, V_u)]

vertices_load = dolfinx.mesh.locate_entities(
    mesh, 0, lambda x: np.isclose(x[1], dim[1]) & np.greater(x[0], 0) & np.less(x[0], dim[0])
)
dofs_load = dolfinx.fem.locate_dofs_topological(V_u.sub(1), 0, vertices_load)

# %% tags=["hide-input"]
pv.set_jupyter_backend("static")
pv_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
plotter = pv.Plotter(window_size=[3840, 2160])
plotter.add_mesh(pv_grid, color="black", line_width=1.5)

for v in vertices_load:
    v_coords = mesh.geometry.x[v]
    arrow = pv.Arrow(v_coords + np.array([0, 4, 0]), [0, -1, 0], scale=4)
    plotter.add_mesh(arrow, color="green", opacity=0.5)

for v in vertices_fixed:
    v_coords = mesh.geometry.x[v]
    sphere = pv.Sphere(0.5, v_coords)
    plotter.add_mesh(sphere, color="red")

plotter.view_xy()
plotter.add_axes()
plotter.camera.elevation += 20
plotter.camera.zoom(1.7)
plotter.show()

# %% [markdown]
# ## Truss Model and Forward Problem
#
# We rely on a linear elastic truss model, same as the one presented in {cite:p}`bleyer2024comet`.
# We choose a *Young's modulus* of $E = 200\, \text{GPa}$ and apply a total load of
# $300\, \text{kN}$, equally distributed across the loaded nodes.
#
# ```{note}
#   The point loads can currently not be taken care of inside the variational assembly of `FENICSx`
#   and require a custom handling of the load vector and thus solver.
#   This is limited by the fact, that both `FFCx` and `DOLFINx` do not support the `ufl` vertex
#   vertex integrals.
#   Changes proposed in [FFCx#764](https://github.com/FEniCS/ffcx/pull/764) and
#   [DOLFINx#3726](https://github.com/FEniCS/dolfinx/pull/3726) resolve this.
# ```
# %%
tangent = ufl.Jacobian(mesh)[:, 0]
tangent /= ufl.sqrt(ufl.inner(tangent, tangent))

F = dolfinx.fem.Function(V_u, name="load")
F.x.array[dofs_load] = -300 * 1e3 / vertices_load.size  # [F] = N

E = 200.0 * 1e9  # [E] = Pa = N/m^2
ε = ufl.dot(ufl.dot(ufl.grad(u), tangent), tangent)  # strain
N = E * s * ε  # normal_force

E = 1 / 2 * ufl.inner(N, ε) * ufl.dx

δE = ufl.derivative(E, u)
δδE = ufl.derivative(δE, u)
a_form = dolfinx.fem.form(δδE)

dolfinx.fem.apply_lifting(F.x.array, [a_form], [bcs])
s.x.array[:] = 1 / 20
A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs)
A.assemble()

opts = PETSc.Options("state_solver")  # type: ignore
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

state_solver = PETSc.KSP().create(comm)  # type: ignore
state_solver.setOptionsPrefix("state_solver")
state_solver.setFromOptions()
state_solver.setOperators(A)
state_solver.setUp()

# %% [markdown]
# ## Optimisation of Cross-sectional Areas
#
# The truss sizing optimisation problem from {cite:p}`Christensen2009` we are interested in is given
# (in reduced form) by
#
# $$
#   \min_{s \in V_s} C(u(s))
#   \quad \text{subject to} \quad
#   s_\text{min} \leq s(x) \leq s_\text{max}, \
#   \int_\Omega s \, \text{d}x = \frac{1}{20} \int_\Omega s_\text{max} \, \text{d} x
# $$
#
# where $u$ is the unique displacement field associated with a given sizing $s$.
#
# For the bounds of the truss cross sectional area we consider an upper bound of
# $s_\text{max} = 10^{-2} \, \text{m}^2$ and set the the lower bound to
# $s_\text{min} = 10^{-3} s_\text{max}$.


# %%
@dolfiny.taoproblem.sync_functions([s])
def C(tao, x) -> float:
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)  # type: ignore
    A.assemble()

    state_solver.solve(F.x.petsc_vec, u.x.petsc_vec)

    val: float = F.x.petsc_vec.dot(u.x.petsc_vec)
    return val


# p = -u
p = dolfinx.fem.Function(V_u)
gx = dolfinx.fem.form(ufl.derivative(ufl.derivative(E, u, p), s))
Gx = dolfinx.fem.assemble_vector(gx)


@dolfiny.taoproblem.sync_functions([s])
def JC(tao, x, J):
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)
    A.assemble()
    state_solver.solve(F.x.petsc_vec, u.x.petsc_vec)

    J.zeroEntries()
    u.x.petsc_vec.copy(p.x.petsc_vec)
    p.x.petsc_vec.scale(-1)

    dolfinx.fem.petsc.assemble_vector(J, gx)


s_max = np.float64(1e-2)
s_min = np.float64(1e-3 * s_max)
max_vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, s_max) * ufl.dx))
g = [s * ufl.dx == max_vol / 20]

opts = PETSc.Options("truss")  # type: ignore
opts["tao_type"] = "almm"
opts["tao_almm_type"] = "phr"
opts["tao_grtol"] = 5e-4
opts["tao_gatol"] = 5e-4
opts["tao_catol"] = 1e-2
opts["tao_max_it"] = 100
opts["tao_recycle_history"] = True

problem = dolfiny.taoproblem.TAOProblem(
    C, [s], bcs=bcs, J=(JC, s.x.petsc_vec.copy()), g=g, lb=s_min, ub=s_max, prefix="truss"
)

compliance: npt.NDArray[np.float64] = np.zeros(100)
volume: npt.NDArray[np.float64] = np.zeros(100)


def monitor(tao):
    it = tao.getIterationNumber()
    compliance[it] = tao.getObjectiveValue()
    volume[it] = problem._g[1][0] + g[0].rhs


problem.tao.setMonitor(monitor)
problem.solve()

A.zeroEntries()
dolfinx.fem.petsc.assemble_matrix(A, a_form, bcs)  # type: ignore
A.assemble()

state_solver.solve(F.x.petsc_vec, u.x.petsc_vec)

# %% tags=["hide-input"]
# verify result|
if problem.tao.getConvergedReason() <= 0:
    raise RuntimeError("Optimisation did not converge.")

if np.abs(dolfinx.fem.assemble_scalar(dolfinx.fem.form(g[0].lhs)) - g[0].rhs) >= 1e-2:
    raise RuntimeError("Volume constraint violated.")

if np.any(s.x.array < s_min):
    raise RuntimeError("Lower bound violated.")

if np.any(s.x.array > s_max):
    raise RuntimeError("Upper bound violated.")

# %% [markdown]
# Convergence of the optimisation, compliance vs. volume with shown volume constraint.
# Relative to the first iterate, for compliance, and relative to the uppoer bound, for volume.
# %% tags=["hide-input"]
matplotlib_inline.backend_inline.set_matplotlib_formats("png")
it = problem.tao.getIterationNumber()
compliance = compliance[:it]
volume = volume[:it]

fig, ax1 = plt.subplots(dpi=400)
ax1.grid()
ax1.set_xlim(0, it - 1)
ax1.set_xlabel("Iteration")
ax1.set_xticks(range(0, it))
ax1.set_ylabel("Compliance")
plt_compliance = ax1.plot(
    np.arange(0, it, dtype=np.int32), compliance / compliance[0], color="tab:orange"
)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.set_ylabel("Volume")
plt_volume = ax2.plot(np.arange(0, it, dtype=np.int32), volume / g[0].rhs)
ax2.axhline(y=1, linestyle="--")
ax2.set_ylim(bottom=0)

ax1.legend(plt_compliance + plt_volume, ["Compliance", "Volume"], loc=7)

plt.show()
# %% [markdown]
# The deformed final design (displacement scaled by $5\times10^3$)
# %% tags=["hide-input"]
plotter = pv.Plotter(window_size=[3840, 2160])

pv_grid.point_data[u.name] = u.x.array.reshape(-1, 3)
pv_grid.cell_data[s.name] = s.x.array

plotter.add_mesh(pv_grid.warp_by_vector(u.name, factor=5e3), color="black", line_width=1.5)

plotter.add_axes()
plotter.view_xy()
plotter.camera.elevation += 20
plotter.camera.zoom(1.7)
plotter.show()

# %% [markdown]
# Visualisation of the optimised truss structure, each truss plotted corresponding to optimisation
# result (opacity of tubes according to ratio of maximal allowed cross sectional area).
# Radii are scaled by factor of $5$, for visualisation purposes.
# %% tags=["hide-input"]
plotter = pv.Plotter(window_size=[3840, 2160])

radius = np.sqrt(s.x.array / np.pi)
radius_max = np.sqrt(s_max / np.pi)
e_to_v = mesh.topology.connectivity(1, 0)
for e_idx in range(e_to_v.num_nodes):
    a, b = e_to_v.links(e_idx)
    plotter.add_mesh(
        pv.Tube(mesh.geometry.x[a], mesh.geometry.x[b], radius=radius[e_idx] * 5, capping=True),
        opacity=radius[e_idx] / radius_max,
    )
plotter.add_axes()
plotter.view_xy()
plotter.camera.elevation += 20
plotter.camera.zoom(1.7)
plotter.show()

# %% tags=["hide-input"]
with dolfinx.io.VTXWriter(comm, "S.bp", s, "bp4") as file:
    file.write(0.0)
with dolfinx.io.VTXWriter(comm, "u.bp", u, "bp4") as file:
    file.write(0.0)
