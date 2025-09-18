# %% [markdown]
# # Truss sizing optimisation
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

dim = np.array([100, 20, 10], dtype=np.float64)
elem_size = 5
mesh = create_truss_x_braced_mesh(
    dolfinx.mesh.create_box(
        comm,
        [np.zeros_like(dim), dim],
        (dim / elem_size).astype(np.int32),  # type: ignore
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
meshtags = dolfinx.mesh.meshtags(mesh, 0, vertices_load, 1)

# %% tags=["hide-input"]
pv.set_jupyter_backend("static")
pv_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
plotter = pv.Plotter(window_size=[3840, 2160])
plotter.add_mesh(pv_grid, color="black", line_width=1.5)

for v in vertices_load:
    v_coords = mesh.geometry.x[v]
    arrow = pv.Arrow(start=v_coords + np.array([0, 4, 0]), direction=[0, -1, 0], scale=4)
    plotter.add_mesh(arrow, color="green", opacity=0.5)

for v in vertices_fixed:
    v_coords = mesh.geometry.x[v]
    sphere = pv.Sphere(radius=0.5, center=v_coords)
    plotter.add_mesh(sphere, color="red")

plotter.view_xy()
plotter.add_axes()
plotter.camera.elevation += 20
plotter.camera.zoom(1.7)
# plotter.show()

# %% [markdown]
# ## Truss Model and Forward Problem
#
# We rely on a linear elastic truss model, same as the one presented in {cite:p}`bleyer2024comet`.
# We choose a *Young's modulus* of $E = 200\, \text{GPa}$ and apply a total load of
# $300\, \text{kN}$, equally distributed across the loaded nodes.
#
# ```{note}
#   The point loads can be incorporated into the continuous formulation with vertex integrals,
#   denoted $\text{d}P$.
#   This allows to express the inherently discrete nature of the load in a variational form.
#
#   A *vertex integral* over a collection of vertices of the underlying mesh
#   $\Omega_v = \{ v_0, \dots, v_n \}$ as
#   $$
#       \int_{\Omega_v} f(x) \, \text{d} P
#       = \sum_{i=1}^n f(v_i).
#   $$
# ```
# %% tags=["hide-output"]
tangent = ufl.Jacobian(mesh)[:, 0]
tangent /= ufl.sqrt(ufl.inner(tangent, tangent))

E = 200.0 * 1e9  # [E] = Pa = N/m^2
ε = ufl.dot(ufl.dot(ufl.grad(u), tangent), tangent)  # strain
N = E * s * ε  # normal_force

dP = ufl.Measure("dP", domain=mesh, subdomain_data=meshtags)
total_load = 300 * 1e3
F = ufl.as_vector([0, -total_load / comm.allreduce(vertices_load.size), 0])
compliance = ufl.inner(F, u) * dP(1)

E = 1 / 2 * ufl.inner(N, ε) * ufl.dx - compliance

a = ufl.derivative(ufl.derivative(E, u), u)
L = ufl.derivative(compliance, u)

state_solver = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    bcs=bcs,
    u=u,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": "True",
        "mat_mumps_cntl_1": 0.0,
    },
    petsc_options_prefix="state_solver",
)

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


# %% tags=["hide-output"]
compliance_form = dolfinx.fem.form(compliance)


@dolfiny.taoproblem.sync_functions([s])
def C(tao, x) -> float:
    state_solver.solve()
    return comm.allreduce(dolfinx.fem.assemble_scalar(compliance_form))  # type: ignore


# p = -u
p = dolfinx.fem.Function(V_u)
gx = dolfinx.fem.form(ufl.derivative(ufl.derivative(E, u, p), s))


@dolfiny.taoproblem.sync_functions([s])
def JC(tao, x, J):
    state_solver.solve()

    J.zeroEntries()
    p.x.array[:] = -u.x.array

    dolfinx.fem.petsc.assemble_vector(J, gx)
    J.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


s_max = np.float64(1e-2)
s_min = np.float64(1e-3 * s_max)
max_vol = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, s_max) * ufl.dx))
)
volume_fraction = 1 / 20
h = [s / max_vol / volume_fraction * ufl.dx <= 1]
s.x.array[:] = 1 / 100 * s_max

opts = PETSc.Options("truss")  # type: ignore
opts["tao_type"] = "python"
opts["tao_python_type"] = "dolfiny.mma.MMA"
opts["tao_monitor"] = ""
opts["tao_max_it"] = (max_it := 50)
opts["tao_mma_move_limit"] = 0.05
opts["tao_mma_subsolver_tao_max_it"] = 30

problem = dolfiny.taoproblem.TAOProblem(
    C, [s], bcs=bcs, J=(JC, s.x.petsc_vec.copy()), h=h, lb=s_min, ub=s_max, prefix="truss"
)


def monitor(tao, comp, volume):
    it = tao.getIterationNumber()
    comp[it] = tao.getPythonContext().getObjectiveValue()
    volume[it] = dolfinx.fem.assemble_scalar(dolfinx.fem.form(h[0].lhs))


comp = np.zeros(max_it, np.float64)
volume = np.zeros(max_it, np.float64)

if comm.size == 1:
    problem.tao.setMonitor(monitor, (comp, volume))
problem.solve()

state_solver.solve()

# %% tags=["hide-input"]
# verify result
# if problem.tao.getConvergedReason() <= 0:
#     raise RuntimeError("Optimisation did not converge.")

if (
    np.abs(comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(h[0].lhs))) - h[0].rhs)
    >= 1e-2
):
    raise RuntimeError("Volume constraint violated.")

if np.any(s.x.array < s_min):
    raise RuntimeError("Lower bound violated.")

if np.any(s.x.array > s_max):
    raise RuntimeError("Upper bound violated.")

with dolfinx.io.VTXWriter(comm, "S.bp", s, "bp4") as file:
    file.write(0.0)
with dolfinx.io.VTXWriter(comm, "u.bp", u, "bp4") as file:
    file.write(0.0)

if comm.size > 1:
    exit()

# %% [markdown]
# Convergence of the optimisation, that is the compliance and the volume vs. outer MMA iteration
# are shown below. Compliance is measured relative to the initial compliance $C_0$, which is
# compliance for the initial guess of $s_\text{init} = 10^{-2} \, \text{m}^2$. Volume is
# shown relative to the upper bound $V_\text{max}$.
# %% tags=["hide-input"]
matplotlib_inline.backend_inline.set_matplotlib_formats("png")
it = problem.tao.getIterationNumber()
comp = comp[:it]  # type: ignore
volume = volume[:it]  # type: ignore

fig, ax1 = plt.subplots(dpi=400)
ax1.set_xlim(0, it - 1)
ax1.set_xlabel("Outer MMA iteration")
ax1.set_ylabel("Rel. compliance $C / C_0$")
plt_compliance = ax1.plot(
    np.arange(0, it, dtype=int),
    comp / comp[0],
    color="tab:orange",
    marker="x",
)
ax1.set_yscale("log")
ax1.grid(True, which="both")

ax2 = ax1.twinx()
ax2.set_ylabel(r"$|1 - V / V_\text{max}|$")
plt_volume = ax2.plot(np.arange(0, it, dtype=int), np.abs(1 - volume / h[0].rhs), marker=".")
ax2.axhline(y=1, linestyle="--")
ax2.set_yscale("log")
ax1.legend(plt_compliance + plt_volume, ["compliance", "volume"], loc=7)

plt.show()
# %% [markdown]
# The deformed final design (displacement scaled by $5\times10^3$)
# %% tags=["hide-input"]
pixels = dolfiny.pyvista.pixels
plotter = pv.Plotter(window_size=[pixels, pixels // 2])

pv_grid.point_data[u.name] = u.x.array.reshape(-1, 3)
pv_grid.cell_data[s.name] = s.x.array

plotter.add_mesh(pv_grid.warp_by_vector(u.name, factor=5e3), color="black", line_width=1.5)

plotter.add_axes()
plotter.view_xy()
plotter.camera.elevation += 20
plotter.camera.zoom(2.0)
plotter.show()

# %% [markdown]
# Visualisation of the optimised truss structure, each truss plotted corresponding to optimisation
# result (opacity of tubes according to ratio of maximal allowed cross sectional area).
# Radii are scaled by factor of $5$, for visualisation purposes.
# %% tags=["hide-input"]
plotter = pv.Plotter(window_size=[pixels, pixels // 2])

radius = np.sqrt(s.x.array / np.pi)
radius_max = np.sqrt(s_max / np.pi)
e_to_v = mesh.topology.connectivity(1, 0)
for e_idx in range(e_to_v.num_nodes):
    a, b = e_to_v.links(e_idx)
    plotter.add_mesh(
        pv.Tube(
            pointa=mesh.geometry.x[a],
            pointb=mesh.geometry.x[b],
            radius=radius[e_idx] * 5,
            capping=True,
        ),
        opacity=radius[e_idx] / radius_max,
    )
plotter.add_axes()
plotter.view_xy()
plotter.camera.elevation += 20
plotter.camera.zoom(2.0)
plotter.show()
