# %% [markdown]
# # Transient elastodynamics for finite deformation
#
# This demo solves the equations of elastodynamics for a cantilever beam subject to gravity loading.
# The model includes both material damping (viscous effects) and geometric nonlinearity through
# finite deformation kinematics.
# Time integration is performed using a generalized-$\alpha$ scheme (Newmark-like) with automatic
# spectral radius control.
#
# In particular, this demo emphasizes:
# - Setting up and solving time-dependent nonlinear solid mechanics problems
# - Finite deformation kinematics and Saint Venant-Kirchhoff material models
# - Automatic time integration with `dolfiny.odeint`
#
# For a demonstration (of very similar nature) to linear elastodynamics we refer to
# the [*Transient elastodynamics with Newmark time-integration*](
# https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html)
# demo of `comet-fenicsx` {cite:p}`bleyer2024comet`.
#
# ## Geometry
#
# In this demo we model a thin cantilever beam clamped at one end, initially at rest in the
# undeformed configuration, and subjected to gravity loading.
# The mesh is generated with `gmsh`.
#
# %% tags=["hide-input"]
import warnings

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import gmsh
import matplotlib.pyplot as plt
import mesh_block3d_gmshapi as mg
import numpy as np
import pyvista

import dolfiny

warnings.filterwarnings("error")

# Basic settings
name = "solid_disp_tda"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters (a thin cantilever beam)
dimensions = (2.0, 0.01, 0.1)  # Length, width, height in m
elements = 20, 2, 2  # Mesh divisions along each dimension

# Create the regular mesh of a block with given dimensions
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 3)
gmsh_model, tdim = mg.mesh_block3d_gmshapi(
    name, *dimensions, *elements, px=1.0, py=1.0, pz=1.0, do_quads=False
)

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0)
mesh = mesh_data.mesh

surface_left = mesh_data.physical_groups["surface_left"].tag
surface_right = mesh_data.physical_groups["surface_right"].tag

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_mesh(
        grid, show_edges=True, color="white", line_width=dolfiny.pyvista.pixels // 1000
    )
    plotter.show_axes()
    plotter.camera_position = [(3.5, 1.2, -2.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.screenshot("solid_disp_tda_mesh.png")
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ```{figure} solid_disp_tda_mesh.png
# :alt: Cantilever beam mesh used for the transient elastodynamics simulation.
# :align: center
# :label: fig-solid-disp-mesh
#
# Cantilever beam mesh used for the transient elastodynamics simulation.
# ```
#
# ## Elastodynamics
#
# Balance of linear momentum in the Lagrangian description reads
#
# $$
#   \rho \ddot{\mathbf{u}} + \eta \dot{\mathbf{u}} - \nabla \cdot (\mathbf{F} \mathbf{S})
#   = \rho \mathbf{b},
# $$
#
# where $\mathbf{u}(\mathbf{x}, t)$ is the displacement, $\dot{\mathbf{u}}(\mathbf{x}, t)$ velocity,
# $\ddot{\mathbf{u}}(\mathbf{x}, t)$ acceleration, $\rho(\mathbf{x})$ is mass density,
# $\eta(\mathbf{x})$ is a damping coefficient, $\mathbf{S}(\mathbf{x}, t)$ the 2nd Piola-Kirchhoff
# (PK2) stress, $\mathbf{b}(\mathbf{x}, t)$ body force and
# $\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$ the deformation gradient between undeformed
# $\mathbf{x}$ and deformed $\mathbf{x} + \mathbf{u}(\mathbf{x}, t)$ configuration.
#
# We employ the (linear) St. Venant-Kirchhoff constitutive model, parametrised in shear modulus
# $\mu$ and Lamé parameter $\lambda$,
# $$
#   \mathbf{S}(\mathbf{E}) = 2 \mu \mathbf{E} + \lambda \text{tr}(\mathbf{E}) \mathbf{I},
# $$
#
# to the Green-Lagrange strain $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T \mathbf{F} - \mathbf{I})$.
#
# The deformation gradient is $\mathbf{F} = \mathbf{I} + \nabla_X \mathbf{u}$, giving the
# Green-Lagrange strain tensor:
# $$\mathbf{E} = \frac{1}{2}(\mathbf{F}^T \mathbf{F} - \mathbf{I})$$
#
# We arrive at the weak form (residual)
#
# $$
#   R(\mathbf{u}, \dot{\mathbf{u}}, \ddot{\mathbf{u}}; \delta \mathbf{u}) =
#   \int_\Omega
#   \rho \, \delta\mathbf{u} \cdot \ddot{\mathbf{u}}
#   + \eta \, \delta\mathbf{u} \cdot \dot{\mathbf{u}}
#   + \delta\mathbf{E}[\delta \mathbf{u}] : \mathbf{S}
#   - \rho \, \delta\mathbf{u} \cdot \mathbf{b}
#   \,\text{d}x.
# $$
#

# %%
ρ = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e4))  # [1e-9 * 1e+4 kg/m^3]
η = dolfinx.fem.Constant(mesh, scalar(1e-9 * 0e4))  # [1e-9 * 0e+4 kg/m^3/s]
µ = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e11))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
λ = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e10))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]

b = dolfinx.fem.Constant(mesh, [0.0, -10, 0.0])  # [m/s^2]

# Define function space and arguments
V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))

u = dolfinx.fem.Function(V, name="u")
ut = dolfinx.fem.Function(V, name="velocity")
utt = dolfinx.fem.Function(V, name="acceleration")

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V))
(δu,) = δm

# Define state and rate as ordered lists for time integration
m, mt, mtt = [u], [ut], [utt]

# Kinematics
I = ufl.Identity(3)  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient

# E = E(u) total strain
E = 1 / 2 * (F.T * F - I)

# S = S(E) stress
S = 2 * µ * E + λ * ufl.tr(E) * I

δE = ufl.derivative(E, m, δm)

residual = (
    ufl.inner(δu, ρ * utt) * dx
    + ufl.inner(δu, η * ut) * dx
    + ufl.inner(δE, S) * dx
    - ufl.inner(δu, ρ * b) * dx
)
# %% [markdown]
# ## Time integration/stepping
#
# The residual at this point has been discretised in space, but not in time.
#  For this purpose, general ODE integrators are provided in `dolfiny.odeint`.
#
# In general, a time-integrator/stepper is a mapping
# $$
#   I: (m_n, \dot{m}_n, \ddot{m}_n) \mapsto (m_{n+1}, \dot{m}_{n+1}, \ddot{m}_{n+1}).
# $$
#
# It represents the *stepping* from time step $n$ to $n+1$.
# Functionally this process in controlled in three steps:
# - `stage()` — construct predictor and auxiliary fields from $(m_n, \dot{m}_n, \ddot{m}_n)$,
# - `discretise_in_time(R)` — substitute the discrete derivatives into $R$,
# - `update()` — after solving the residual equation for $m_{n+1}$, compute $\dot{m}_{n+1}$ and
#   $\ddot{m}_{n+1}$.
#
# Currently `dolfiny`'s second order integrator `ODEInt2` supports the generalised alpha method(s)
# {cite:p}`Chung1993`.
#
# %%
time = dolfinx.fem.Constant(mesh, scalar(0.0))  # [s]
dt = dolfinx.fem.Constant(mesh, scalar(1e-2))  # [s]
nT = 200

odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=m, xt=mt, xtt=mtt, rho=0.95)

residual = odeint.discretise_in_time(residual)
forms = ufl.extract_blocks(residual)

# %% tags=["hide-input"]
# Set up output file for visualization
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
ofile.write_mesh_data(mesh_data)

# Function for output at lower resolution
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (3,))), name="u")

# Write initial state
dolfiny.interpolation.interpolate(u, uo)
ofile.write_function(uo, time.value)

# Configure PETSc solver options
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-09
opts["snes_max_it"] = 12

opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_cntl_1"] = 0.0  # Disable relative pivoting threshold

# Clamped side
bcs = [
    dolfinx.fem.dirichletbc(
        dolfinx.fem.Function(V),
        dolfiny.mesh.locate_dofs_topological(V, mesh_data.facet_tags, surface_left),
    )
]

if comm.size == 1:
    point_eval = np.array([[dimensions[0], dimensions[1] / 2, dimensions[2] / 2]])

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point_eval)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point_eval)

    assert colliding_cells.num_nodes == 1
    assert colliding_cells.links(0).size >= 1
    cell = colliding_cells.links(0)[0]

    time_history = np.zeros(nT + 1, dtype=scalar)
    displacement_history = np.zeros((nT + 1, 3), dtype=scalar)

    displacement_history[0] = u.eval(point_eval, np.array([cell]))

    plotter_disp = pyvista.Plotter(
        off_screen=False, window_size=(res := 2048, int(res * 0.7)), theme=dolfiny.pyvista.theme
    )
    plotter_disp.open_gif(f"{name}_disp.gif", fps=30)
    orig_points = grid.points.copy()
    grid.point_data["u"] = np.zeros(grid.points.shape[0])
    actor = plotter_disp.add_mesh(
        grid, scalars="u", n_colors=10, scalar_bar_args={"position_y": 0.85}, clim=(0, 0.25)
    )
    plotter_disp.show_axes()
    plotter_disp.camera_position = [(3.5, 1.2, -2.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter_disp.camera.zoom(1.5)

    def plot_step(_u):
        vals = _u.x.array.reshape((-1, 3))
        disp_mag = np.linalg.norm(vals, axis=1)

        grid.point_data["u"] = disp_mag
        grid.points = orig_points + vals

        plotter_disp.write_frame()

    plot_step(uo)

# %% tags=["hide-output"]
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix=name, bcs=bcs)

# Process time steps
for time_step in range(1, nT + 1):
    dolfiny.utils.pprint(
        f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}\n"
    )

    # Stage next time step
    odeint.stage()

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Update solution states for time integration
    odeint.update()

    # Write output
    dolfiny.interpolation.interpolate(u, uo)
    ofile.write_function(uo, time.value)

    if comm.size == 1:
        time_history[time_step] = time.value
        displacement_history[time_step] = u.eval(point_eval, np.array([cell]))

        plot_step(uo)

# %% tags=["hide-input"]
ofile.close()

if comm.size == 1:
    plotter_disp.close()
    plotter_disp.deep_clean()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(time_history, displacement_history[:, 0], label="$u_x$", marker="o", ms=4, lw=1)
    ax.plot(time_history, displacement_history[:, 1], label="$u_y$", marker="s", ms=4, lw=1)
    ax.plot(time_history, displacement_history[:, 2], label="$u_z$", marker="^", ms=4, lw=1)
    ax.set_xlabel("Time $[s]$", fontsize=12)
    ax.set_ylabel("Displacement $[m]$", fontsize=12)
    ax.legend()
    ax.grid(linewidth=0.25)
    plt.tight_layout()
    plt.savefig(f"{name}_deflection.png", dpi=300)
    plt.close()

# %% [markdown]
# ```{figure} solid_disp_tda_disp.gif
# :alt: GIF of cantilever deflection over time.
# :align: center
# :label: fig-solid-disp-tda-disp
#
# Deflected cantilever over time.
# ```
#
# ```{figure} solid_disp_tda_deflection.png
# :alt: Midpoint deflection of free surface
# :align: center
# :label: fig-solid-disp-tda-deflection
#
# Displacement components over time of the midpoint for the non-clamped face, showing the
# oscillation of the cantilever.
# ```
