# %% [markdown]
# # Third medium contact 
#
# This demo solves the C-shape benchmark problem with the Third Medium 
# Contact (TMC) method, using different regularization techniques. 
# In TMC, the region between potential contact surfaces is filled with a fictitious hyperelastic continuum, 
# the third medium, that is highly compliant in the pre-contact phase and stiffens sharply as it is compressed, 
# so that contact is enforced implicitly through the medium's own constitutive response 
# rather than through explicit surface detection.
#
# In particular, this demo emphasizes:
# - setting up and solving large-deformation contact problems with the TMC method
# - effect of different regularization strategies on convergence behavior and third medium deformation 
# - multi-domain formulation for the body and third medium regions requiring the use of submeshes and entity maps
# - displacement-controlled continuation with adaptive step rejection for highly nonlinear problems
# 
# ## Geometry
# 
# The C-shape configuration, first introduced by [Bluhm et al.](https://doi.org/10.1007/s00466-021-01974-x), 
# consists of a C-shaped elastic body, clamped on the left side and loaded by a vertical displacement at its 
# top-right corner node, with the third medium filling the gap between the two beams. 
#
# %% tags=["hide-input"]
import warnings
from datetime import datetime

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import fem
from dolfinx.io import VTXWriter

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import dolfiny
from dolfiny.utils import pprint

warnings.filterwarnings("error")

# Basic settings
name = "tmc_cbox"
comm = MPI.COMM_WORLD

# Dimensions [m]
L = 1.0  # length of the C-shape
H = 0.5  # height of the C-shape
T = 0.1  # thickness of the two beams

Nx, Ny = 40, 20  # cells along x and y within the C-shape
dL = L / Nx  # characteristic cell size

# Element type
quad = dolfinx.mesh.CellType.quadrilateral

# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0, 0], [L+dL, H]],
    [Nx+1, Ny],
    cell_type=quad,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
)

tdim = mesh.topology.dim 
fdim = tdim - 1 

# Define subdomains to mark third medium and body cells
tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol)

def thirdmedium_layer(x):
    return (x[0] >= L - tol)

def left(x):
    return np.isclose(x[0], 0.0)

# Mark cells
BODY_marker = 1
TM_marker = 2

num_cells_local = (
    mesh.topology.index_map(tdim).size_local
    + mesh.topology.index_map(tdim).num_ghosts
)
markers = np.full(num_cells_local, BODY_marker, dtype=np.int32)
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium)] = TM_marker
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium_layer)] = TM_marker
ct = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells_local), markers)
ct.name = "cell_tags"

# Mark facets
LEFT_marker = 2

mesh.topology.create_connectivity(fdim, tdim) # facets-to-cells connectivity
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)

facet_markers = np.zeros(num_facets_local, dtype=np.int32)
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, left)] = (
    LEFT_marker
)
f_to_c = mesh.topology.connectivity(fdim, tdim)

ft_indices = np.flatnonzero(facet_markers)

ft = dolfinx.mesh.meshtags(
    mesh, fdim, ft_indices, facet_markers[ft_indices]
)
ft.name = "facet_tags"

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

# Create third medium submesh 
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

# DG0 field carrying the region tags, used to identify body and third medium cells
tm_func = fem.Function(fem.functionspace(mesh, ("DG", 0)), name="cell_markers")
tm_func.x.array[:] = ct.values

## Material Properties
E = 1.0  # Young's modulus of the body [MPa]
nu = 0.4  # Poisson ratio of the body
K = E / (3 * (1 - 2 * nu))  # bulk modulus [MPa] 
mu = E / (2 * (1 + nu))  # shear modulus [MPa]   

K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)

def plot_contact_evolution_pyvista(name, u, tm_func, num_cells_owned, tdim, comm):
    """PyVista plotter for the deformed configurations"""
    if comm.rank > 0:
        return None, None

    grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))

    # A higher-order cell is drawn by tessellating it into flat sub-cells; the level
    # controls how finely. Linear cells need no subdivision.
    subdivision_levels = 0 if grid.get_cell(0).is_linear else 3

    # tm_func (BODY_marker=1 / TM_marker=2) is DG0: one value per cell, so it maps
    # directly to cell_data. The region assignment is fixed for the
    # whole run, so this is set once here.
    grid.cell_data["cell_marker"] = tm_func.x.array[:num_cells_owned]

    # VTK points are always stored as 3D, even for a 2D mesh, so the displacement is
    # padded with a zero out-of-plane component before being handed over.
    u_3d = np.zeros((grid.n_points, 3))
    grid.point_data["u"] = u_3d

    def warped_surface():
        """Warp the reference grid by u and tessellate it for rendering.

        Returns the coloured surface and, separately, the element outlines. The
        outlines come from separate_cells(): tessellation diagonals are interior to a
        cell and so are shared by two sub-cells, but after the cells are split apart
        every cell's own boundary becomes a feature edge. This draws the true (curved)
        element edges rather than the diagonals that show_edges would expose.
        """
        warped = grid.warp_by_vector("u", factor=1.0)
        surface = warped.extract_surface(
            nonlinear_subdivision=subdivision_levels, algorithm="dataset_surface"
        )
        edges = (
            warped.separate_cells()
            .extract_surface(nonlinear_subdivision=subdivision_levels, algorithm="dataset_surface")
            .extract_feature_edges()
        )
        return surface, edges

    surface, edges = warped_surface()

    plotter = pv.Plotter(
        off_screen=False, window_size=(res := 2048, int(res * 0.7)), theme=dolfiny.pyvista.theme
    )
    plotter.open_gif(f"{name}_disp.gif", fps=5)
    plotter.add_mesh(
        surface,
        scalars="cell_marker",
        n_colors=2,
        clim=(1, 2),
        scalar_bar_args={"n_labels": 2},
    )
    plotter.add_mesh(
        edges,
        style="wireframe",
        color="black",
        line_width=dolfiny.pyvista.pixels // 1000,
    )
    # Counter showing the applied vertical displacement of the current frame.
    load_text = plotter.add_text("")
    load_text = plotter.add_text("", position=(0.5, 0.85), viewport=True)
    load_text.prop.justification_horizontal = "center"
    plotter.show_axes()
    # Set explicit camera_position
    xmin, xmax, ymin, ymax, _, _ = grid.bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    plotter.camera_position = [(cx, cy, diag), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.zoom(0.9)

    def plot_step(_u, _u_y):
        u_3d[:, :tdim] = _u.x.array.reshape((-1, tdim))
        grid.point_data["u"] = u_3d
        # Warping produces new datasets, so copy them into the ones the actors already
        # hold instead of re-adding the actors (which would reset the scalar bar).
        new_surface, new_edges = warped_surface()
        surface.copy_from(new_surface)
        edges.copy_from(new_edges)
        load_text.input = f"u_y = {_u_y:.3f}"
        plotter.render()
        plotter.write_frame()

    return plotter, plot_step

def run_adaptive_loading(
    name,
    problem,
    state_functions,
    displacement,
    applied_y,
    output_file,
    plotter,
    plot_step,
    reaction_form,
    dofs_point_y,
    target_displacement,
    initial_increment,
    min_increment,
    adaptive_load,
    max_failure=2,
    tolerance=1e-6,
):
    """Run the load stepping loop for one regularization case."""
    state_snapshots = [state.x.array.copy() for state in state_functions]
    force_history = []
    load_history = []

    dl = initial_increment
    load = dl
    last_load = load
    n_failures = 0
    load_step = 1
    num_iterations = 0

    pprint("------------------------------------")
    pprint(f"Simulation Start ({name})")
    pprint("------------------------------------")
    start_time = datetime.now()

    while load <= (target_displacement + tolerance):
        applied_y.value = -load

        pprint(f"\nLoad step {load_step}, u_y: {applied_y.value:.3f}", flush=True)

        problem.solve(u_init=state_functions)

        reason = problem.status(verbose=True)
        num_iterations += problem.snes.getIterationNumber()

        if reason < 0:
            if adaptive_load and dl / 2 >= min_increment:
                n_failures += 1
                dl = dl / 2
                load = last_load + dl
                for state, snapshot in zip(state_functions, state_snapshots):
                    state.x.array[:] = snapshot
                    state.x.scatter_forward()
                pprint(f"  step rejected ({reason}); retrying with dl = {dl:.5f}")
            else:
                pprint("Solver failed to converge, aborting.")
                break
        else:
            last_load = load
            output_file.write(load)
            if plot_step is not None:
                plot_step(displacement, applied_y.value)
            load_step += 1
            n_failures = 0

            force_vector = fem.assemble_vector(fem.form(reaction_form))
            force_history.append(abs(force_vector.array[dofs_point_y]))
            load_history.append(abs(applied_y.value))
            print(
                f"lambda = {applied_y.value:.3f}, reaction force = {force_vector.array[dofs_point_y][0]:.6f}"
            )

            load += dl
            state_snapshots = [state.x.array.copy() for state in state_functions]

        if adaptive_load and n_failures > max_failure:
            pprint("Too many failures, aborting.")
            break

    output_file.close()
    if plotter is not None:
        plotter.close()
        plotter.deep_clean()

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    pprint("-----------------------------------------")
    pprint(f"End computation ({name})")
    pprint(f"Elapsed time: {elapsed_time}")
    pprint(f"Total number of iterations: {num_iterations}\n")

    return force_history, load_history, num_iterations, elapsed_time

# %% [markdown]
# ## TMC formulation

# equations here..


# %% [markdown]
# ## HuHu-LuLu regularization

# equations here..


# %% tags=["hide-input", "hide-output"]

def kinematics(u):
    """Deformation gradient, its determinant and the plane-strain first invariant."""
    F = ufl.Identity(tdim) + ufl.grad(u)
    J = ufl.det(F)
    I1 = ufl.tr(F.T * F) + 1.0  # +1: out-of-plane component, F_33 = 1
    return F, J, I1

def psi_body(J, I1):
    return K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2/3) * I1 - 3)

def psi_third(J, I1):
    return mu_body / 2 * (J ** (-2 / 3) * I1 - 3)  # isochoric part only, no volumetric term

# Define function spaces and functions 
element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))

u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

# Define state 
m = [u]
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V))
(δu,) = δm

# BCs
left_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(LEFT_marker)
)
bc_left = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V
)
mesh.topology.create_connectivity(0, tdim) # nodes-to-cells connectivity
node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]

gamma = fem.Constant(mesh, 1.0e-6)  # relative contact stiffness of the third medium

F, J, I1 = kinematics(u)

# Integration measures
QRULE_BODY, QDEG_BODY = "default", 4
QRULE_TM, QDEG_TM = "GLL", 3
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})

# Define the potential energy contributions
# Elastic energy of the body
Pi_body = (
    psi_body(J, I1) * dxVol
)

# Third medium elastic energy 
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
Ell = dolfinx.fem.Constant(mesh, np.max(L_i))  # 1.0
alpha = fem.Constant(mesh, 1.0e-06)
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K)

Hu = ufl.grad(ufl.grad(u)) # Hessian of displacement
Lu = ufl.div(ufl.grad(u)) # Laplacian of displacement

HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(ufl.Identity(tdim))

Pi_Hu = k_r / 2 * (HuHu) * dxThird
Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird

Pi_r = Pi_HuLu

# Define the residual and forms for the nonlinear problem
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

# Instantiate the nonlinear problem and solver
opts = PETSc.Options(name)
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "bt"
# opts["snes_linesearch_order"] = 1
opts["snes_atol"] = 1.0e-08
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 50
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Setup output file for storing results
name_HuLu = f"{name}_HuLu"
ofile = VTXWriter(comm, f"{name_HuLu}.bp", [u, tm_func])
ofile.write(0.0) # write initial state

plotter, plot_step = plot_contact_evolution_pyvista(name_HuLu, u, tm_func, num_cells_owned, tdim, comm)
if plot_step is not None:
    plot_step(u, 0.0)  # undeformed configuration as the first frame

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

adaptive_load = True
dl = 0.05  # initial load increment
dl_min = dl / 16
v_bar = -0.7  # final applied vertical displacement

disp_residual = ufl.derivative(Pi_body, u)

force_array_HL, loading_array_HL, num_iterations, elapsedTime = run_adaptive_loading(
    name="HuLu",
    problem=problem,
    state_functions=m,
    displacement=u,
    applied_y=applied_y,
    output_file=ofile,
    plotter=plotter,
    plot_step=plot_step,
    reaction_form=disp_residual,
    dofs_point_y=dofs_point_y,
    target_displacement=abs(v_bar),
    initial_increment=dl,
    min_increment=dl_min,
    adaptive_load=adaptive_load,
)


# %% [markdown]
# ```{figure} tmc_cbox_HuLu_disp.gif
# :alt: Successive deformed configurations of the C-box with HuHu-LuLu regularization.
# :align: center
# :label: fig-tmc-hulu
#
# Contact evolution for HuHu-LuLu regularization.
# ```


# %% [markdown]
# ## Wriggers regularization

# %% tags=["hide-input", "hide-output"]
element_deg = 1
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))
P = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg))
Q = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg))
W = ufl.MixedFunctionSpace(V, P, Q)

# Functions
u = fem.Function(V, name="displacement")
p1 = fem.Function(P)
q = fem.Function(Q)

# State and variation
m = [u, p1, q]
δm = ufl.TestFunctions(W)

# BCs
left_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(LEFT_marker)
)
bc_left = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V
)
mesh.topology.create_connectivity(0, tdim) # nodes-to-cells connectivity
node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]

# Integration measures for first order elements
QRULE_BODY, QDEG_BODY = "default", 2  # 2x2 Gauss-Legendre (standard)
QRULE_TM, QDEG_TM = "GLL", 1  # Gauss-Lobatto degree 1 = the four cell vertices
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})

F, J, I1 = kinematics(u)

# Define the potential energy contributions
# Elastic energy of the body
Pi_body = (
    psi_body(J, I1) * dxVol
)

# Third medium elastic energy 
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
beta_1 = dolfinx.fem.Constant(mesh, 1.0e4)  
beta_2 = dolfinx.fem.Constant(mesh, 10.) 
alpha_r = dolfinx.fem.Constant(mesh, 100.)

trF = ufl.tr(F)
skF = F[0,1] - F[1,0]

skew_term = (skF / trF) - 1/Ell * p1

Pi_grad = (
    beta_1 * skew_term**2 + alpha_r * ufl.inner(ufl.grad(p1), ufl.grad(p1))
    ) * dxThird

Pi_J = (
    beta_2 * (J - q)**2 + alpha_r * ufl.inner(ufl.grad(q), ufl.grad(q))
    ) * dxThird

Pi_r = gamma/2 * (Pi_grad + Pi_J) 

# Nonlinear problem and solver using new regularization term
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

# output file for storing results
name_W = f"{name}_Wriggers"
ofile = VTXWriter(comm, f"{name_W}.bp", [u, tm_func])
ofile.write(0.0) # write initial state

plotter, plot_step = plot_contact_evolution_pyvista(name_W, u, tm_func, num_cells_owned, tdim, comm)
if plot_step is not None:
    plot_step(u, 0.0)  # undeformed configuration as the first frame

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

adaptive_load = True
v_bar = -1.0  # final applied vertical displacement

disp_residual = ufl.derivative(Pi_body, u)

force_array_W, loading_array_W, num_iterations, elapsedTime = run_adaptive_loading(
    name="Wriggers",
    problem=problem,
    state_functions=m,
    displacement=u,
    applied_y=applied_y,
    output_file=ofile,
    plotter=plotter,
    plot_step=plot_step,
    reaction_form=disp_residual,
    dofs_point_y=dofs_point_y,
    target_displacement=abs(v_bar),
    initial_increment=dl,
    min_increment=dl_min,
    adaptive_load=adaptive_load,
)

# %% [markdown]
# ```{figure} tmc_cbox_Wriggers_disp.gif
# :alt: Successive deformed configurations of the C-box with first-order regularization.
# :align: center
# :label: fig-tmc-wriggers
#
# Contact evolution for Wriggers regularization.
# ```

# %% [markdown]
# ## Deformation-gradient-based regularization

# %% tags=["hide-input", "hide-output"]
element_deg_u = 1
element_deg_theta = 1
V = fem.functionspace(mesh, ("Lagrange", element_deg_u, (tdim,)))
V_theta = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg_theta, (tdim, tdim)))
W = ufl.MixedFunctionSpace(V, V_theta)

# Functions
u = fem.Function(V, name="displacement")
theta = fem.Function(V_theta, name="theta")

# State and variations
m = [u, theta]
δm = ufl.TestFunctions(W)

# BCs
left_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(LEFT_marker)
)
bc_left = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V
)
mesh.topology.create_connectivity(0, tdim) # nodes-to-cells connectivity
node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]


F, J, I1 = kinematics(u)

# Define the potential energy contributions
# Elastic energy of the body
Pi_body = (
    psi_body(J, I1) * dxVol
)

# Third medium elastic energy 
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
p_theta = dolfinx.fem.Constant(mesh, 5.0e-2) 
alpha_r = dolfinx.fem.Constant(mesh, 1.0e-2)

penalty_term = theta - F
Pi_penalty = (
    p_theta / 2 * ufl.inner(penalty_term, penalty_term) 
    ) * dxThird

Pi_reg = (
    gamma * alpha_r / 2 * ufl.inner(ufl.grad(theta), ufl.grad(theta))
    ) * dxThird

Pi_r =  (Pi_penalty + Pi_reg)  

# Nonlinear problem and solver using new regularization term
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

# output file for storing results
name_V = f"{name}_Vorwerk"
ofile = VTXWriter(comm, f"{name_V}.bp", [u, tm_func])
ofile.write(0.0) # write initial state


problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

adaptive_load = True
v_bar = -1.0  # final applied vertical displacement

disp_residual = ufl.derivative(Pi_body, u)

force_array_V, loading_array_V, num_iterations, elapsedTime = run_adaptive_loading(
    name="Vorwerk",
    problem=problem,
    state_functions=m,
    displacement=u,
    applied_y=applied_y,
    output_file=ofile,
    plotter=None,
    plot_step=None,
    reaction_form=disp_residual,
    dofs_point_y=dofs_point_y,
    target_displacement=abs(v_bar),
    initial_increment=dl,
    min_increment=dl_min,
    adaptive_load=adaptive_load,
)


# %% [markdown]
# ## Comparison

# %% tags=["hide-input"]
if comm.rank == 0:
    plt.figure(dpi=300)
    # plt.title("C-shape box: reaction at the loaded corner", fontsize=12)
    plt.xlabel(r"Applied displacement $|u_y|$", fontsize=12)
    plt.ylabel(r"Vertical reaction force $|R_y|$", fontsize=12)
    plt.grid(linewidth=0.25)
    for loads, forces, formulation in (
        (loading_array_HL, force_array_HL, "HuHu-LuLu"),
        (loading_array_W, force_array_W, "Wriggers first-order"),
        (loading_array_V, force_array_V, "Deformation-gradient-based"),
    ):
        plt.plot(
            loads, forces, linestyle="-", linewidth=1.0, marker=".", markersize=4.0,
            label=formulation,
        )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{name}_force_displacement.png", dpi=300)
    plt.close()

# %% [markdown]
# ```{figure} tmc_cbox_force_displacement.png
# :alt: Reaction force against applied displacement for both regularizations.
# :align: center
# :label: fig-tmc-force-displacement
#
# Reaction-displacement curves for different regularizations.
# ```


