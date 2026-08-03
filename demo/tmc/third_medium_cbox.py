# %% [markdown]
# # Third medium contact

# %% tags=["hide-input"]
import dataclasses
import warnings
from collections.abc import Callable
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

name = "tmc_cbox"
comm = MPI.COMM_WORLD

# %% [markdown]
# ## Geometry 

# %% tags=["hide-input"]
# Geometry [m]
L = 1.0  # length of the C-shape
H = 0.5  # height of the C-shape
T = 0.1  # thickness of the two beams

Nx, Ny = 40, 20  # cells along x and y within the C-shape
dL = L / Nx  # characteristic cell size

mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0.0, 0.0], [L + dL, H]],
    [Nx + 1, Ny],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
)

tdim = mesh.topology.dim
fdim = tdim - 1

# Define subdomains to mark third medium and body cells
tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol)

def thirdmedium_layer(x):
    return x[0] >= L - tol

def left(x):
    return np.isclose(x[0], 0.0)

# Mark cells
BODY_marker = 1
TM_marker = 2

num_cells_local = (
    mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
)
markers = np.full(num_cells_local, BODY_marker, dtype=np.int32)
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium)] = TM_marker
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium_layer)] = TM_marker
ct = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells_local), markers)
ct.name = "cell_tags"

# Mark facets
LEFT_marker = 2

mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(0, tdim)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
)
facet_markers = np.zeros(num_facets_local, dtype=np.int32)
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, left)] = LEFT_marker
ft_indices = np.flatnonzero(facet_markers)
ft = dolfinx.mesh.meshtags(mesh, fdim, ft_indices, facet_markers[ft_indices])
ft.name = "facet_tags"

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(mesh.topology.index_map(0).size_local, op=MPI.SUM)
pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

# Create third medium submesh 
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

# DG0 field carrying the region tags, used to identify body and third medium cells
tm_func = fem.Function(fem.functionspace(mesh, ("DG", 0)), name="cell_markers")
tm_func.x.array[:] = ct.values

# %% [markdown]
# ## Constitutive model

# %% tags=["hide-input"]
E = 1.0  # Young's modulus of the body [MPa]
nu = 0.4  # Poisson ratio of the body
K = E / (3 * (1 - 2 * nu))  # bulk modulus [MPa]
mu = E / (2 * (1 + nu))  # shear modulus [MPa]

K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)
gamma = fem.Constant(mesh, 1.0e-6)  # relative contact stiffness of the third medium

# Characteristic size of the problem
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_max - x_min
Ell = fem.Constant(mesh, np.max(L_i))  # characteristic length [m]


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


# %% [markdown]
# ## Modular structure for multiple regularizations

# %% tags=["hide-input"]
@dataclasses.dataclass
class Formulation:
    """Regularization-specific parameters."""

    label: str  # for figure legends
    tag: str  # for file names and PETSc option prefixes
    u_degree: int  # polynomial degree of the displacement field
    q_body: tuple[str, int]  # quadrature (rule, degree) on the body
    q_third: tuple[str, int]  # quadrature (rule, degree) on the third medium
    aux_fields: tuple[tuple[str, tuple], ...]  # (name, element) of each auxiliary field
    regularization: Callable  # (u, aux, dx_third) -> regularization energy
    v_bar: float  # final applied vertical displacement 
    theta_degree: int | None = None  # degree used only by the Vorwerk formulation


@dataclasses.dataclass
class Wrapper:
    """A formulation, discretized and ready to be driven along a load path."""

    formulation: Formulation
    u: fem.Function
    m: list
    problem: dolfiny.snesproblem.SNESProblem
    applied_y: fem.Constant
    dofs_point_y: np.ndarray
    force_form: fem.Form


def build_problem(f: Formulation) -> Wrapper:
    dx_vol = dx(
        BODY_marker,
        metadata={"quadrature_rule": f.q_body[0], "quadrature_degree": f.q_body[1]},
    )
    dx_third = dx(
        TM_marker,
        metadata={"quadrature_rule": f.q_third[0], "quadrature_degree": f.q_third[1]},
    )

    V = fem.functionspace(mesh, ("Lagrange", f.u_degree, (tdim,)))
    aux_spaces = [fem.functionspace(third_medium_mesh, e) for _, e in f.aux_fields]

    u = fem.Function(V, name="displacement")
    aux = [fem.Function(W, name=n) for (n, _), W in zip(f.aux_fields, aux_spaces, strict=True)]
    m = [u, *aux]
    δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V, *aux_spaces))

    F, J, I1 = kinematics(u)
    Pi_body = psi_body(J, I1) * dx_vol
    Pi = Pi_body + gamma * psi_third(J, I1) * dx_third + f.regularization(u, aux, dx_third)

    # absent = [w.name for w in m if w not in Pi.coefficients()]
    # assert not absent, (
    #     f"{absent} do not enter the total energy of formulation '{f.label}': their residual "
    #     "blocks would vanish identically and the tangent matrix would be singular"
    # )

    # BCs
    bc_left = fem.dirichletbc(
        np.zeros(tdim, dtype=dolfinx.default_scalar_type),
        fem.locate_dofs_topological(V, fdim, ft.find(LEFT_marker)),
        V,
    )
    node = dolfinx.mesh.locate_entities(
        mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H)
    )
    dofs_point_y = fem.locate_dofs_topological(V.sub(1), 0, node)
    applied_y = fem.Constant(mesh, 0.0)
    bcs = [bc_left, fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))]

    forms = ufl.extract_blocks(ufl.derivative(Pi, m, δm))

    opts = PETSc.Options(f"{name}_{f.tag}")  # type: ignore[attr-defined]
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "bt"  # backtrack rather than overshoot out of {J > 0}
    opts["snes_atol"] = 1.0e-08
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 50
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    problem = dolfiny.snesproblem.SNESProblem(
        forms, m, bcs=bcs, prefix=f"{name}_{f.tag}", entity_maps=[medium_map]
    )

    # Reaction force at the top-right node where the vertical displacement is applied 
    force_form = fem.form(ufl.derivative(Pi_body, u))

    return Wrapper(f, u, m, problem, applied_y, dofs_point_y, force_form)


def deformation_plotter(gif_path, u):
    """PyVista animation of the deformed configuration, coloured by cell marker."""
    if comm.rank > 0:
        return None, None

    grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))

    # A higher-order cell is drawn by tessellating it into flat sub-cells; the level controls
    # how finely. Linear cells need no subdivision.
    subdivision_levels = 0 if grid.get_cell(0).is_linear else 3

    # tm_func is DG0: one value per cell, so it maps directly to cell_data. The region
    # assignment is fixed for the whole run, so this is set once here.
    grid.cell_data["cell_marker"] = tm_func.x.array[:num_cells_owned]

    # VTK points are always stored as 3D, even for a 2D mesh.
    u_3d = np.zeros((grid.n_points, 3))
    grid.point_data["u"] = u_3d

    def warped_surface():
        """Warp the reference grid by u and tessellate it for rendering.

        Returns the coloured surface and, separately, the element outlines. The outlines come
        from separate_cells(): tessellation diagonals are interior to a cell and so are shared
        by two sub-cells, but after the cells are split apart every cell's own boundary
        becomes a feature edge. This draws the true (curved) element edges rather than the
        diagonals that show_edges would expose.
        """
        warped = grid.warp_by_vector("u", factor=1.0)
        surface = warped.extract_surface(
            nonlinear_subdivision=subdivision_levels, algorithm="dataset_surface"
        )
        edges = (
            warped.separate_cells()
            .extract_surface(
                nonlinear_subdivision=subdivision_levels, algorithm="dataset_surface"
            )
            .extract_feature_edges()
        )
        return surface, edges

    surface, edges = warped_surface()

    plotter = pv.Plotter(
        off_screen=False, window_size=(res := 2048, int(res * 0.7)), theme=dolfiny.pyvista.theme
    )
    plotter.open_gif(gif_path, fps=5)
    plotter.add_mesh(
        surface, scalars="cell_marker", n_colors=2, clim=(1, 2), scalar_bar_args={"n_labels": 2}
    )
    plotter.add_mesh(
        edges, style="wireframe", color="black", line_width=dolfiny.pyvista.pixels // 1000
    )
    # Counter showing the applied vertical displacement of the current frame.
    load_text = plotter.add_text("", position=(0.5, 0.85), viewport=True)
    load_text.prop.justification_horizontal = "center"
    plotter.show_axes()
    xmin, xmax, ymin, ymax, _, _ = grid.bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    plotter.camera_position = [(cx, cy, diag), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.zoom(.9)

    def plot_step(_u, _u_y):
        u_3d[:, :tdim] = _u.x.array.reshape((-1, tdim))
        grid.point_data["u"] = u_3d
        # Warping produces new datasets, so copy them into the ones the actors already hold
        # instead of re-adding the actors (which would reset the scalar bar).
        new_surface, new_edges = warped_surface()
        surface.copy_from(new_surface)
        edges.copy_from(new_edges)
        load_text.input = f"u_y = {_u_y:.3f}"
        plotter.render()
        plotter.write_frame()

    return plotter, plot_step


def run_load_path(wrapper: Wrapper, dl_0=0.05, dl_min_factor=16, max_failures=3):
    """Run a load path by incrementally increasing the applied vertical displacement at the
    top-right corner of the C-shape box. 
    
    Adaptive load step size: load increment is halved when a step fails, and the step is retried.
    If the increment falls below `dl_0 / dl_min_factor` or if `max_failures` consecutive steps are rejected, the continuation is aborted.
    """
    f = wrapper.formulation
    u, aux = wrapper.u, wrapper.m[1:]

    ofile = VTXWriter(comm, f"{name}_{f.tag}.bp", [u, tm_func])
    ofile.write(0.0)
    plotter, plot_step = deformation_plotter(f"{name}_{f.tag}_disp.gif", u)
    if plot_step is not None:
        plot_step(u, 0.0)  # undeformed configuration as the first frame

    # Last converged equilibrium, to fall back on when a step is rejected.
    u_prev = u.x.array.copy()
    aux_prev = [w.x.array.copy() for w in aux]
    last_load = 0.0

    dl = dl_0  # load increment 
    dl_min = dl_0 / dl_min_factor  # smallest load increment  
    load = dl
    failures = 0  # consecutive rejected steps
    step = 1
    total_iterations = 0
    loads: list[float] = []
    forces: list[float] = []

    pprint(f"\n--- {f.label}: continuation to u_y = {f.v_bar:.3f} ---")
    start_time = datetime.now()

    while load <= abs(f.v_bar) + tol:
        wrapper.applied_y.value = -load
        pprint(f"\nLoad step {step}, u_y: {wrapper.applied_y.value:.3f}", flush=True)

        wrapper.problem.solve(u_init=wrapper.m)
        reason = wrapper.problem.status(verbose=True)
        total_iterations += wrapper.problem.snes.getIterationNumber()

        if reason < 0:
            if dl / 2 < dl_min or failures + 1 >= max_failures:
                pprint(f"Solver failed (reason {reason}) at dl = {dl:.5f}, aborting.")
                break
            failures += 1
            dl = dl / 2
            load = last_load + dl
            u.x.array[:] = u_prev
            u.x.scatter_forward()  
            for w, w_prev in zip(aux, aux_prev, strict=True):
                w.x.array[:] = w_prev
                w.x.scatter_forward()
            pprint(f"  step rejected ({reason}); retrying with dl = {dl:.5f}")
            continue

        last_load = load
        ofile.write(load)
        if plot_step is not None:
            plot_step(u, wrapper.applied_y.value)

        reaction = fem.assemble_vector(wrapper.force_form)
        local = abs(reaction.array[wrapper.dofs_point_y][0]) if wrapper.dofs_point_y.size else 0.0
        forces.append(comm.allreduce(local, op=MPI.MAX))
        loads.append(abs(wrapper.applied_y.value))
        pprint(f"u_y = {wrapper.applied_y.value:.3f}, reaction force = {forces[-1]:.6f}")

        u_prev[:] = u.x.array
        for w, w_prev in zip(aux, aux_prev, strict=True):
            w_prev[:] = w.x.array
        failures = 0
        step += 1
        load += dl

    ofile.close()
    if plotter is not None:
        plotter.close()
        plotter.deep_clean()

    pprint(
        f"\n{f.label}: reached u_y = {-loads[-1] if loads else 0.0:.3f} in {total_iterations} "
        f"Newton iterations, elapsed {datetime.now() - start_time}"
    )

    return loads, forces


# %% [markdown]
# ## HuHu-LuLu regularization

# %% tags=["hide-input", "hide-output"]
alpha = fem.Constant(mesh, 1.0e-06)  # regularization scaling
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K)  # regularization strength parameter


def regularization_hulu(u, aux, dx_third):
    Hu = ufl.grad(ufl.grad(u))  # Hessian of the displacement
    Lu = ufl.div(ufl.grad(u))   # Laplacian of the displacement
    HuHu = ufl.inner(Hu, Hu)
    LuLu = ufl.inner(Lu, Lu) / ufl.tr(ufl.Identity(tdim))
    return k_r / 2 * (HuHu - LuLu) * dx_third


HULU = Formulation(
    label="HuHu-LuLu",
    tag="HuLu",
    u_degree=2,  # second gradients require at least quadratic shape functions
    q_body=("default", 4), 
    q_third=("GLL", 3),     # Gauss-Lobatto 
    aux_fields=(),
    regularization=regularization_hulu,
    v_bar=-0.7,  # final applied vertical displacement 
)

hulu = build_problem(HULU)
loads_hulu, forces_hulu = run_load_path(hulu)

# %% [markdown]
# ```{figure} tmc_cbox_HuLu_disp.gif
# :alt: Successive deformed configurations of the C-box with HuHu-LuLu regularization.
# :align: center
# :label: fig-tmc-hulu
#
# Contact evolution for HuHu-LuLu regularization.
# ```

# %% [markdown]
# ## Wriggers first-order regularization

# %% tags=["hide-input", "hide-output"]
beta_1 = fem.Constant(mesh, 1.0e4)  # penalty tying p to the rotation ratio
beta_2 = fem.Constant(mesh, 10.0)   # penalty tying q to the volume ratio
alpha_r = fem.Constant(mesh, 100.0) # regularization parameter for the gradient of auxiliary fields


def regularization_wriggers(u, aux, dx_third):
    p, q = aux
    F, J, _ = kinematics(u)
    skew_F = F[0, 1] - F[1, 0]  # numerator of tan(phi): the skew part of F
    trace_F = ufl.tr(F)         # denominator of tan(phi): the stretch part (trace) of F

    Pi_rot = (
        beta_1 * (skew_F / trace_F - p / Ell) ** 2
        + alpha_r * ufl.inner(ufl.grad(p), ufl.grad(p))
    ) * dx_third
    Pi_vol = (
        beta_2 * (J - q) ** 2 + alpha_r * ufl.inner(ufl.grad(q), ufl.grad(q))
    ) * dx_third

    return gamma / 2 * (Pi_rot + Pi_vol)


WRIGGERS = Formulation(
    label="Wriggers first-order",
    tag="Wriggers",
    u_degree=1,  
    q_body=("default", 2),
    q_third=("GLL", 1),  # Gauss-Lobatto degree 1: the four cell vertices
    aux_fields=(("p", ("Lagrange", 1)), ("q", ("Lagrange", 1))),
    regularization=regularization_wriggers,
    v_bar=-1.0,  # final applied vertical displacement [m]
)

wriggers = build_problem(WRIGGERS)
loads_wriggers, forces_wriggers = run_load_path(wriggers)

# %% [markdown]
# ```{figure} tmc_cbox_Wriggers_disp.gif
# :alt: Successive deformed configurations of the C-box with first-order regularization.
# :align: center
# :label: fig-tmc-wriggers
#
# Contact evolution for Wriggers regularization.
# ```

# %% [markdown]
# ## Deformation-gradient-based first-order regularization

# %% tags=["hide-input", "hide-output"]
p_theta = dolfinx.fem.Constant(mesh, 5.0e-2) # penalty-like parameter 
alpha_r = dolfinx.fem.Constant(mesh, 1.0e-2) # regularization parameter for the gradient of auxiliary field 
theta_deg = 1  # polynomial degree of the auxiliary field theta

def regularization_theta(u, aux, dx_third):
    theta = aux[0]
    F, J, _ = kinematics(u)

    penalty_term = theta - F

    Pi_theta = (
        p_theta / 2 * ufl.inner(penalty_term, penalty_term) 
    ) * dx_third

    Pi_reg = (
        gamma * alpha_r / 2 * ufl.inner(ufl.grad(theta), ufl.grad(theta))
    ) * dx_third

    return (Pi_theta + Pi_reg)

VORWERK_THETA = Formulation(
    label="Vorwerk first-order",
    tag="Vorwerk",
    u_degree=1,
    theta_degree=theta_deg,
    q_body=("default", 2),
    q_third=("GLL", 1),  # Gauss-Lobatto degree 1: the four cell vertices
    aux_fields=(("theta", ("Lagrange", theta_deg, (tdim, tdim))),),
    regularization=regularization_theta,
    v_bar=-1.0,  # final applied vertical displacement [m]
)

vorwerk_theta = build_problem(VORWERK_THETA)
loads_vorwerk_theta, forces_vorwerk_theta = run_load_path(vorwerk_theta)

# %% [markdown]
# ## Comparison

# %% tags=["hide-input", "hide-output"]
if comm.rank == 0:
    plt.figure(dpi=300)
    # plt.title("C-shape box: reaction at the loaded corner", fontsize=12)
    plt.xlabel(r"Applied displacement $|u_y|$", fontsize=12)
    plt.ylabel(r"Vertical reaction force $|R_y|$", fontsize=12)
    plt.grid(linewidth=0.25)
    for loads, forces, formulation in (
        (loads_hulu, forces_hulu, HULU),
        (loads_wriggers, forces_wriggers, WRIGGERS),
        (loads_vorwerk_theta, forces_vorwerk_theta, VORWERK_THETA),
    ):
        plt.plot(
            loads, forces, linestyle="-", linewidth=1.0, marker=".", markersize=4.0,
            label=formulation.label,
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

