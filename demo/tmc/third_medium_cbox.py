# %% [markdown]
# ---
# authors:
#   - az
#   - mh
#   - ptk
#   - name: Andrea Antonelli
#     email: aantonelli@student.ethz.ch
#     affiliation: ETH Zürich
# ---

# %% [markdown]
# # Third medium contact
#
# This demo solves the C-shape benchmark problem with the Third Medium Contact (TMC) method, using
# different regularization techniques proposed in the literature.

# In particular, this demo emphasizes:
# - setting up and solving large-deformation contact problems with the TMC method
# - the effect of different regularization strategies on convergence behavior and third medium
#   deformation
# - multi-domain formulation for the body and third medium regions requiring the use of submeshes
#   and entity maps
# - adaptive load-stepping strategy for highly nonlinear problems
#
# ## Geometry
#
# The C-shape configuration, first introduced by {cite:t}`Bluhm2021`, consists of a C-shaped elastic
# body of length $L$, height $H = 0.5 L$ and thickness $T = 0.1 L$, clamped on the left side and
# loaded by a vertical displacement at the top-right corner node of the upper beam. The third medium
# fills the gap between the two beams and is further extended by an additional column of elements
# from $x = L$ to $x = L + \Delta L$ to completely embed the potential contact region. The geometry
# is discretized with quadrilateral elements to maintain mesh uniformity across different
# regularization strategies, although first-order regularizations allow the use of triangular
# elements as well.
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

# Dimensions [mm]
L = 1.0  # length of the C-shape
H = 0.5  # height of the C-shape
T = 0.1  # thickness of the two beams

Nx, Ny = 40, 20  # cells along x and y within the C-shape
dL = L / Nx  # characteristic cell size

# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0, 0], [L + dL, H]],
    [Nx + 1, Ny],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
    ghost_mode=dolfinx.mesh.GhostMode.none,
)

tdim = mesh.topology.dim
fdim = tdim - 1

# Define subdomains to mark third medium and body cells
tol = 1.0e-6


def thirdmedium(x):
    return (x[0] >= L - tol) | ((x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol))


def left(x):
    return np.isclose(x[0], 0.0)


# Mark cells
BODY_marker = 1
TM_marker = 2

mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(0, tdim)

im_c = mesh.topology.index_map(tdim)
num_cells_local = im_c.size_local + im_c.num_ghosts
markers = np.full(num_cells_local, BODY_marker, dtype=np.int32)
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium)] = TM_marker
ct = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells_local), markers)
ct.name = "cell_tags"

# Mark facets
LEFT_marker = 2

im_f = mesh.topology.index_map(fdim)
num_facets_local = im_f.size_local + im_f.num_ghosts

facet_markers = np.zeros(num_facets_local, dtype=np.int32)
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, left)] = LEFT_marker
ft_indices = np.flatnonzero(facet_markers)
ft = dolfinx.mesh.meshtags(mesh, fdim, ft_indices, facet_markers[ft_indices])
ft.name = "facet_tags"

# Locate the top-right corner node for applying the vertical displacement
node_topr = dolfinx.mesh.locate_entities(
    mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H)
)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)
pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

# Create third medium submesh
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(mesh, tdim, ct.find(TM_marker))[0:2]

# DG0 field carrying the region tags, used to identify body and third medium cells
V_tm = fem.functionspace(mesh, ("DG", 0))
tm_func = fem.Function(V_tm, name="cell_markers")

cell_dofs = V_tm.dofmap.list[ct.indices].flatten()
tm_func.x.array[cell_dofs] = ct.values
tm_func.x.scatter_forward()


def frame_recorder(comm):
    """Collect the deformed states of one run for later side-by-side display.
    Rendering works in serial only.
    """

    if comm.size > 1:
        return None, None

    frames = []

    def record_step(_u, _u_y):
        frames.append((float(_u_y), _u.x.array.copy()))

    return frames, record_step


# %% [markdown]
# ## TMC formulation

# The TMC approach allows to solve large-deformations contact problems between deformable bodies by
# introducing a fictitious hyperelastic continuum, the third medium, between the contacting
# surfaces. The constitutive response of the third medium is designed to be highly compliant in the
# pre-contact phase while rapidly stiffening when compressed to near-zero volume, allowing the
# transfer of contact tractions. In this way, the contact problem reduces to a standard
# hyperelasticity problem, thus avoiding the explicit contact search and the imposition of
# inequality constraints as in classical contact algorithms.

# Generally, the body and third medium are modeled with the same compressible Neo-Hookean strain
# energy density in the form:

# $$ \Psi_{body}(\boldsymbol{u}) = \frac{\mu}{2}\left(J^{-2/3}\operatorname{tr}\boldsymbol{C} -
# 3\right) + \frac{K}{2}\left(\ln J\right)^{2}, $$

# with $\boldsymbol{F} = \boldsymbol{I} + \nabla\boldsymbol{u}$ the deformation gradient, $J =
# \det\boldsymbol{F}$ its determinant and $\boldsymbol{C} = \boldsymbol{F}^{T}\boldsymbol{F}$ the
# right Cauchy-Green deformation tensor. $K$ and $\mu$ are, respectively, the initial bulk and shear
# moduli of the body. To keep the influence of the third medium negligible before contact, its
# strain energy density is scaled by a small parameter $\gamma$, with the stiffening contribution
# coming from the $\ln J$ volumetric term as $J \rightarrow 0$. However, as discussed in
# {cite:t}`Faltus2024`, in 2D plane-strain conditions this term can be omitted, since the isochoric
# term stiffens as $J \rightarrow 0$, combined with the plane strain constraint $F_{33} = 1$, is
# sufficient to prevent penetration. Therefore, the strain energy density for the third medium
# reduces to:

# $$ \Psi_{tm}^{2D}(\boldsymbol{u}) = \gamma
# \left[\frac{\mu}{2}\left(J^{-2/3}\operatorname{tr}\boldsymbol{C} - 3\right)\right]. $$

# When large deformations occur in the pre-contact phase, the third medium elements become severely
# distorted, preventing convergence in the nonlinear solution process. A way to mitigate this issue
# is to introduce a regularization contribution $\Psi_r$ to the third medium strain energy density
# that should provide sufficient stabilization while minimally influencing its deformation. Several
# forms of $\Psi_r$ have been proposed in the literature, mainly differing in the deformation modes
# they penalize.


# The material parameters for the body are set from the Young's modulus $E = 1.0$ MPa and Poisson's
# ratio $\nu = 0.4$, leading to $K = 5/3$ MPa and $\mu = 5/14$ MPa, while the relative contact
# stiffness is set to $\gamma = 10^{-6}$.

# %%
## Material Properties
E = 1.0  # Young's modulus of the body [MPa]
nu = 0.4  # Poisson ratio of the body
K = E / (3 * (1 - 2 * nu))  # bulk modulus [MPa]
mu = E / (2 * (1 + nu))  # shear modulus [MPa]

K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)

gamma = fem.Constant(mesh, 1.0e-6)  # relative contact stiffness of the third medium


def kinematics(u):
    """Deformation gradient, its determinant and the plane-strain first invariant."""
    F = ufl.Identity(tdim) + ufl.grad(u)
    J = ufl.det(F)
    I1 = ufl.tr(F.T * F) + 1.0  # +1: out-of-plane component, F_33 = 1
    return F, J, I1


def psi_body(J, I1):
    return K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2 / 3) * I1 - 3)


def psi_third(J, I1):
    return mu_body / 2 * (J ** (-2 / 3) * I1 - 3)  # isochoric part only, no volumetric term


# %% [markdown]
# ## HuHu-LuLu regularization
#
# The first regularization was proposed by {cite:t}`Bluhm2021` and penalizes higher-order
# deformation modes through the Hessian of the displacement field $\mathbb{H}\boldsymbol{u}$, hence
# the name "HuHu regularization":
#
# $$ \Psi_r^{Hu}(\boldsymbol{u}) = \frac{k_r}{2} \mathbb{H}\boldsymbol{u} \cdot
# \mathbb{H}\boldsymbol{u}, $$
#
# where $k_r = \alpha L^2 K$, $L$ is a characteristic length of the problem and $\alpha$ a
# dimensionless constant to be chosen as small as possible while still stabilizing the third medium.
# A value of $\alpha = 10^{-6}$ is adopted in the following.
#
# To reduce the penalization on bending and quadratic compression modes, {cite:t}`Frederiksen2025`
# proposed to subtract a term in the Laplacian of the displacement field $\mathbb{L}\boldsymbol{u}$
# from the HuHu regularization:
#
# $$ \Psi_r^{HuLu} = \frac{k_r}{2} (\mathbb{H} \boldsymbol{u} \cdot \mathbb{H} \boldsymbol{u}
# - \frac{1}{\operatorname{tr} {\boldsymbol{I}}} \mathbb{L} \boldsymbol{u} \cdot \mathbb{L}
# \boldsymbol{u}), $$
#
# leading to the so-called "HuHu-LuLu regularization". Both regularizations rely on second
# derivatives of the displacement field, hence at least quadratic elements are required to obtain an
# effective, spatially varying Hessian and Laplacian within each cell. Moreover, for the C-shape
# benchmark considered here the third medium is dominated by shear, skew and linear compression
# deformation modes that HuHu and HuHu-LuLu penalize equivalently. The two regularizations are
# therefore expected to produce almost identical results, as already observed in
# {cite:t}`Frederiksen2025`. A Gauss-Lobatto quadrature rule is employed for third medium elements
# to mitigate element inversion, increasing robustness under severe skew deformations.
#
# %% Define function spaces and functions
element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))

u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

# Define state
m = [u]
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V))
(δu,) = δm

# BCs
left_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(LEFT_marker))
bc_left = dolfinx.fem.dirichletbc(np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V)
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node_topr)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]

F, J, I1 = kinematics(u)

# Integration measures
QRULE_BODY, QDEG_BODY = "default", 4
QRULE_TM, QDEG_TM = "GLL", 3
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})

# Define the potential energy contributions Elastic energy of the body
Pi_body = psi_body(J, I1) * dxVol

# Third medium elastic energy
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
L_char = dolfinx.fem.Constant(mesh, np.max(L_i))
alpha = fem.Constant(mesh, 1.0e-06)
k_r = fem.Constant(mesh, alpha.value * L_char.value**2 * K)

Hu = ufl.grad(ufl.grad(u))  # Hessian of displacement
Lu = ufl.div(ufl.grad(u))  # Laplacian of displacement

HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(ufl.Identity(tdim))

# Pi_Hu = k_r / 2 * (HuHu) * dxThird

Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird
Pi_r = Pi_HuLu

# %% [markdown]
# ### Solution strategy
#
# Solving this contact problem with the TMC method amounts to finding the stationary point of the
# total potential energy of the system,
#
# $$ \Pi(\boldsymbol{u}) = \int_{\Omega_{b}} \Psi_{body}(\boldsymbol{u}) \, \mathrm{d}\Omega
# + \int_{\Omega_{tm}} \Psi_{tm}^{2D}(\boldsymbol{u}) \, \mathrm{d}\Omega
# + \int_{\Omega_{tm}} \Psi_{r}^{HuLu}(\boldsymbol{u}) \, \mathrm{d}\Omega , $$
#
# with $\Omega_{b}$ and $\Omega_{tm}$ the body and third medium subdomains. The load is applied
# through prescribed displacement only, so no external work term appears. The nonlinear system is
# solved using the PETSc Newton solver with backtracking line search, coupled with an adaptive load
# stepping strategy to improve robustness and convergence: on failure the state is reset to the last
# converged one and the increment is halved, until the target displacement $\bar{v}$ is reached or
# the increment falls below $\Delta l_{\min}$.

# %% tags=["hide-input", "hide-output"] Define the residual and forms for the nonlinear problem
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

# Prepare vertical reaction force at the loaded corner node
disp_residual = ufl.derivative(Pi_body, u)
reaction_form = fem.form(disp_residual)
reaction_vector = dolfinx.fem.petsc.create_vector(V)
owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
dofs_point_y_owned = dofs_point_y[dofs_point_y < owned_size]

# Instantiate the nonlinear problem and solver
opts = PETSc.Options(name)
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "bt"
opts["snes_atol"] = 1.0e-08
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 50
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Setup output file for storing results
name_HuLu = f"{name}_HuLu"
ofile = VTXWriter(comm, f"{name_HuLu}.bp", [u, tm_func])
ofile.write(0.0)  # write initial state

# Deformed states of all runs, displayed side by side in the comparison section
runs = []
frames_HL, record_HL = frame_recorder(comm)

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

adaptive_load = True
dl = 0.05  # initial load increment
dl_min = dl / 16  # minimum load increment

v_bar = -1.0  # final applied vertical displacement [mm]


def run_adaptive_loading(
    name,
    problem,
    state_functions,
    displacement,
    target_displacement,
    adaptive_load,
    initial_increment,
    min_increment,
    record_step,
):
    """Run the load stepping loop for one regularization case."""
    state_snapshots = [s.x.array.copy() for s in state_functions]
    force_history = []
    load_history = []

    if record_step is not None:
        record_step(displacement, 0.0)  # undeformed configuration as the first frame

    dl = initial_increment
    load = dl
    last_load = load
    load_step = 1
    num_iterations = 0

    pprint("------------------------------------")
    pprint(f"Simulation Start ({name})")
    pprint("------------------------------------")
    start_time = datetime.now()

    while load <= (target_displacement + 1e-6):
        applied_y.value = -load

        pprint(f"\nLoad step {load_step}, u_y: {applied_y.value:.3f}", flush=True)

        problem.solve(u_init=state_functions)

        reason = problem.status(verbose=True)
        num_iterations += problem.snes.getIterationNumber()

        if reason < 0:
            if adaptive_load and (dl / 2 >= min_increment):
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
            ofile.write(load)
            if record_step is not None:
                record_step(displacement, applied_y.value)
            load_step += 1

            with reaction_vector.localForm() as reaction_local:
                reaction_local.set(0.0)
            dolfinx.fem.petsc.assemble_vector(reaction_vector, reaction_form)
            reaction_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            force_y = abs(
                comm.allreduce(reaction_vector.array[dofs_point_y_owned].sum(), op=MPI.SUM)
            )
            force_history.append(force_y)
            load_history.append(abs(applied_y.value))
            pprint(f"lambda = {applied_y.value:.3f}, reaction force = {force_y:.6e}")

            load += dl
            state_snapshots = [s.x.array.copy() for s in state_functions]

    ofile.close()

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    pprint("-----------------------------------------")
    pprint(f"End computation ({name})")
    pprint(f"Elapsed time: {elapsed_time}")
    pprint(f"Total number of iterations: {num_iterations}\n")

    return force_history, load_history, num_iterations, elapsed_time


force_array_HL, loading_array_HL, num_iterations, elapsedTime = run_adaptive_loading(
    name="HuLu",
    problem=problem,
    state_functions=m,
    displacement=u,
    target_displacement=abs(v_bar),
    adaptive_load=adaptive_load,
    initial_increment=dl,
    min_increment=dl_min,
    record_step=record_HL,
)
runs.append(("HuHu-LuLu", V, frames_HL))  # V is rebound below, so capture it here


# %% [markdown]
# ## Wriggers regularization
#
# Both regularizations above act on second derivatives of the displacement field, preventing the use
# of linear elements. {cite:t}`Wriggers2025` propose a regularization that removes this restriction
# and allows the third medium to be discretized with first-order elements.
#
# The underlying idea, initially proposed by {cite:t}`Faltus2024`, is to consider curvature at the
# element level as the primary problem of an unregularized medium, whereas stretch and volume-change
# gradients are desired behaviors during the compliant pre-contact phase. Penalizing curvature
# alone, expressed as the gradient of the rotation tensor $\boldsymbol{R}$ from the polar
# decomposition $\boldsymbol{F} = \boldsymbol{R}\boldsymbol{U}$, thus yields a more compliant medium
# than the Hessian-based forms above. In 2D the rotation tensor can be expressed in terms of a
# single angle $\varphi$, and the symmetry of $\boldsymbol{U}$ yields the explicit relation:
#
# $$ \tan\varphi = \frac{F_{12} - F_{21}}{F_{11} + F_{22}}, $$
#
# so that penalizing $\nabla\boldsymbol{R}$ reduces to penalizing $\nabla\varphi$. Penalizing the
# tangent instead of the angle itself expresses the regularization directly through the components
# of $\boldsymbol{F}$, avoiding the arctangent, and leads to:
#
# $$ \Psi_r^{\tan}(\boldsymbol{u}) = \frac{\gamma}{2}\alpha_r\left(\left\|\nabla\left[\frac{F_{12} -
# F_{21}}{F_{11} + F_{22}}\right]\right\|^2 + \|\nabla J\|^2\right). $$
#
# As only rotations are penalized, stretch deformations remain unconstrained and elements may
# approach a locally vanishing volume; the second term, involving $\nabla J$, restores control over
# the volumetric modes. This form still involves second derivatives of $\boldsymbol{u}$. They are
# eliminated by introducing two auxiliary scalar fields $p$ and $q$ which approximate $\tan\varphi$
# and $J$ in a penalty-like fashion, so that the gradient penalization acts on the auxiliary fields
# alone:
#
# $$ \Psi_r(\boldsymbol{u}, p, q) = \frac{\gamma}{2}\left[\beta_1\left(\frac{F_{12} - F_{21}}{F_{11}
# + F_{22}} - p\right)^2 + \alpha_r\|\nabla p\|^2\right] + \frac{\gamma}{2}\left[\beta_2\left(J -
# q\right)^2 + \alpha_r\|\nabla q\|^2\right]. $$
#
# Only first derivatives remain, so $\boldsymbol{u}$, $p$ and $q$ can all be interpolated with
# linear shape functions. The solution is then the stationary point of the total potential energy,
# which now depends on the auxiliary fields as well:
#
# $$ \Pi(\boldsymbol{u}, p, q) = \int_{\Omega_{b}} \Psi_{body}(\boldsymbol{u}) \, \mathrm{d}\Omega +
# \int_{\Omega_{tm}} \Psi_{tm}^{2D}(\boldsymbol{u}) \, \mathrm{d}\Omega + \int_{\Omega_{tm}}
# \Psi_{r}(\boldsymbol{u}, p, q) \, \mathrm{d}\Omega , $$
#
# solved with the same Newton solver and adaptive load-stepping strategy as above. The displacement
# uses $Q_1$ elements on the full mesh, while $p$ and $q$ are $Q_1$ fields on the third medium
# submesh only, so that the additional unknowns stay confined to $\Omega_{tm}$. A Gauss-Lobatto rule
# of degree one is used here for third medium elements.
#
# The regularization parameters are taken as $\beta_1 = 10^{4}$, $\beta_2 = 10$ and $\alpha_r =
# 100$. The $\nabla J$ contribution is retained here, although {cite:t}`Wriggers2025` observe on
# this same benchmark (Section 4.2.3) that it can be dropped when linear shape functions are used,
# since the $p$ and $(p,q)$ formulations give the same results.

# %%
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
left_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(LEFT_marker))
bc_left = dolfinx.fem.dirichletbc(np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V)
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node_topr)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]

# Integration measures for first order elements
QRULE_BODY, QDEG_BODY = "default", 2  # 2x2 Gauss-Legendre (standard)
QRULE_TM, QDEG_TM = "GLL", 1  # Gauss-Lobatto degree 1 = the four cell vertices
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})

F, J, I1 = kinematics(u)

# Define the potential energy contributions Elastic energy of the body
Pi_body = psi_body(J, I1) * dxVol

# Third medium elastic energy
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
beta_1 = dolfinx.fem.Constant(mesh, 1.0e4)
beta_2 = dolfinx.fem.Constant(mesh, 10.0)
alpha_r = dolfinx.fem.Constant(mesh, 100.0)

trF = ufl.tr(F)
skF = F[0, 1] - F[1, 0]

skew_term = (skF / trF) - p1

Pi_grad = (beta_1 * skew_term**2 + alpha_r * ufl.inner(ufl.grad(p1), ufl.grad(p1))) * dxThird

Pi_J = (beta_2 * (J - q) ** 2 + alpha_r * ufl.inner(ufl.grad(q), ufl.grad(q))) * dxThird

Pi_r = gamma / 2 * (Pi_grad + Pi_J)

# %% tags=["hide-input", "hide-output"] Nonlinear problem and solver using new regularization term
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

disp_residual = ufl.derivative(Pi_body, u)
reaction_form = fem.form(disp_residual)
reaction_vector = dolfinx.fem.petsc.create_vector(V)
owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
dofs_point_y_owned = dofs_point_y[dofs_point_y < owned_size]

# output file for storing results
name_W = f"{name}_Wriggers"
ofile = VTXWriter(comm, f"{name_W}.bp", [u, tm_func])
ofile.write(0.0)  # write initial state

frames_W, record_W = frame_recorder(comm)

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

force_array_W, loading_array_W, num_iterations, elapsedTime = run_adaptive_loading(
    name="Wriggers",
    problem=problem,
    state_functions=m,
    displacement=u,
    target_displacement=abs(v_bar),
    adaptive_load=True,
    initial_increment=dl,
    min_increment=dl_min,
    record_step=record_W,
)
runs.append(("Wriggers first-order", V, frames_W))

# %% [markdown]
# ## Deformation-gradient-based regularization
#
# Another low-order-compatible regularization strategy has recently been proposed by
# {cite:t}`Vorwerk2026`. Starting from the HuHu regularization, which penalizes spatial variations
# of $\boldsymbol{F}$, they introduce a deformation-gradient-like field $\boldsymbol{\Theta}$ in the
# third medium that is weakly coupled to the physical deformation gradient through a penalty term in
# the form:
#
# $$ \tilde{\Psi}_r(\boldsymbol{u}, \boldsymbol{\Theta}) =
# \frac{p_{\Theta}}{2}\left\|\boldsymbol{\Theta} - \boldsymbol{F}\right\|^{2}, $$
#
# with $p_{\Theta}$ the penalty parameter. The regularization is then applied directly to the
# gradient of $\boldsymbol{\Theta}$, leading to the final form of the regularization contribution:
#
# $$ \Psi_r(\boldsymbol{u}, \boldsymbol{\Theta}) = \frac{p_{\Theta}}{2}\left\|\boldsymbol{\Theta} -
# \boldsymbol{F}\right\|^{2}
# + \frac{\alpha_r}{2}\left\|\nabla\boldsymbol{\Theta}\right\|^{2}, $$
#
# where $\alpha_r$ is the regularization parameter. The new field is discretized independently of
# the displacement, and since only first derivatives of $\boldsymbol{\Theta}$ appear, a gradient
# control of $\boldsymbol{F}$ is recovered without evaluating second derivatives of
# $\boldsymbol{u}$, so that first-order elements become admissible for both fields. The potential
# energy of the system now takes the form:
#
# $$ \Pi(\boldsymbol{u}, \boldsymbol{\Theta}) = \int_{\Omega_{b}} \Psi_{body}(\boldsymbol{u}) \,
# \mathrm{d}\Omega + \int_{\Omega_{tm}} \Psi_{tm}^{2D}(\boldsymbol{u}) \, \mathrm{d}\Omega +
# \int_{\Omega_{tm}} \Psi_{r}(\boldsymbol{u}, \boldsymbol{\Theta}) \, \mathrm{d}\Omega , $$
#
# and the stationary conditions lead to a nonlinear system solved as above. The displacement uses
# $Q_1$ elements on the full mesh, while $\boldsymbol{\Theta}$ is a $Q_1$ tensor-valued field on the
# third medium submesh only.
#
# The regularization parameters are taken as $p_{\Theta} = 5 \cdot 10^{-2}$ and $\alpha_r =
# 10^{-8}$.
#
# %%
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
left_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(LEFT_marker))
bc_left = dolfinx.fem.dirichletbc(np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V)
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, node_topr)
applied_y = dolfinx.fem.Constant(mesh, 0.0)
bc_point_y = dolfinx.fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_y]

F, J, I1 = kinematics(u)

# Define the potential energy contributions Elastic energy of the body
Pi_body = psi_body(J, I1) * dxVol

# Third medium elastic energy
Pi_third = gamma * psi_third(J, I1) * dxThird

# Regularization
p_theta = dolfinx.fem.Constant(mesh, 5.0e-2)
alpha_r = dolfinx.fem.Constant(mesh, 1.0e-8)

# initialize theta as the Identity tensor
I_tm = fem.Constant(third_medium_mesh, np.eye(tdim, dtype=dolfinx.default_scalar_type))
theta.interpolate(fem.Expression(I_tm, V_theta.element.interpolation_points))

penalty_term = theta - F
Pi_penalty = (p_theta / 2 * ufl.inner(penalty_term, penalty_term)) * dxThird

Pi_reg = (alpha_r / 2 * ufl.inner(ufl.grad(theta), ufl.grad(theta))) * dxThird

Pi_r = Pi_penalty + Pi_reg

# %% tags=["hide-input", "hide-output"]

# Nonlinear problem and solver using new regularization term
residual = ufl.derivative(Pi_body + Pi_third + Pi_r, m, δm)
forms = ufl.extract_blocks(residual)

disp_residual = ufl.derivative(Pi_body, u)
reaction_form = fem.form(disp_residual)
reaction_vector = dolfinx.fem.petsc.create_vector(V)
owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
dofs_point_y_owned = dofs_point_y[dofs_point_y < owned_size]

# output file for storing results
name_V = f"{name}_Vorwerk"
ofile = VTXWriter(comm, f"{name_V}.bp", [u, tm_func])
ofile.write(0.0)  # write initial state

frames_V, record_V = frame_recorder(comm)

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

force_array_V, loading_array_V, num_iterations, elapsedTime = run_adaptive_loading(
    name="Vorwerk",
    problem=problem,
    state_functions=m,
    displacement=u,
    target_displacement=abs(v_bar),
    adaptive_load=True,
    initial_increment=dl,
    min_increment=dl_min,
    record_step=record_V,
)
runs.append(("Deformation-gradient-based", V, frames_V))

# %% [markdown]
# ## Comparison


# %% tags=["hide-input"]
def warped_surface(grid, subdivision_levels):
    """Warp the reference grid by u and tessellate it for rendering.

    Returns the coloured surface and element outlines.
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


def plot_comparison_pyvista(filename, runs, tm_func, num_cells_owned, tdim):
    """Replay the recorded runs side by side into a single animation."""

    panel_px = 900  # Edge length of one panel [px]
    plotter = pv.Plotter(
        off_screen=False,
        shape=(1, len(runs)),
        window_size=(panel_px * len(runs), panel_px),
        theme=dolfiny.pyvista.theme,
    )

    panels = []
    # All panels share one framing, wide enough to hold every state of every run.
    corner_min = np.full(tdim, np.inf)
    corner_max = np.full(tdim, -np.inf)

    for i, (label, V_run, frames) in enumerate(runs):
        grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_run))

        # A higher-order cell is drawn by tessellating it into flat sub-cells; the level controls
        # how finely. Linear cells need no subdivision.
        subdivision_levels = 0 if grid.get_cell(0).is_linear else 3

        # tm_func (BODY_marker=1 / TM_marker=2) is DG0: one value per cell, so it maps directly to
        # cell_data. The region assignment is fixed for the whole run, so this is set once here. It
        # lives on the parent mesh, shared by all runs.
        grid.cell_data["cell_marker"] = tm_func.x.array[:num_cells_owned]

        # VTK points are always stored as 3D, even for a 2D mesh, so the displacement is padded with
        # a zero out-of-plane component before being handed over.
        u_3d = np.zeros((grid.n_points, 3))
        grid.point_data["u"] = u_3d

        reference = grid.points[:, :tdim]
        for _, u_dofs in frames:
            deformed = reference + u_dofs.reshape((-1, tdim))
            corner_min = np.minimum(corner_min, deformed.min(axis=0))
            corner_max = np.maximum(corner_max, deformed.max(axis=0))

        surface, edges = warped_surface(grid, subdivision_levels)

        plotter.subplot(0, i)
        # The two colours only separate body from third medium; the figure caption says which is
        # which.
        plotter.add_mesh(
            surface, scalars="cell_marker", n_colors=2, clim=(1, 2), show_scalar_bar=False
        )
        plotter.add_mesh(
            edges,
            style="wireframe",
            color="black",
            line_width=dolfiny.pyvista.pixels // 1000,
        )
        plotter.add_text(
            label,
            position="upper_edge",
            font_size=panel_px // 45,
            font="courier",
        )
        # Counter showing the applied vertical displacement of the current frame.
        load_text = plotter.add_text(
            "", position=(0.04, 0.04), viewport=True, font_size=panel_px // 45
        )

        panels.append((frames, subdivision_levels, grid, u_3d, surface, edges, load_text))

    # Set up camera position for all panels.
    centre = (corner_min + corner_max) / 2
    span = corner_max - corner_min
    panel_aspect = 1.0  # panel width / height, as set by window_size
    margin = 1.15  # empty rim around the extreme configuration
    for i in range(len(panels)):
        plotter.subplot(0, i)
        # The theme's parallel projection reaches only the first renderer, so the remaining panels
        # have to be switched over explicitly or they render in perspective.
        plotter.enable_parallel_projection()
        plotter.camera_position = [
            (centre[0], centre[1], 1.0),
            (centre[0], centre[1], 0.0),
            (0.0, 1.0, 0.0),
        ]
        plotter.camera.parallel_scale = margin * max(span[1], span[0] / panel_aspect) / 2

    # The runs do not reach the same load level: a panel that runs out of frames holds its last
    # converged state until the end.
    num_frames = max(len(frames) for frames, *_ in panels)

    plotter.open_gif(filename, fps=5)
    for frame in range(num_frames):
        for frames, subdivision_levels, grid, u_3d, surface, edges, load_text in panels:
            index = min(frame, len(frames) - 1)
            u_y, u_dofs = frames[index]
            u_3d[:, :tdim] = u_dofs.reshape((-1, tdim))
            grid.point_data["u"] = u_3d
            new_surface, new_edges = warped_surface(grid, subdivision_levels)
            surface.copy_from(new_surface)
            edges.copy_from(new_edges)
            load_text.input = f"u_y = {u_y:.3f}"
        plotter.render()
        plotter.write_frame()

    plotter.close()
    plotter.deep_clean()


if comm.size == 1:
    plot_comparison_pyvista(f"{name}_comparison.gif", runs, tm_func, num_cells_owned, tdim)

# %% [markdown]
# ```{figure} tmc_cbox_comparison.gif
# :alt: Deformed configurations of the C-box for the three regularizations, side by side.
# :align: center
# :label: fig-tmc-comparison
#
# Contact evolution for the HuHu-LuLu, Wriggers first-order and deformation-gradient-based
# regularizations, with the elastic body in blue and the third medium in red.
# ```


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
        (line,) = plt.plot(
            loads,
            forces,
            linestyle="-",
            linewidth=1.0,
            marker=".",
            markersize=4.0,
            label=formulation,
        )
        # Mark the last reached displacement with a larger cross in the curve colour.
        plt.plot(
            loads[-1],
            forces[-1],
            linestyle="none",
            marker="x",
            markersize=10.0,
            markeredgewidth=2.0,
            color=line.get_color(),
            zorder=5,
        )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{name}_force_displacement.png", dpi=300)
    plt.close()

# %% [markdown]
# ```{figure} tmc_cbox_force_displacement.png
# :alt: Reaction force against applied displacement for all regularizations.
# :align: center
# :label: fig-tmc-force-displacement
#
# Vertical reaction-displacement curves for the three regularizations, evaluated at the
# loaded corner node. The cross marks the last converged step for each simulation.
# ```
