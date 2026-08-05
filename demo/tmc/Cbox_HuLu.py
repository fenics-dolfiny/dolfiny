"""
Third medium contact example from DOI: https://doi.org/10.1016/j.cma.2024.117595
C-Box in 2D plane-strain with HuHu-LuLu regularization
"""

from mpi4py import MPI
import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from dolfinx.io import VTXWriter
from dolfinx import fem

from dolfinx import fem
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc
import argparse

# For timing the code
from datetime import datetime

# Basic settings
name = "cbox"
comm = MPI.COMM_WORLD

# User parameters (argparse)
parser = argparse.ArgumentParser(
    description="C-Box benchmark, HuHu-LuLu regularization."
)
parser.add_argument("--reg", choices=("Hu", "HuLu"), default="Hu",
                    help="Regularization type: Hu (Hessian) or HuLu (Hessian + Laplacian)")
parser.add_argument("--reg_scaling", action="store_true", default=False,
                    help="apply the Frederiksen exp(-5*det(F)) ad-hoc scaling to the HuHu-LuLu "
                         "regularization (Eq. 6) -- breaks tangent symmetry")

args = parser.parse_args()

# Dimensions
L = 1.0
H = 0.5
T = 0.1
Nx = 40
Ny = 20
dL = L / Nx # element size 

# ## finer mesh
# Nx = 80
# Ny = 40
# dL = L / Nx 

tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol)

def thirdmedium_layer(x):
    return (x[0] >= L - tol)

def left(x):
    return np.isclose(x[0], 0.0)

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

tdim = mesh.topology.dim # 2 for 2D
fdim = tdim - 1 # 1 for facets in 2D

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

# # Export mesh and markers for inspection
# with dolfinx.io.XDMFFile(comm, f"{name}/{name}_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(ct, mesh.geometry)
#     xdmf.write_meshtags(ft, mesh.geometry)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

# Integration measures
QRULE_BODY, QDEG_BODY = "default", 4
QRULE_TM, QDEG_TM = "GLL", 3
# metadata = {"quadrature_rule": "GLL", "quadrature_degree": 3}
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Create third medium submesh 
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

# Create function spaces and functions 
element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))
V_tm = fem.functionspace(mesh, ("DG", 0)) 

u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

tm_func = fem.Function(V_tm, name="cell_markers")
tm_func.x.array[:] = ct.values

# Define state and variation
m = [u]
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V))
(δu,) = δm

pprint(f"Total number of DOFs: {len(u.x.array)}")

# Kinematics (2D plane strain)
X = ufl.SpatialCoordinate(mesh)
I = ufl.Identity(tdim)
F_2D = I + ufl.grad(u)
J = ufl.det(F_2D)
C_2D = F_2D.T * F_2D
I1 = ufl.tr(C_2D) + 1 # add 1 to account for plane strain out-of-plane component

## Material Properties
# Body
E = 1.0
nu = 0.4
K = E / (3 * (1 - 2 * nu))  # 5/3
mu = E / (2 * (1 + nu))     # 5/14
K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)
Psi_body = K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2/3) * I1 - 3)

Pi = (
    Psi_body * dxVol
)

# Third medium
gamma = fem.Constant(mesh, 1.0e-6)
Psi_third = mu_body / 2 * (J ** (-2/3) * I1 - 3) 
Pi_third = gamma * Psi_third * dxThird

# regularization
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
Ell = dolfinx.fem.Constant(mesh, np.max(L_i))  # 1.0
alpha = fem.Constant(mesh, 1.0e-06)
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K)

regularization_type = args.reg

Hu = ufl.grad(ufl.grad(u)) # Hessian of displacement
Lu = ufl.div(ufl.grad(u)) # Laplacian of displacement

HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(I)

Pi_Hu = k_r / 2 * (HuHu) * dxThird
Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird  # without exp(-5|F|) to preserve symmetry of tangent problem

if regularization_type == "Hu":
    pprint("Using HuHu regularization")
    Pi_r = Pi_Hu
elif regularization_type == "HuLu":
    pprint("Using HuHu-LuLu regularization")
    Pi_r = Pi_HuLu

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

# Define the residual and forms for the nonlinear problem
if args.reg_scaling:
    # Frederiksen Eq. (6): regularization contribution written directly as a residual, with
    # the exp(-5*det(F)) weight applied to the variation of the selected regularization.
    # delta-u enters via the test function derivatives Hδu, Lδu.
    Hdu = ufl.grad(ufl.grad(δu))
    Ldu = ufl.div(ufl.grad(δu))
    reg_scale = ufl.exp(-5.0 * J)
    if regularization_type == "Hu":
        R_reg = k_r * reg_scale * ufl.inner(Hu, Hdu) * dxThird
    elif regularization_type == "HuLu":
        R_reg = k_r * reg_scale * (
            ufl.inner(Hu, Hdu) - ufl.inner(Lu, Ldu) / ufl.tr(I)
        ) * dxThird
    residual = ufl.derivative(Pi + Pi_third, m, δm) + R_reg
else:
    residual = ufl.derivative(Pi + Pi_third + Pi_r, m, δm)

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

problem = dolfiny.snesproblem.SNESProblem(
    forms,
    m,
    bcs=bcs,
    prefix=name,
    entity_maps=[medium_map],
)

# problem = dolfinx.fem.petsc.NonlinearProblem(
#     residual,
#     u,
#     bcs=bcs,
#     entity_maps=[medium_map],
#     petsc_options_prefix=name,
#     petsc_options={
#         "snes_type": "newtonls",
#         "snes_linesearch_type": "bt",
#         # "snes_linesearch_order": 1,
#         "snes_atol": 1e-8,
#         "snes_rtol": 1e-8,
#         "snes_max_it": 50,
#         "snes_monitor": None,
#         "snes_converged_reason": None,
#         #"snes_error_if_not_converged": True,
#         "ksp_type": "preonly",
#         "pc_type": "lu",
#         # "pc_type": "cholesky",
#         "pc_factor_mat_solver_type": "mumps",
#     },
# )

# output file for storing results
ofile = VTXWriter(comm, f"{name}/{name}_{regularization_type}.bp", [u, tm_func])
ofile.write(0.0) # write initial state

# Adaptive loading strategy -- setup parameters
adaptive_load = True
MAX_FAILURES = 2
dl = 0.05 # initial load increment
dl_min = dl / 16

# store previous solution and cell markers for adaptive loading
u_prev = u.x.array.copy()
tm_prev = tm_func.x.array.copy()

v_bar = -0.7   # final applied vertical displacement

num_iterations = 0 # store total number of iterations across all loading steps
load = dl
last_load = 0.0 # load of the last converged step, i.e. the state stored in u_prev
n = 0 # used to track the number of successive failures for adaptive loading
ii = 1 # load step counter

# Setting up the reaction force computation (parallel compatible)
disp_residual = ufl.derivative(Pi, u)
reaction_form = fem.form(disp_residual)
reaction_vector = dolfinx.fem.petsc.create_vector(V)
owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
dofs_point_y_owned = dofs_point_y[dofs_point_y < owned_size]
force_array = [] # store reaction force at the top right corner for each load step
loading_array = [] # store applied load for each load step
pointA_array = [] # store the vertical displacement of point A for each load step

# min(J) over the third medium, evaluated at the same GLL points used for the assembly 
gll_points, _ = basix.make_quadrature(
    basix.CellType.quadrilateral, QDEG_TM, basix.QuadratureType.gll
)
J_third = fem.Expression(J, gll_points)
tm_cells = ct.find(TM_marker)
tm_cells = tm_cells[tm_cells < num_cells_owned].astype(np.int32)

# Point A of Frederiksen et al. Fig. 4: the lower right corner of the C-shape, at (L, 0).
nodeA = dolfinx.mesh.locate_entities(
    mesh, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], 0.0)
)
dofA_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, nodeA)
dofA_y_owned = dofA_y[dofA_y < owned_size]

# print a message for simulation startup
pprint("------------------------------------")
pprint("Simulation Start")
pprint("------------------------------------")
# Store start time 
startTime = datetime.now()

while load <= (abs(v_bar) + tol):
    
    # Update boundary condition value
    applied_y.value = - load
    
    pprint(f"\nLoad step {ii}, u_y: {applied_y.value:.3f}", flush=True)

    # Solve the problem
    problem.solve(u_init=m)

    # Assert convergence of nonlinear solver
    reason = problem.status(verbose=True)
    
    num_iterations += problem.snes.getIterationNumber()
    
    if reason < 0:
        if adaptive_load and dl / 2 >= dl_min:
            # Reject the step and retry from the last equilibrium with half the increment.
            n += 1
            dl = dl / 2 # half load increment
            load = last_load + dl
            u.x.array[:] = u_prev
            u.x.scatter_forward() # the restored ghost values would otherwise be stale
            tm_func.x.array[:] = tm_prev
            pprint(f"  step rejected ({reason}); retrying with dl = {dl:.5f}")
        else:
            pprint(f"Solver failed to converge (reason {reason}) at dl = {dl:.5f}, aborting.")
            break

    else:
        last_load = load
        ofile.write(load)
        ii += 1
        n = 0  # reset the failure counter for adaptive loading

        # Reporting min(J)
        Jc = J_third.eval(mesh, tm_cells)
        minJ = comm.allreduce(float(Jc.min()) if Jc.size else np.inf, op=MPI.MIN)

        # reaction force at the top right corner
        with reaction_vector.localForm() as reaction_local:
            reaction_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(reaction_vector, reaction_form)
        reaction_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        force_y = comm.allreduce(reaction_vector.array[dofs_point_y_owned].sum(), op=MPI.SUM)
        force_array.append(abs(force_y))
        loading_array.append(abs(applied_y.value))

        # vertical displacement of point A
        pointA_array.append(comm.allreduce(u.x.array[dofA_y_owned].sum(), op=MPI.SUM))

        pprint(
            f"lambda = {applied_y.value:.4f}, reaction force = "
            f"{force_y:.6f}, min(J) third medium = {minJ:.3e}"
        )

        load += dl
        u_prev[:] = u.x.array
        tm_prev[:] = tm_func.x.array

    if adaptive_load and n > MAX_FAILURES:
        pprint("Too many failures, aborting.")
        break

ofile.close() # close output file

# Store end time and compute elapsed time
endTime = datetime.now()
elapsedTime = endTime - startTime

pprint("-----------------------------------------")
pprint("End computation")
pprint(f"Elapsed time: {elapsedTime}")
pprint(f"Total number of iterations: {num_iterations}")

