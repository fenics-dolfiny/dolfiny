"""
Third medium contact example from DOI: https://arxiv.org/abs/2606.28036
C-Box in 2D plane-strain with "deformation-gradient-based" regularization
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
name = "cbox_Vorwerk"
comm = MPI.COMM_WORLD

# User parameters (argparse)
parser = argparse.ArgumentParser(
    description="C-Box benchmark, Vorwerk regularization."
)
parser.add_argument("--ct", choices=("quad", "tri"), default="quad",
                    help="Element type: quad (quadrilateral) or tri (triangle)")

args = parser.parse_args()

# Dimensions
L = 1.0
H = 0.5
T = 0.1
Nx = 40
Ny = 20
dL = L / Nx # element size

# # finer mesh
# dL = 0.02
# Nx = int(L/dL)
# Ny = int(H/dL)

tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol)

def thirdmedium_layer(x):
    return (x[0] >= L - tol)

def left(x):
    return np.isclose(x[0], 0.0)

# Element type
tri = dolfinx.mesh.CellType.triangle
quad = dolfinx.mesh.CellType.quadrilateral

if args.ct == "quad":
    cell_type = quad
elif args.ct == "tri":
    cell_type = tri

# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0, 0], [L+dL, H]],
    [Nx+1, Ny],
    cell_type=cell_type,
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

# # # Export mesh and markers for inspection
# # with dolfinx.io.XDMFFile(comm, f"{name}/{name}_mesh.xdmf", "w") as xdmf:
# #     xdmf.write_mesh(mesh)
# #     xdmf.write_meshtags(ct, mesh.geometry)
# #     xdmf.write_meshtags(ft, mesh.geometry)
# # pprint(f"Mesh saved to {name}/{name}_mesh.xdmf")


num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

# Integration measures
QRULE_BODY, QDEG_BODY = "default", 2  # 2x2 Gauss-Legendre (standard)
if cell_type == quad:
    QRULE_TM, QDEG_TM = "GLL", 1
else:
    QRULE_TM, QDEG_TM = "default", 2
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxVol = dx(BODY_marker, metadata={"quadrature_rule": QRULE_BODY, "quadrature_degree": QDEG_BODY})
dxThird = dx(TM_marker, metadata={"quadrature_rule": QRULE_TM, "quadrature_degree": QDEG_TM})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Create function spaces and functions 
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

element_deg_u = 1
element_deg_theta = 1
V = fem.functionspace(mesh, ("Lagrange", element_deg_u, (tdim,)))
V_theta = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg_theta, (tdim, tdim)))
W = ufl.MixedFunctionSpace(V, V_theta)
V_tm = fem.functionspace(mesh, ("DG", 0)) 

# Functions
u = fem.Function(V, name="displacement")
theta = fem.Function(V_theta, name="theta")

tm_func = fem.Function(V_tm, name="cell_markers")
tm_func.x.array[:] = ct.values

# State and variations
m = [u, theta]
δm = ufl.TestFunctions(W)
 
# Kinematics (2D plane strain)
X = ufl.SpatialCoordinate(mesh)
I = ufl.Identity(tdim)
F_2D = ufl.variable(I + ufl.grad(u))
J = ufl.det(F_2D)
C_2D = F_2D.T * F_2D
I1 = ufl.tr(C_2D) + 1 # add 1 to account for plane strain out-of-plane component

# Material Properties
# body
E = 1.0
nu = 0.4
K = E / (3 * (1 - 2 * nu))  # 5/3
mu = E / (2 * (1 + nu))     # 5/14
K_body = dolfinx.fem.Constant(mesh, K)
mu_body = dolfinx.fem.Constant(mesh, mu)
Psi_body = K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2/3) * I1 - 3)

Pi = (
    Psi_body * dxVol
)

# Third medium
gamma = dolfinx.fem.Constant(mesh, 1.0e-6)
Psi_third = mu_body / 2 * (J ** (-2/3) * I1 - 3) # in 2D plane strain, no need for volumetric term
Pi_third = gamma * Psi_third * dxThird

# regularization
p_theta = dolfinx.fem.Constant(mesh, 5.0e-2)  # TEST: can be smaller, e.g., 1.0e-2  
alpha_r = dolfinx.fem.Constant(mesh, 1.0e-2)

penalty_term = theta - F_2D
Pi_penalty = (
    p_theta / 2 * ufl.inner(penalty_term, penalty_term) 
    ) * dxThird

Pi_reg = (
    gamma * alpha_r / 2 * ufl.inner(ufl.grad(theta), ufl.grad(theta))
    ) * dxThird

Pi_R =  (Pi_penalty + Pi_reg)    
   

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

# Nonlinear problem and solver
residual = ufl.derivative(Pi + Pi_third + Pi_R, m, δm)
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


# # problem = dolfinx.fem.petsc.NonlinearProblem(
# #     forms,
# #     m,
# #     bcs=bcs,
# #     entity_maps=[medium_map],
# #     petsc_options_prefix=name,
# #     petsc_options={
# #         "snes_type": "newtonls",
# #         "snes_linesearch_type": "bt",
# #         # "snes_linesearch_order": 1,
# #         # "snes_linesearch_type": "basic",
# #         "snes_atol": 1e-8,
# #         "snes_rtol": 1e-8,
# #         "snes_max_it": 50,
# #         "snes_monitor": None,
# #         "snes_converged_reason": None,
# #         #"snes_error_if_not_converged": True,
# #         "ksp_type": "preonly",
# #         "pc_type": "lu",
# #         # "pc_type": "cholesky",
# #         "pc_factor_mat_solver_type": "mumps",
# #     },
# # )


# min(J) over the third medium, evaluated at the same quadrature points used for the assembly 
if cell_type == quad:
    rule = basix.QuadratureType.gll
else:
    rule = basix.QuadratureType.default

J_corner_expr = fem.Expression(
    J, basix.make_quadrature(mesh.basix_cell(), QDEG_TM, rule=rule)[0]
)
tm_cells_owned = ct.find(TM_marker)
tm_cells_owned = tm_cells_owned[
    tm_cells_owned < mesh.topology.index_map(tdim).size_local
].astype(np.int32)


def third_medium_min_J():
    """Extract minimum of J over the third medium"""
    u.x.scatter_forward()
    Jc = J_corner_expr.eval(mesh, tm_cells_owned)
    local = float(Jc.min()) if Jc.size else np.inf
    return comm.allreduce(local, op=MPI.MIN)

u_prev = u.x.array.copy()
tm_prev = tm_func.x.array.copy()

# output file for storing results
filename = f"{name}_Q{element_deg_u}.bp" if cell_type == quad else f"{name}_P{element_deg_u}.bp"
ofile = VTXWriter(comm, f"{name}/{filename}", [u, tm_func])
ofile.write(0.0) # write initial state

# Adaptive loading
adaptive_load = True
MAX_FAILURE = 2
dl = 0.05 # initial load increment
dl_min = dl / 16

lmbda = 2.0
v_bar = -0.5*lmbda  # final applied vertical displacement (u_y = -1.0)

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
        n = 0 # reset the number of successive failures for adaptive loading

        # reaction force at the top right corner
        with reaction_vector.localForm() as reaction_local:
            reaction_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(reaction_vector, reaction_form)
        reaction_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        force_y = comm.allreduce(reaction_vector.array[dofs_point_y_owned].sum(), op=MPI.SUM)
        force_array.append(abs(force_y))
        loading_array.append(abs(applied_y.value))
        # Reporting min(J)
        Jmin = third_medium_min_J()

        pprint(f"lambda = {applied_y.value:.3f}, reaction force = {force_y:.6f}, min(J) = {Jmin:.3e}")

        if Jmin <= 0.0:
            pprint(
                f"Third medium inverted: min(J) = {Jmin:.6e} at u_y = {applied_y.value:.3f}. "
                "The solve converged, but onto an inadmissible state -- aborting."
            )
            break

        load += dl
        
        u_prev[:] = u.x.array
        tm_prev[:] = tm_func.x.array
    
    if adaptive_load and n > MAX_FAILURE:
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
