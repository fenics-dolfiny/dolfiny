"""
Third medium contact example from DOI: https://doi.org/10.1016/j.cma.2024.117595
C-Box in 2D plane-strain with HuHu-LuLu regularization
"""

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx import fem

from dolfinx import fem
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc

# For timing the code
from datetime import datetime

# Basic settings
name = "cbox_HuLu_2D"
comm = MPI.COMM_WORLD

# Dimensions
L = 4.0
H = 2.0
T = 0.4
dL = L / 40 # element size 

tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= T - tol) & (x[1] >= T - tol) & (x[1] <= H - T + tol)

def thirdmedium_layer(x):
    return (x[0] >= L - tol)

def top(x):
    return np.isclose(x[1], H)

def left(x):
    return np.isclose(x[0], 0.0)

# Element type
quad = dolfinx.mesh.CellType.quadrilateral

# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0, 0], [L+dL, H]],
    [41, 20],
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
NO_TRACTION_marker = 3
POTENTIAL_CONTACT_marker = 4

mesh.topology.create_connectivity(fdim, tdim) # facets-to-cells connectivity
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)
all_body_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(BODY_marker), tdim, fdim
)
all_air_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(TM_marker), tdim, fdim
)
interface_facets = np.intersect1d(all_body_facets, all_air_facets)
all_exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

facet_markers = np.zeros(num_facets_local, dtype=np.int32)
facet_markers[np.intersect1d(all_exterior_facets, all_body_facets)] = NO_TRACTION_marker
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, left)] = (
    LEFT_marker
)
facet_markers[interface_facets] = POTENTIAL_CONTACT_marker
f_to_c = mesh.topology.connectivity(fdim, tdim)

ft_indices = np.flatnonzero(facet_markers)

ft = dolfinx.mesh.meshtags(
    mesh, fdim, ft_indices, facet_markers[ft_indices]
)
ft.name = "facet_tags"

# Export mesh and markers for inspection
with dolfinx.io.XDMFFile(comm, f"{name}_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")
pprint(f"Mesh saved to {name}_mesh.xdmf")

# Integration measures
metadata = {"quadrature_rule": "GLL", "quadrature_degree": 3}
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
dxVol = dx(BODY_marker)
dxThird = dx(TM_marker)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Create function spaces and functions 
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))
V_tm = fem.functionspace(mesh, ("DG", 0)) 

# Functions
u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

# Kinematics (2D plane strain)
X = ufl.SpatialCoordinate(mesh)
phi = X + u
I = ufl.Identity(tdim)
F_2D = I + ufl.grad(u)
trF = ufl.tr(F_2D)
skF = F_2D[0,1] - F_2D[1,0]
J = ufl.det(F_2D)
C_2D = F_2D.T * F_2D
I1 = ufl.tr(C_2D) + 1 # add 1 to account for plane strain out-of-plane component

## Material Properties
# Body
E = 1.0
nu = 0.4
K = E / (3 * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)
Psi_body = K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2/3) * I1 - 3)

b = dolfinx.fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
t = dolfinx.fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
Pi = (
    Psi_body * dxVol
    - ufl.inner(b, phi) * dxVol
    - ufl.inner(t, phi) * ds((NO_TRACTION_marker))
)

# Third medium
gamma0 = fem.Constant(mesh, 1.0e-7)
Pi_third = gamma0 * Psi_body * dxThird

# regularization
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
Ell = dolfinx.fem.Constant(mesh, np.max(L_i))
alpha = fem.Constant(mesh, 1.0e-06)
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K)

Hu = ufl.grad(ufl.grad(u)) # Hessian of displacement
Lu = ufl.div(ufl.grad(u)) # Laplacian of displacement

HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(I)

Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird


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
residual = ufl.derivative(Pi + Pi_third + Pi_HuLu, u, δu)
# forms = ufl.extract_blocks(residual)

problem = dolfinx.fem.petsc.NonlinearProblem(
    residual,
    u,
    bcs=bcs,
    entity_maps=[medium_map],
    petsc_options_prefix=name,
    petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-6,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_converged_reason": None,
        #"snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)

tm_func = fem.Function(V_tm, name="cell_markers")
tm_func.x.array[:] = ct.values

u_prev = u.x.array.copy()
tm_prev = tm_func.x.array.copy()

# output file for storing results
ofile = VTXWriter(comm, f"{name}_P{element_deg}.bp", [u, tm_func])
ofile.write(0.0) # write initial state

# Adaptive loading
adaptive_load = True
MAX_FAILURE = 5
NUM_SUCCESSIVE_SOLVES = 0


v_bar = -0.6*L  # final applied vertical displacement

num_iterations = 0 # store total number of iterations across all loading steps
loading_steps = 20
load = 1. / loading_steps # load increment
dl = load
n = 0 # used to adaptively increase/decrease load increment 
last_load = load

# print a message for simulation startup
pprint("------------------------------------")
pprint("Simulation Start")
pprint("------------------------------------")
# Store start time 
startTime = datetime.now()

while load <= (1.1):
    
    # Update boundary condition value
    applied_y.value = v_bar * load
    
    pprint(f"\nLoading step: {load:.3f}, u_y: {applied_y.value:.3f}", flush=True)

    # Solve the problem
    problem.solve()
    reason = problem.solver.getConvergedReason()
    
    num_iterations += problem.solver.getIterationNumber()
    n += 1
    
    if reason < 0:
        if adaptive_load:
            load = last_load + dl/(n+1)
            u.x.array[:] = u_prev.copy()
            tm_func.x.array[:] = tm_prev.copy()
            NUM_SUCCESSIVE_SOLVES = 0
        else:
            pprint("Solver failed to converge, aborting.")
            break
    
    else:
        n = 0
        last_load = load
        NUM_SUCCESSIVE_SOLVES += 1
        ofile.write(load)
        
        load += dl
        # if adaptive_load:
        #     # load += NUM_SUCCESSIVE_SOLVES * dl
        #     load += 2 * dl # double load increment after successful solve
        
        u_prev[:] = u.x.array.copy()
        tm_prev[:] = tm_func.x.array.copy()
    
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

