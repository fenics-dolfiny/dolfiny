"""
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
Third medium contact example from DOI: 10.1007/s00466-025-02628-y
"""

from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from dolfinx.io import VTXWriter

from dolfinx import fem
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc

# For timing the code
from datetime import datetime

# Basic settings
name = "cbox_W_3D"
comm = MPI.COMM_WORLD

tol = 1.0e-6

def thirdmedium(x):
    return (x[0] >= 0.1 - tol) & (x[2] >= 0.1 - tol) & (x[2] <= 0.6 + tol)


def top(x):
    return np.isclose(x[2], 0.7)


def left(x):
    return np.isclose(x[0], 0.0)


# Element type
tet = dolfinx.mesh.CellType.tetrahedron
hex = dolfinx.mesh.CellType.hexahedron

mesh = dolfinx.mesh.create_box(
    comm,
    [[0, 0, 0], [2, 0.7, 0.7]],
    [40, 14, 14],
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    cell_type=hex,
)
tdim = mesh.topology.dim # 3
fdim = tdim - 1 # 2

# Tag cells as either body or third medium
BODY_marker = 1
TM_marker = 2

num_cells_local = (
    mesh.topology.index_map(tdim).size_local
    + mesh.topology.index_map(tdim).num_ghosts
)
markers = np.full(num_cells_local, BODY_marker, dtype=np.int32)
markers[dolfinx.mesh.locate_entities(mesh, tdim, thirdmedium)] = TM_marker
ct = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells_local), markers)

# Tag facets
LEFT_marker = 2
NO_TRACTION_marker = 3
POTENTIAL_CONTACT_marker = 4

mesh.topology.create_connectivity(fdim, tdim)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)
all_body_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(BODY_marker), tdim, fdim
)
all_tm_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(TM_marker), tdim, fdim
)
interface_facets = np.intersect1d(all_body_facets, all_tm_facets)
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
ft.name = "facet_markers"

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
metadata = {"quadrature_degree": 2} # should be equivalent to 2x2x2 Gauss integration for H1
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
dxVol = dx(BODY_marker)
dxThird = dx(TM_marker)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)


# Define function spaces and functions
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(TM_marker)
)[0:2]

V = fem.functionspace(mesh, ("Lagrange", 1, (tdim,))) # H1 element for displacement
P = fem.functionspace(third_medium_mesh, ("Lagrange", 1, (3,))) # same approx for p_i variables
W = ufl.MixedFunctionSpace(V, P)
V_tm = fem.functionspace(mesh, ("DG", 0)) # for storing tm cell markers

u = fem.Function(V, name="displacement")
p = fem.Function(P) # variables to allow linear ansatz

# Define state and test functions
m = [u, p]
δm = ufl.TestFunctions(W)

# Kinematics 
I = ufl.Identity(len(u))
x = ufl.SpatialCoordinate(mesh)
phi = x + u
F = I + ufl.grad(u)
J = ufl.det(F)
C = F.T * F


## Material Properties
# Body
E = 5
nu = 0.3
K = E / (3 * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
K_body = fem.Constant(mesh, K)
mu_body = fem.Constant(mesh, mu)
Psi_body = K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2 / 3) * ufl.tr(C) - 3)


b = fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
t = fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
Pi = (
    Psi_body * dxVol
    - ufl.inner(b, phi) * dxVol
    - ufl.inner(t, phi) * ds((NO_TRACTION_marker))
)


# Third medium
# bulk contribution
gamma = fem.Constant(mesh, 2.0e-6)
Pi_third = gamma * Psi_body * dxThird

# regularization
beta_1 = fem.Constant(mesh, 1.0)
beta_2 = fem.Constant(mesh, 100.0)
f0_skew = 0.5 * (F[0, 1] - F[1, 0])
f1_skew = 0.5 * (F[0, 2] - F[2, 0])
f2_skew = 0.5 * (F[1, 2] - F[2, 1])
fi_skew = ufl.as_vector((f0_skew, f1_skew, f2_skew))
L_i = np.zeros(3)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
d = fem.Constant(mesh, np.max(L_i))

skew_term = fi_skew - 1 / d * p
Pi_fi = (
    beta_1 / 2  * ufl.dot(skew_term, skew_term) + beta_2 / 2 * ufl.inner(ufl.grad(p), ufl.grad(p))
) * dxThird


## Boundary conditions
# Clamped side
left_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(LEFT_marker)
)
bc_left = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), left_dofs, V
)

# Point displacement
mesh.topology.create_connectivity(0, tdim)
node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 2.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.7))
dofs_point_z = fem.locate_dofs_topological(V.sub(2), 0, node)
dofs_point_y = fem.locate_dofs_topological(V.sub(1), 0, node)
applied_z = fem.Constant(mesh, 0.0)
applied_y = fem.Constant(mesh, 0.0)
bc_point_z = fem.dirichletbc(applied_z, dofs_point_z, V.sub(2))
bc_point_y = fem.dirichletbc(applied_y, dofs_point_y, V.sub(1))
bcs = [bc_left, bc_point_z, bc_point_y]


residual = ufl.derivative(Pi + Pi_third + Pi_fi, m, δm)
forms = ufl.extract_blocks(residual)

# problem = dolfiny.snesproblem.SNESProblem(
#     forms,
#     m,
#     prefix=name,
#     bcs=bcs,
# )

# # Configure PETSc solver options
# opts = PETSc.Options(name)

# opts["snes_type"] = "newtonls"
# opts["snes_linesearch_type"] = "bt"
# opts["snes_atol"] = 1.0e-06
# opts["snes_rtol"] = 1.0e-06
# opts["snes_max_it"] = 25
# opts["snes_monitor"] = None

# opts["ksp_type"] = "preonly"
# opts["pc_type"] = "lu"
# opts["pc_factor_mat_solver_type"] = "mumps"


problem = NonlinearProblem(
    forms,
    m,
    bcs=bcs,
    entity_maps=[medium_map],
    petsc_options_prefix=name,
    petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-6,
        "snes_max_it": 25,
        "snes_monitor": None,
        "snes_converged_reason": None,
        # "snes_error_if_not_converged": True,
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
ofile = VTXWriter(comm, f"{name}_line.bp", [u, tm_func])
ofile.write(0.0) # write initial state

# Adaptive loading
adaptive_load = False
MAX_FAILURE = 5
NUM_SUCCESSIVE_SOLVES = 0

# full_disp = np.array([0.0, 0.25, -1.0]) # Point contact
full_disp = np.array([0.0, 0.5, -2.0]) # Line contact

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

while load <= 1:
    
    # Update boundary condition values
    applied_z.value = full_disp[2] * load
    applied_y.value = full_disp[1] * load
    
    pprint(f"\nLoading step: {load:.3f}, u_z: {applied_z.value:.3f}, u_y: {applied_y.value:.3f}", flush=True)

    # Solve the problem
    problem.solve()
    reason = problem.solver.getConvergedReason()
    
    num_iterations += problem.solver.getIterationNumber()
    n += 1
    
    if reason < 0:
        if adaptive_load:
            # load = last_load + dl/(n+1) 
            load = last_load + dl/2 # half load increment
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
        if adaptive_load:
        #     # load += NUM_SUCCESSIVE_SOLVES * dl
            load += 2 * dl # double load increment after successful solve
        
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
