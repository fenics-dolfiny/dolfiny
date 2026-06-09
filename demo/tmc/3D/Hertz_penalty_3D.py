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
name = "hertz_pen_3D"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
# 1/4 half space dimensions
W, L, H = 2.0, 2.0, 1.0
# sphere radius
R = 1.0

mesh = dolfinx.mesh.create_box(comm, [[0.0, 0.0, 0.0], [W, L, H]],
                            [10, 10, 5], cell_type=dolfinx.mesh.CellType.tetrahedron)

tdim = mesh.topology.dim # 3
fdim = tdim - 1 # 2
N = ufl.FacetNormal(mesh)

def bottom(x):
    return np.isclose(x[2], 0.0)

def x_symm(x):
    return np.isclose(x[0], 0.0)

def y_symm(x):
    return np.isclose(x[1], 0.0)

def top(x):
    return np.isclose(x[2], H)

# Tag facets for boundary conditions
BOTTOM_marker = 1
X_SYMM_marker = 2
Y_SYMM_marker = 3
TOP_marker = 4

mesh.topology.create_connectivity(fdim, tdim)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)

facet_markers = np.zeros(num_facets_local, dtype=np.int32)
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, bottom)] = BOTTOM_marker
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, x_symm)] = X_SYMM_marker
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, y_symm)] = Y_SYMM_marker
facet_markers[dolfinx.mesh.locate_entities(mesh, fdim, top)] = TOP_marker

f_to_c = mesh.topology.connectivity(fdim, tdim)

ft_indices = np.flatnonzero(facet_markers)

ft = dolfinx.mesh.meshtags(
    mesh, fdim, ft_indices, facet_markers[ft_indices]
)
ft.name = "facet_markers"

# Export mesh and markers for inspection
with dolfinx.io.XDMFFile(comm, f"{name}/{name}_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")
pprint(f"Mesh saved to {name}/{name}_mesh.xdmf")

# Integration measures
metadata = {"quadrature_degree": 1} # one quadrature point for TET1 elements
dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Define function spaces and functions
V = fem.functionspace(mesh, ("Lagrange", 1, (tdim,))) # TET1 element for displacement

u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)

# Kinematics 
I = ufl.Identity(len(u))
X = ufl.SpatialCoordinate(mesh) # reference coordinates
x = X + u # current coordinates
F = ufl.variable(I + ufl.grad(u))
J = ufl.det(F)
C = ufl.variable(F.T * F)

## Material Properties
# Body
K1 = 5/4
mu1 = 5/14
K_body = fem.Constant(mesh, K1)
mu_body = fem.Constant(mesh, mu1)
Psi_body = K_body / 2 * ufl.ln(J) ** 2 + mu_body / 2 * (J ** (-2 / 3) * ufl.tr(C) - 3)
PK2 = 2 * ufl.diff(Psi_body, C)
PK1 = F * PK2

Pi_elastic = Psi_body * dx

# Contact energy with penalty regularization
def ppos(x):
    return (x + abs(x)) / 2

g0 = 0.1  # initial gap between sphere and half-space
applied_disp = fem.Constant(mesh, 0.0) # applied displacement, will be updated in loading loop

x_ind = ufl.as_vector([0, 0, H + g0 + R + applied_disp])

penetration = ppos(R**2 - (x - x_ind)**2)
k_pen = fem.Constant(mesh, 1e3) # penalty parameter

Pi_penalty = k_pen / 2 * penetration**2 * ds(TOP_marker)

# Total potential energy
Pi = Pi_elastic + Pi_penalty
# Pi = Pi_elastic 

## Boundary conditions
# Bottom fixed
bottom_facets = dolfinx.mesh.locate_entities(mesh, fdim, bottom)
bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
bc_bottom = fem.dirichletbc(np.zeros(tdim, dtype=dolfinx.default_scalar_type), bottom_dofs, V)

# Symmetry planes
x_symm_facets = dolfinx.mesh.locate_entities(mesh, fdim, x_symm)
x_symm_dofs = fem.locate_dofs_topological(V.sub(0), fdim, x_symm_facets)
bc_x_symm = fem.dirichletbc(0.0, x_symm_dofs, V.sub(0)) # u_x = 0 on x-symmetry

y_symm_facets = dolfinx.mesh.locate_entities(mesh, fdim, y_symm)
y_symm_dofs = fem.locate_dofs_topological(V.sub(1), fdim, y_symm_facets)
bc_y_symm = fem.dirichletbc(0.0, y_symm_dofs, V.sub(1)) # u_y = 0 on y-symmetry 

bcs = [bc_bottom, bc_x_symm, bc_y_symm]

# Define the nonlinear problem and solver
residual = ufl.derivative(Pi, u, v)

petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": 1e-9,
        "snes_rtol": 1e-9,
        "snes_max_it": 25,
        "snes_monitor": None,
        "snes_converged_reason": None,
        # "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

problem = NonlinearProblem(
    residual,
    u,
    bcs=bcs,
    petsc_options_prefix=name,
    petsc_options=petsc_options,
)

# output file for storing results
ofile = VTXWriter(comm, f"{name}/{name}.bp", [u])
ofile.write(0.0) # write initial state

# Adaptive loading
adaptive_load = False
MAX_FAILURE = 5
NUM_SUCCESSIVE_SOLVES = 0


full_disp = -(g0 + 0.5)  # u_z = -1.0 at full load

num_iterations = 0 # store total number of iterations across all loading steps
loading_steps = 60
load = 1. / loading_steps # load increment
dl = load
n = 0 # used to adaptively increase/decrease load increment 
last_load = load
ii = 1 # counter for steps with successful convergence 
u_prev = u.x.array.copy() # store previous solution for adaptive loading

# print a message for simulation startup
pprint("------------------------------------")
pprint("Simulation Start")
pprint("------------------------------------")
# Store start time 
startTime = datetime.now()

while load <= (1.0 + 1e-6):
    
    # Update boundary condition values
    applied_disp.value = full_disp * load
    
    pprint(f"\nLoad step {ii}: u_z = {applied_disp.value:.3f}", flush=True)

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
            NUM_SUCCESSIVE_SOLVES = 0
        else:
            pprint("Solver failed to converge, aborting.")
            break
    
    else:
        n = 0
        last_load = load
        NUM_SUCCESSIVE_SOLVES += 1
        ofile.write(load)
        ii += 1

        load += dl
        # if adaptive_load:
        # #     # load += NUM_SUCCESSIVE_SOLVES * dl
        #     load += 2 * dl # double load increment after successful solve
        
        u_prev[:] = u.x.array.copy()
    
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

