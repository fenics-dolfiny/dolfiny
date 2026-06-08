"""
Third-medium approach applied to 3D Hertz contact problem
Inspired by the work of Wriggers and co-workers, see https://doi.org/10.1016/j.jmps.2026.106617
"""

from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from dolfinx.io import VTXWriter
from mesh_Hertz3D_gmsh import mesh_Hertz3D_gmsh

from dolfinx import fem
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc

# For timing the code
from datetime import datetime

# Basic settings
name = "hertz_W_3D"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
# 1/4 half space dimensions
W, L, H = 2.0, 2.0, 1.0
# third medium dimensions
g0 = 0.1
H2 = H + g0
# sphere radius
R = 1.0

cell_tags = {"body": 1, "indenter": 2, "tm": 3}
facet_tags = {"sphere_top": 10, "bottom": 11, "x_symm": 12, "y_symm": 13, "body_top": 14}

verbosity = 1

mesh_data = mesh_Hertz3D_gmsh(cell_tags, facet_tags, W=W, L=L, H=H, H2=H2, R=R, verbosity=verbosity)
mesh = mesh_data.mesh
ct = mesh_data.cell_tags
ft = mesh_data.facet_tags

tdim = mesh.topology.dim # 3
fdim = tdim - 1 # 2


num_cells_local = (
    mesh.topology.index_map(tdim).size_local
    + mesh.topology.index_map(tdim).num_ghosts
)

mesh.topology.create_connectivity(fdim, tdim)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)

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
metadata = {"quadrature_degree": 1} # one quadrature point for TET1 elements
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
dxVol = dx(cell_tags["body"])
dxThird = dx(cell_tags["tm"])
dxInd = dx(cell_tags["indenter"])
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)


# Define function spaces and functions
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(cell_tags["tm"])
)[0:2]

V = fem.functionspace(mesh, ("Lagrange", 1, (tdim,))) # TET1 element for displacement
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

b = fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
t = fem.Constant(
    mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
)
Pi = (
    Psi_body * dxVol
    - ufl.inner(b, phi) * dxVol)

# Third medium
# bulk contribution
gamma = fem.Constant(mesh, 1.0e-6)
Pi_third = gamma * Psi_body * dxThird

# Indenter
K2 = 100 * K1
mu2 = 100 * mu1
K_ind = fem.Constant(mesh, K2)
mu_ind = fem.Constant(mesh, mu2)
Psi_ind = K_ind / 2 * ufl.ln(J) ** 2 + mu_ind / 2 * (J ** (-2 / 3) * ufl.tr(C) - 3)
Pi_indenter = Psi_ind * dxInd

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
# Bottom fixed
bottom_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(facet_tags["bottom"])
)
bc_bottom = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), bottom_dofs, V
)

# Symmetry planes
x_symm_dofs = dolfinx.fem.locate_dofs_topological(
    V.sub(0), fdim, ft.find(facet_tags["x_symm"])
)
y_symm_dofs = dolfinx.fem.locate_dofs_topological(
    V.sub(1), fdim, ft.find(facet_tags["y_symm"])
)

bc_x_symm = dolfinx.fem.dirichletbc(
    np.array(0.0, dtype=dolfinx.default_scalar_type), x_symm_dofs, V.sub(0)
)
bc_y_symm = dolfinx.fem.dirichletbc(
    np.array(0.0, dtype=dolfinx.default_scalar_type), y_symm_dofs, V.sub(1)
)

# Applied displacement top of sphere
sphere_top_z_dof = dolfinx.fem.locate_dofs_topological(
    V.sub(2), fdim, ft.find(facet_tags["sphere_top"])
)

applied_z = fem.Constant(mesh, 0.0)
bc_sphere_z = dolfinx.fem.dirichletbc(applied_z, sphere_top_z_dof, V.sub(2))

bcs = [bc_bottom, bc_x_symm, bc_y_symm, bc_sphere_z]

residual = ufl.derivative(Pi + Pi_third + Pi_indenter + Pi_fi, m, δm)
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
ofile = VTXWriter(comm, f"{name}.bp", [u, tm_func])
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

# print a message for simulation startup
pprint("------------------------------------")
pprint("Simulation Start")
pprint("------------------------------------")
# Store start time 
startTime = datetime.now()

while load <= 1:
    
    # Update boundary condition values
    applied_z.value = full_disp * load
    
    pprint(f"\nStep {ii}/{loading_steps}: u_z = {applied_z.value:.3f}", flush=True)

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
        ii += 1
        
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
