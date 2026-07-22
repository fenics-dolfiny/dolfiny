"""
Third-medium approach applied to double-arch benchmark.
Wriggers' regularization, see https://doi.org/10.1016/j.cma.2025.117740
"""

from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from dolfinx.io import VTXWriter
from double_arch_gmsh_NEW import mesh_double_arch_gmsh  # external-only full-box TM 
# from double_arch_gmsh import mesh_double_arch_gmsh

from dolfinx import fem
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc

# For timing the code
from datetime import datetime
import argparse
import os

# Basic settings
name = "double_arch_W"
comm = MPI.COMM_WORLD

# User parameters (argparse)
parser = argparse.ArgumentParser(
    description="Double-arch third-medium contact, Wriggers first-order regularization."
)
parser.add_argument("--gamma", type=float, default=1.0e-4,
                    help="relative contact stiffness gamma multiplying the TM energy")
parser.add_argument("--beta1", type=float, default=1.0e5,
                    help="penalty enforcing skew(F)/tr(F) ~ p/reg_len")
parser.add_argument("--beta2", type=float, default=10.0,
                    help="penalty enforcing J ~ q (strength of the grad(J) control)")
parser.add_argument("--alpha_r", type=float, default=100.0,
                    help="regularization scaling on grad(p), grad(q)")
parser.add_argument("--nTM", type=int, default=10,
                    help="number of elements across the third medium")
parser.add_argument("--full_disp", type=float, default=-90.0,
                    help="full prescribed vertical displacement [mm] on top of arches")
parser.add_argument("--quad_degree", type=int, default=3,
                    help="quadrature degree (standard Gauss)")
parser.add_argument("--exp_tag", type=str, default="",
                    help="optional tag appended to the output folder name")
args, _ = parser.parse_known_args() # allow unknown args to be passed without throwing an error 

# Geometry and mesh parameters
Lx, H, Lz = 260., 50., 50. # block dimensions
Di, t = 90., 5. # inner diameter and thickness of arches
g0 = 20. # initial vertical gap between arches and block

# define number of elements
nL, nH, nt = 24, 10, 2
nDi = 12
nTM = args.nTM

# arch1: INNER, arch2: OUTER
cell_tags = {"block": 1, "arch1": 2, "arch2": 3, "tm": 4}
facet_tags = {"bottom": 1, "top_arches": 2}

verbosity = 1
vtk_file = True # export mesh to Paraview for visualization

dim = 2 # solve 3D problem or 2D plane-strain problem
name = f"{name}_{dim}D"
if args.exp_tag:
    name = f"{name}_{args.exp_tag}"
os.makedirs(name, exist_ok=True) # ensure output folder exists (needed for tagged experiments)

model = mesh_double_arch_gmsh(
    tdim=dim, cell_tags=cell_tags, facet_tags=facet_tags, Lx=Lx, Lz=Lz, H=H, Di=Di, t=t, g0=g0,
    nL=nL, nH=nH, nt=nt, nDi=nDi, nTM=nTM,
    verbosity=verbosity, comm=comm, name=name, vtk_file=vtk_file)

# Extract mesh data for Dolfinx
mesh_data = dolfinx.io.gmsh.model_to_mesh(model, comm, rank=0, gdim=dim)

mesh = mesh_data.mesh
ct = mesh_data.cell_tags
ft = mesh_data.facet_tags

tdim = mesh.topology.dim 
fdim = tdim - 1 

num_cells_local = (
    mesh.topology.index_map(tdim).size_local
    + mesh.topology.index_map(tdim).num_ghosts
)

mesh.topology.create_connectivity(fdim, tdim)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local
    + mesh.topology.index_map(fdim).num_ghosts
)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)

pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

# Integration measures
metadata = {"quadrature_degree": args.quad_degree}  # standard Gauss quadrature
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
# dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxA1 = dx(cell_tags["arch1"])  # integration measure for arch1
dxA2 = dx(cell_tags["arch2"])  # integration measure for arch2
dxThird = dx(cell_tags["tm"])  # integration measure for third medium
dxBlock = dx(cell_tags["block"]) # integration measure for block
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Define third-medium submesh and mapping
third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(cell_tags["tm"])
)[0:2]

# Define function spaces and functions
element_deg = 1
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,))) 

if dim == 3:
    P = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg, (dim,)))
else:
    P = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg))

Q = fem.functionspace(third_medium_mesh, ("Lagrange", element_deg))
W = ufl.MixedFunctionSpace(V, P, Q)
V_tm = fem.functionspace(mesh, ("DG", 0)) # for storing tm cell markers

u = fem.Function(V, name="displacement")
p = fem.Function(P)
q = fem.Function(Q)

# Define state and state variation
m = [u, p, q]
δm = ufl.TestFunctions(W)

# Kinematics -- 2D plane strain or 3D
if dim == 3:
    I = ufl.Identity(len(u))
    X = ufl.SpatialCoordinate(mesh)
    F = ufl.variable(I + ufl.grad(u))
    trF = ufl.tr(F)
    f0_skew = 0.5 * (F[0, 1] - F[1, 0])
    f1_skew = 0.5 * (F[0, 2] - F[2, 0])
    f2_skew = 0.5 * (F[1, 2] - F[2, 1])
    skF = ufl.as_vector((f0_skew, f1_skew, f2_skew))
    J = ufl.det(F)
    C = ufl.variable(F.T * F)
    I1 = ufl.tr(C)
    I3 = J**2
else:
    I = ufl.Identity(len(u))
    X = ufl.SpatialCoordinate(mesh)
    F_2D = ufl.variable(I + ufl.grad(u))
    trF = ufl.tr(F_2D)
    skF = F_2D[0,1] - F_2D[1,0]
    C_2D = ufl.variable(F_2D.T * F_2D)
    I1 = ufl.tr(C_2D) + 1
    J = ufl.det(F_2D)
    I3 = J**2

## Material Properties
nu = 0.3
E_a1 = 1e5  # INNER arch Young's modulus [MPa]
E_a2 = 1e3  # OUTER arch Young's modulus [MPa]
E_block = 300. # block Young's modulus [MPa]

mu_a1 = fem.Constant(mesh, E_a1 / (2 * (1 + nu)))
lam_a1 = fem.Constant(mesh, E_a1 * nu / ((1 + nu) * (1 - 2 * nu)))

mu_a2 = fem.Constant(mesh, E_a2 / (2 * (1 + nu)))
lam_a2 = fem.Constant(mesh, E_a2 * nu / ((1 + nu) * (1 - 2 * nu)))

mu_block = fem.Constant(mesh, E_block / (2 * (1 + nu)))
lam_block = fem.Constant(mesh, E_block * nu / ((1 + nu) * (1 - 2 * nu)))

# Strain energy density for the different materials -- compressible Neo-Hookean model
Psi_a1 = mu_a1 / 2 * (I1 - 3) - mu_a1 * ufl.ln(J) + lam_a1 / 2 * ufl.ln(J)**2 
Psi_a2 = mu_a2 / 2 * (I1 - 3) - mu_a2 * ufl.ln(J) + lam_a2 / 2 * ufl.ln(J)**2
Psi_block = mu_block / 2 * (I1 - 3) - mu_block * ufl.ln(J) + lam_block / 2 * ufl.ln(J)**2
# Potential energy contributions for the different materials
Pi = (
    Psi_a1 * dxA1
    + Psi_a2 * dxA2
    + Psi_block * dxBlock
    )

# Third medium 
# fictitious material properties for third medium as the softer of the bodies (here, the block)
K_TM = E_block / (3 * (1 - 2 * nu))
K_tm = fem.Constant(mesh, K_TM)
mu_tm = mu_block
TM_VOLUMETRIC = True
if dim == 3:
    Psi_tm = K_tm / 2 * ufl.ln(J) ** 2 + mu_tm / 2 * (J ** (-2/3) * I1 - 3)
else: # in 2D plane-strain, the deviatoric contribution is sufficient to guarantee third-medium stiffnening under compression
    Psi_tm = mu_tm / 2 * (J ** (-2/3) * I1 - 3)
    if TM_VOLUMETRIC: # restore the ln(J)^2 volumetric barrier 
        Psi_tm = Psi_tm + K_tm / 2 * ufl.ln(J) ** 2

gamma = fem.Constant(mesh, args.gamma)
Pi_third = gamma * Psi_tm * dxThird

# regularization
L_i = np.zeros(tdim)
for dim in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, dim].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, dim].min(), op=MPI.MIN)
    L_i[dim] = x_i_max - x_i_min
Ell = dolfinx.fem.Constant(mesh, np.max(L_i))

beta_1 = dolfinx.fem.Constant(mesh, args.beta1)
beta_2 = dolfinx.fem.Constant(mesh, args.beta2)
alpha_r = dolfinx.fem.Constant(mesh, args.alpha_r)

skew_term = (skF / trF) - 1/Ell * p

Pi_grad = (
    beta_1 * skew_term**2 + alpha_r * ufl.inner(ufl.grad(p), ufl.grad(p))
    ) * dxThird

Pi_J = (
    beta_2 * (J - q)**2 + alpha_r * ufl.inner(ufl.grad(q), ufl.grad(q))
    ) * dxThird

Pi_R = gamma/2 * (Pi_grad + Pi_J)  

## Boundary conditions
# Bottom fixed
bottom_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(facet_tags["bottom"])
)
bc_bottom = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), bottom_dofs, V
)

# Apply bcs to top of arches (fix x,z directions, allow y-direction displacement)
top_arches_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, ft.find(facet_tags["top_arches"])
)

applied_disp = dolfinx.fem.Constant(mesh, (0.,) * tdim)

bc_top_arches = dolfinx.fem.dirichletbc(
    applied_disp, top_arches_dofs, V
)

bcs = [bc_bottom, bc_top_arches]

# Define the residual and forms for the nonlinear problem
residual = ufl.derivative(Pi + Pi_third + Pi_R, m, δm)
forms = ufl.extract_blocks(residual)

# opts = PETSc.Options(name)
# opts["snes_type"] = "newtonls"
# opts["snes_linesearch_type"] = "bt"
# opts["snes_linesearch_order"] = 1
# opts["snes_rtol"] = 1.0e-06
# opts["snes_max_it"] = 50
# opts["ksp_type"] = "preonly"
# opts["pc_type"] = "cholesky"
# opts["pc_factor_mat_solver_type"] = "mumps"

# problem = dolfiny.snesproblem.SNESProblem(
#     forms,
#     m,
#     bcs=bcs,
#     prefix=name,
#     entity_maps=[medium_map],
# )

problem = dolfinx.fem.petsc.NonlinearProblem(
    forms,
    m,
    bcs=bcs,
    entity_maps=[medium_map],
    petsc_options_prefix=name,
    petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 1,
        # "snes_linesearch_monitor": None,
        "snes_rtol": 1.0e-08,
        "snes_atol": 1.0e-08,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "preonly",
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": "mumps",
    },
)

tm_func = fem.Function(V_tm, name="cell_markers")
tm_func.x.array[:] = ct.values

# J = det(F) diagnostic on a DG-0 space (Linear elements + standard Gauss quadrature): one value per cell 
# (element-wise averaged det(F) at the interpolation point). 
# per-cell J the natural monitor of the third-medium volume collapse (min(J) -> 0).
V_J = fem.functionspace(mesh, ("DG", 0))
J_func = fem.Function(V_J, name="J_det")
J_expr = fem.Expression(J, V_J.element.interpolation_points)
tm_cells = ct.find(cell_tags["tm"])
tm_J_dofs = np.unique(V_J.dofmap.list[tm_cells].reshape(-1))


def report_min_J():
    """Interpolate J, report min over TM and how many TM cells are near/below collapse."""
    J_func.interpolate(J_expr)
    J_tm = J_func.x.array[tm_J_dofs]
    min_J = float(J_tm.min())
    n_low = int((J_tm < 0.1).sum())
    n_neg = int((J_tm <= 0.0).sum())
    pprint(
        f"    min(J)|TM = {min_J:+.4e}   #cells J<0.1 = {n_low}/{len(tm_J_dofs)}   #cells J<=0 = {n_neg}",
        flush=True,
    )
    return min_J

u_prev = u.x.array.copy()
tm_prev = tm_func.x.array.copy()

# output file for storing results
ofile = VTXWriter(comm, f"{name}/{name}.bp", [u, tm_func])
ofile.write(0.0) # write initial state
# separate writer for J (DG-0): VTX requires one element type per file
ofile_J = VTXWriter(comm, f"{name}/{name}_J.bp", [J_func])
ofile_J.write(0.0)

# identify half-ring middle node 
mesh.topology.create_connectivity(0, tdim) # nodes-to-cells connectivity
m_node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], g0+t))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, m_node)

# Adaptive loading
adaptive_load = False
MAX_FAILURE = 2
NUM_SUCCESSIVE_SOLVES = 0

full_disp = args.full_disp # [mm] full displacement applied to top of arches
pre_contact_disp_step = 1.0
contact_disp_step = 0.5
threshold = 20.0 # up to this displacement, use larger load increment (even if contact is established)
dl_free = pre_contact_disp_step / abs(full_disp)
dl_contact = contact_disp_step / abs(full_disp)

num_iterations = 0 # store total number of iterations across all loading steps
load = dl_free # first increment before contact
dl = dl_free
n = 0 # used to adaptively increase/decrease load increment 
last_load = load
ii = 1 # counter for steps with successful convergence 

loading_history = np.empty(0, dtype=np.float64)
m_node_displ = np.empty(0, dtype=np.float64)
history_file = f"{name}/loading_history.npz"
ii_load = 1 # counter for loading history storage


def save_loading_history():
    np.savez(
        history_file,
        loading_history=loading_history,
        m_node_displ=m_node_displ,
    )

# print a message for simulation startup
pprint("------------------------------------")
pprint("Simulation Start")
pprint("------------------------------------")
# Store start time 
startTime = datetime.now()

while load <= (1.0 + 1e-6):
    
    # Update boundary condition value along y-direction
    applied_disp.value[1] = full_disp * load
    
    pprint(f"\n Load step {ii}: u_y = {applied_disp.value[1]:.3f}", flush=True)

    # Solve the problem
    problem.solve()
    reason = problem.solver.getConvergedReason()

    report_min_J()  # diagnostic: min(J) over TM (on the final iterate, converged or not)

    num_iterations += problem.solver.getIterationNumber()
    n += 1
    
    if reason < 0:
        if adaptive_load:
            # half the load increment after a failed solve
            dl = dl / 2.0
            load = last_load + dl
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
        ofile_J.write(load)


        current_disp = abs(full_disp) * load
        if current_disp < threshold:
            dl = dl_free
        else:
            dl = min(dl, dl_contact)

        if current_disp >= g0: # contact established, store loading history
            loading_history = np.append(loading_history, ii_load) 
            m_node_displ = np.append(m_node_displ, np.asarray(u.x.array[dofs_point_y]).reshape(-1)[0])
            save_loading_history()
            ii_load += 1

        load += dl
        ii += 1
        
        u_prev[:] = u.x.array.copy()
        tm_prev[:] = tm_func.x.array.copy()
    
    if adaptive_load and n > MAX_FAILURE:
        pprint("Too many failures, aborting.")
        break

ofile.close() # close output file
ofile_J.close()

# # persist the last converged state for post-processing
save_loading_history()

# Store end time and compute elapsed time
endTime = datetime.now()
elapsedTime = endTime - startTime

pprint("-----------------------------------------")
pprint("End computation") 
pprint(f"Elapsed time: {elapsedTime}")
pprint(f"Total number of iterations: {num_iterations}")


