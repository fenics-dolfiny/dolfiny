"""
Third-medium approach applied to double-arch benchmark.
HuHu/HuHu-LuLu regularization, see https://doi.org/10.1016/j.cma.2024.117595
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
from dolfinx import geometry
import dolfiny
from dolfiny.utils import pprint
from petsc4py import PETSc
# print(PETSc.Sys.getVersion())

import os
import argparse

# For timing the code
from datetime import datetime

# Basic settings
name = "double_arch_HL"
comm = MPI.COMM_WORLD

# User parameters (argparse)
parser = argparse.ArgumentParser(
    description="Double-arch third-medium contact, HuHu-LuLu regularization."
)
parser.add_argument("--gamma", type=float, default=1.0e-4,
                    help="relative contact stiffness multiplying the TM energy")
parser.add_argument("--alpha_r", type=float, default=1.0e-4,
                    help="regularization scaling")
parser.add_argument("--nTM", type=int, default=10,
                    help="number of elements across the third medium")
parser.add_argument("--full_disp", type=float, default=-90.0,
                    help="full prescribed vertical displacement [mm] on top of arches")
parser.add_argument("--quad_degree", type=int, default=3,
                    help="quadrature degree (Gauss-Lobatto integration scheme)")
parser.add_argument("--exp_tag", type=str, default="",
                    help="optional tag appended to the output folder name")
parser.add_argument("--diagnostic", action="store_true", default=False,
                    help="report diagnostic info: min(J)|TM, max(J)|TM, slip/gap at interface, etc.")
parser.add_argument("--reg_scaling", action="store_true", default=False,
                    help="apply the Frederiksen exp(-5*det(F)) ad-hoc scaling to the HuHu-LuLu "
                         "regularization (Eq. 6); breaks tangent symmetry -> uses LU not Cholesky")
args, _ = parser.parse_known_args() # allow unknown args to be passed without throwing an error 

# Geometry and mesh parameters
Lx, H, Lz = 260., 50., 50. # block dimensions
Di, t = 90., 5. # inner diameter and thickness of arches
lR = 10. # length of rectangular region in third-medium (for structured mesh)
g0 = 20. # initial vertical gap between arches and block

# define number of elements (see mesh_double_arch_gmsh() for details)
nL, nH, nt = 24, 10, 1
nDi = 12
nTM = args.nTM

# finer mesh
# nL, nH, nt = 44, 10, 2
# nDi = 22
# nTM = 20

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
    tdim=dim, cell_tags=cell_tags, facet_tags=facet_tags, Lx=Lx, Lz=Lz, H=H, Di=Di, t=t, g0=g0, lR=lR,
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
metadata = {"quadrature_rule": "GLL", "quadrature_degree": args.quad_degree}
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
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
element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))
V_tm = fem.functionspace(mesh, ("DG", 0)) # for storing tm cell markers
# DG-1 space to diagnose the Jacobian J = det(F): its 4 corner dofs/quad capture the
# corner-node inversion (J -> 0) that GLL quadrature is meant to mitigate.
V_J = fem.functionspace(mesh, ("DG", 1))
J_func = fem.Function(V_J, name="J_det")

u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

# Define state and state variation
m = [u]
δm = δu

# Kinematics -- 2D plane strain or 3D
if dim == 3:
    I = ufl.Identity(len(u))
    X = ufl.SpatialCoordinate(mesh)
    F = ufl.variable(I + ufl.grad(u))
    J = ufl.det(F)
    C = ufl.variable(F.T * F)
    I1 = ufl.tr(C)
    I3 = J**2
else:
    I = ufl.Identity(len(u))
    X = ufl.SpatialCoordinate(mesh)
    F_2D = ufl.variable(I + ufl.grad(u))
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

# Strain energy density for the different materials -- compressible Neo-Hookean model (Simo-Pister form)
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
alpha = fem.Constant(mesh, args.alpha_r)
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K_TM)

Hu = ufl.grad(ufl.grad(u)) # Hessian of displacement
Lu = ufl.div(ufl.grad(u))  # Laplacian of displacement

HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(I)

Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird  # without exp(-5|F|) to preserve symmetry of tangent problem
# Pi_Hu = k_r / 2 * (HuHu) * dxThird
Pi_R = Pi_HuLu

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
if args.reg_scaling:
    # Frederiksen Eq. (6): regularization contribution written directly as a residual, with
    # the exp(-5*det(F)) weight applied to the (HuHu - LuLu) variation. delta-u enters via
    # the test function derivatives Hδu, Lδu.
    Hdu = ufl.grad(ufl.grad(δu))
    Ldu = ufl.div(ufl.grad(δu))
    reg_scale = ufl.exp(-5.0 * J)
    R_reg = k_r * reg_scale * (
        ufl.inner(Hu, Hdu) - ufl.inner(Lu, Ldu) / ufl.tr(I)
    ) * dxThird
    residual = ufl.derivative(Pi + Pi_third, u, δu) + R_reg
else:
    residual = ufl.derivative(Pi + Pi_third + Pi_R, u, δu)
forms = ufl.extract_blocks(residual)


## dolfiny SNESproblem works, can be used in place of dolfinx.fem.petsc.NonlinearProblem

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

# The exp(-5*det(F)) regularization scaling makes the tangent NON-symmetric, so Cholesky
# (which assumes a symmetric matrix) is invalid -> fall back to an LU factorization.
pc_type = "lu" if args.reg_scaling else "cholesky"

problem = dolfinx.fem.petsc.NonlinearProblem(
    residual,
    u,
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
        "pc_type": pc_type,
        "pc_factor_mat_solver_type": "mumps",
    },
)

tm_func = fem.Function(V_tm, name="cell_markers")
tm_func.x.array[:] = ct.values

# J = det(F) diagnostic: interpolate det(F) into V_J and monitor its minimum over the
# third-medium cells to detect the onset of element inversion (J -> 0).
diagnostic = args.diagnostic
J_expr = fem.Expression(J, V_J.element.interpolation_points)
tm_cells = ct.find(cell_tags["tm"])
tm_J_dofs = np.unique(V_J.dofmap.list[tm_cells].reshape(-1))
coords_J = V_J.tabulate_dof_coordinates()  # (n_dofs, 3) reference coords of DG-1 J dofs


def report_min_max_J():
    """Interpolate J, report min/max over TM, its reference location, and #dofs near inversion.
    """
    J_func.interpolate(J_expr)
    J_tm = J_func.x.array[tm_J_dofs]
    imin = int(np.argmin(J_tm))
    min_J = float(J_tm[imin])
    loc = coords_J[tm_J_dofs[imin], :2].copy()
    n_low = int((J_tm < 0.1).sum())
    n_neg = int((J_tm <= 0.0).sum())
    # also report the maximum J over the TM: the two-sided ln(J)^2 barrier penalizes
    # expansion (J>1) as well, so max(J)>>1 signals the medium is being STRETCHED (the
    # opening/separation phase of the fold) and thus generating spurious tensile pressure.
    max_J = float(J_tm.max())
    n_exp = int((J_tm > 1.5).sum())
    pprint(
        f"    min(J)|TM = {min_J:+.4e} @ (x={loc[0]:+.1f}, y={loc[1]:+.1f})   "
        f"#dofs J<0.1 = {n_low}/{len(tm_J_dofs)}   #dofs J<=0 = {n_neg}   "
        f"max(J)|TM = {max_J:+.3e}   #dofs J>1.5 = {n_exp}",
        flush=True,
    )


u_prev = u.x.array.copy()
tm_prev = tm_func.x.array.copy()

# output file for storing results
ofile = VTXWriter(comm, f"{name}/{name}.bp", [u, tm_func])
ofile.write(0.0) # write initial state
# separate writer for J (DG-1): VTX requires one element type per file
ofile_J = VTXWriter(comm, f"{name}/{name}_J.bp", [J_func])
ofile_J.write(0.0)

# identify half-ring middle node 
mesh.topology.create_connectivity(0, tdim) # nodes-to-cells connectivity
m_node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], g0+t))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, m_node)

# --- Interface tangential-slip diagnostic (step B) --------------------------------------
# Measure the tangential (x) slip that the third medium accommodates between the OUTER-ring
# lower surface (arch2, radius R_ext, bonded to the TM top) and the block top edge (y=0,
# bonded to the TM bottom) directly beneath it, at several x-stations. In true FRICTIONLESS
# contact these two surfaces slide freely relative to one another; a BONDED continuous medium
# instead ties them, so slip ~ 0 would show the medium suppresses exactly the sliding that
# drives the reference's late fold. u is evaluated at fixed REFERENCE coordinates, so the
# values are material-point displacements and slip(x) = u_x(ring) - u_x(block).
center_y = g0 + Di / 2 + 2 * t                 # = 75, arch/ring centre height
R_ext = Di / 2 + 2 * t                         # = 55, outer-ring radius
slip_stations = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # x-stations [mm], must be < R_ext
ring_pts = np.column_stack(
    [slip_stations, center_y - np.sqrt(R_ext ** 2 - slip_stations ** 2)]
)  # arch2 lower surface
block_pts = np.column_stack([slip_stations, np.zeros_like(slip_stations)])  # block top (y=0)

bb_tree = geometry.bb_tree(mesh, tdim)


def _eval_u(points_xy):
    """Evaluate u at reference points (N,2); return (N,tdim), NaN where a point is not found."""
    pts = np.zeros((len(points_xy), 3))
    pts[:, :2] = points_xy
    cand = geometry.compute_collisions_points(bb_tree, pts)
    coll = geometry.compute_colliding_cells(mesh, cand, pts)
    out = np.full((len(points_xy), tdim), np.nan)
    cells, keep = [], []
    for i in range(len(points_xy)):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            keep.append(i)
    if keep:
        out[keep] = u.eval(pts[keep], np.array(cells, dtype=np.int32)).reshape(len(keep), tdim)
    return out


def eval_interface_slip():
    """Return (slip_x, gap_y) over slip_stations for the current displacement u."""
    u_ring = _eval_u(ring_pts)
    u_block = _eval_u(block_pts)
    slip_x = u_ring[:, 0] - u_block[:, 0]                       # tangential slip [mm]
    gap_y = (ring_pts[:, 1] + u_ring[:, 1]) - (block_pts[:, 1] + u_block[:, 1])  # current gap [mm]
    return slip_x, gap_y

# Adaptive loading
adaptive_load = False
MAX_FAILURE = 2
NUM_SUCCESSIVE_SOLVES = 0

full_disp = args.full_disp # [mm] full displacement applied to top of arches
pre_contact_disp_step = 1.0
contact_disp_step = 0.5
threshold = 20.0 # up to this displacement, use larger load increment (i.e. "pre_contact_disp_step") even if contact is established
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
# step-B diagnostic histories (one row per stored step, columns = slip_stations)
slip_history = np.empty((0, len(slip_stations)), dtype=np.float64)
gap_history = np.empty((0, len(slip_stations)), dtype=np.float64)
minJ_history = np.empty(0, dtype=np.float64)          # min(J)|TM per stored step
minJ_loc_history = np.empty((0, 2), dtype=np.float64)  # (x, y) of the crush per stored step
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

    report_min_max_J() # min(J)/max(J) over TM + its location (converged or not)

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
        

        if current_disp >= g0: # contact established, store loading history + slip/crush diagnostics
            loading_history = np.append(loading_history, ii_load)
            m_node_displ = np.append(m_node_displ, np.asarray(u.x.array[dofs_point_y]).reshape(-1)[0])
            
            if diagnostic:        
                slip_x, gap_y = eval_interface_slip()
                pprint(
                    f"    slip[x=0,20,40] = {slip_x[0]:+.3f}, {slip_x[2]:+.3f}, {slip_x[4]:+.3f} mm   "
                    f"gap[x=0,20,40] = {gap_y[0]:.3f}, {gap_y[2]:.3f}, {gap_y[4]:.3f} mm",
                    flush=True,
                )
            
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


