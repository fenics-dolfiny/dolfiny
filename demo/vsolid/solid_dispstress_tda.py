#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_block3d_gmshapi as mg

import dolfiny

# Basic settings
name = "solid_dispstress_tda"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dx, dy, dz = 2.0, 0.01, 0.1
nx, ny, nz = 20, 2, 2

# Create the regular mesh of a block with given dimensions
gmsh_model, tdim = mg.mesh_block3d_gmshapi(
    name, dx, dy, dz, nx, ny, nz, px=1.0, py=1.0, pz=1.0, do_quads=False
)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# Write mesh and meshtags to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh, mts)

# Read mesh and meshtags from file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "r") as ifile:
    mesh, mts = ifile.read_mesh_meshtags()

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_left = interfaces_keys["surface_left"]
surface_right = interfaces_keys["surface_right"]

# Solid material parameters
rho = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e4))  # [1e-9 * 1e+4 kg/m^3]
eta = dolfinx.fem.Constant(mesh, scalar(1e-9 * 0e4))  # [1e-9 * 0e+4 kg/m^3/s]
mu = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e11))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
la = dolfinx.fem.Constant(mesh, scalar(1e-9 * 1e10))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]

# Load
b = dolfinx.fem.Constant(mesh, [0.0, -10, 0.0])  # [m/s^2]

# Global time
time = dolfinx.fem.Constant(mesh, scalar(0.0))  # [s]
# Time step size
dt = dolfinx.fem.Constant(mesh, scalar(1e-2))  # [s]
# Number of time steps
nT = 200

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(3,))
Se = basix.ufl.element("Regge", mesh.basix_cell(), 1)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
S = dolfinx.fem.Function(Sf, name="S")

ut = dolfinx.fem.Function(Uf, name="ut")
St = dolfinx.fem.Function(Sf, name="St")

utt = dolfinx.fem.Function(Uf, name="utt")
Stt = dolfinx.fem.Function(Sf, name="Stt")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Sf))
δu, δS = δm

# Define state and rate as (ordered) list of functions
m, mt, mtt = [u, S], [ut, St], [utt, Stt]

# Create other functions
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (3,))), name="u")  # for output
So = dolfinx.fem.Function(
    dolfinx.fem.functionspace(mesh, ("P", 1, (3, 3), True)), name="S"
)  # for output

# Time integrator
odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=m, xt=mt, xtt=mtt, rho=0.95)

# Configuration gradient
I = ufl.Identity(3)  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
# E = E(u) total strain
E = 1 / 2 * (F.T * F - I)
# E = E(S) elastic strain
Es = 1 / (2 * mu) * S - la / (2 * mu * (3 * la + 2 * mu)) * ufl.tr(S) * I

# Variation of rate of Green-Lagrange strain
δE = ufl.derivative(E, m, δm)

# Weak form (as one-form)
form = (
    ufl.inner(δu, rho * utt) * dx
    + ufl.inner(δu, eta * ut) * dx
    + ufl.inner(δE, S) * dx
    + ufl.inner(δS, E - Es) * dx
    - ufl.inner(δu, rho * b) * dx
)

# Optional: linearise weak form
# form = dolfiny.expression.linearise(form, m)  # linearise around zero state

# Overall form (as one-form)
form = odeint.discretise_in_time(form)
# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)
# Write initial state
dolfiny.interpolation.interpolate(u, uo)
dolfiny.interpolation.interpolate(S, So)
ofile.write_function(uo, time.value)
ofile.write_function(So, time.value)

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_left)

# Process time steps
for time_step in range(1, nT + 1):
    dolfiny.utils.pprint(
        f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}\n"
    )

    # Stage next time step
    odeint.stage()

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(u_, surface_left_dofs_Uf),  # disp left (clamped)
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Update solution states for time integration
    odeint.update()

    # Write output
    dolfiny.interpolation.interpolate(u, uo)
    dolfiny.interpolation.interpolate(S, So)
    ofile.write_function(uo, time.value)
    ofile.write_function(So, time.value)

ofile.close()
