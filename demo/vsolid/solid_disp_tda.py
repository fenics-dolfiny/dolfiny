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
name = "solid_disp_tda"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dimensions = (2.0, 0.01, 0.1)
elements = 20, 2, 2

# Create the regular mesh of a block with given dimensions
gmsh_model, tdim = mg.mesh_block3d_gmshapi(
    name, *dimensions, *elements, px=1.0, py=1.0, pz=1.0, do_quads=False
)

# Get mesh and meshtags
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
surface_left = mesh_data.physical_groups["surface_left"].tag
surface_right = mesh_data.physical_groups["surface_right"].tag

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
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(3,))

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
ut = dolfinx.fem.Function(Uf, name="ut")
utt = dolfinx.fem.Function(Uf, name="utt")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf))
(δu,) = δm

# Define state and rate as (ordered) list of functions
m, mt, mtt = [u], [ut], [utt]

# Create other functions for output
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (3,))), name="u")

# Time integrator
odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=m, xt=mt, xtt=mtt, rho=0.95)

# Configuration gradient
I = ufl.Identity(3)  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
# E = E(u) total strain
E = 1 / 2 * (F.T * F - I)
# S = S(E) stress
S = 2 * mu * E + la * ufl.tr(E) * I

# Variation of rate of Green-Lagrange strain
δE = ufl.derivative(E, m, δm)

# Weak form (as one-form)
form = (
    ufl.inner(δu, rho * utt) * dx
    + ufl.inner(δu, eta * ut) * dx
    + ufl.inner(δE, S) * dx
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
# Write mesh data
ofile.write_mesh_data(mesh_data)

# Write initial state
dolfiny.interpolation.interpolate(u, uo)
ofile.write_function(uo, time.value)

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
opts["mat_mumps_cntl_1"] = 0.0

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, mesh_data.facet_tags, surface_left)

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
    ofile.write_function(uo, time.value)

ofile.close()
