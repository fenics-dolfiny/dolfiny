#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_annulus_gmshapi as mg
import numpy as np

import dolfiny

# Basic settings
name = "bingham_lm_block"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
iR = 1.0
oR = 2.0
nR = 10 * 4
nT = 7 * 4
x0 = 0.0
y0 = 0.0

# Create the regular mesh of an annulus with given dimensions
gmsh_model, tdim = mg.mesh_annulus_gmshapi(name, iR, oR, nR, nT, x0, y0, do_quads=False)

# Get mesh and meshtags
mesh_data = dolfinx.io.gmshio.model_to_mesh(gmsh_model, comm, rank=0, gdim=2)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
ring_inner = mesh_data.physical_groups["ring_inner"][1]
ring_outer = mesh_data.physical_groups["ring_outer"][1]
domain = mesh_data.physical_groups["domain"][1]

# Fluid material parameters
rho = dolfinx.fem.Constant(mesh, scalar(2.0))  # [kg/m^3]
mu = dolfinx.fem.Constant(mesh, scalar(1.0))  # [kg/m/s]
tau_zero = dolfinx.fem.Constant(mesh, scalar(0.2))  # [kg/m/s^2]
tau_zero_regularisation = dolfinx.fem.Constant(mesh, scalar(1.0e-3))  # [-]

# Max inner ring velocity
v_inner_max = 0.1  # [m/s]
# Normal and tangential velocity at inner ring
v_n = dolfinx.fem.Constant(mesh, scalar(0.0))  # [m/s]
v_t = dolfinx.fem.Constant(mesh, scalar(0.0))  # [m/s] -- value set/updated in analysis

# Global time
time = dolfinx.fem.Constant(mesh, scalar(0.0))  # [s]
# Time step size
dt = dolfinx.fem.Constant(mesh, scalar(0.05))  # [s]
# Number of time steps
nT = 80

# Define integration measures
dx = ufl.Measure(
    "dx", domain=mesh, subdomain_data=mesh_data.cell_tags, metadata={"quadrature_degree": 4}
)
ds = ufl.Measure(
    "ds", domain=mesh, subdomain_data=mesh_data.facet_tags, metadata={"quadrature_degree": 4}
)


# Inner ring velocity
def v_inner_(t=0.0, vt=v_inner_max, g=5, a=1, b=3):
    return vt * 0.25 * (np.tanh(g * (t - a)) + 1) * (np.tanh(-g * (t - b)) + 1)


# Define elements
Ve = basix.ufl.element("P", mesh.basix_cell(), degree=2, shape=(mesh.geometry.dim,))
Pe = basix.ufl.element("P", mesh.basix_cell(), degree=1)
Le = basix.ufl.element("P", mesh.basix_cell(), degree=2)

# Define function spaces
Vf = dolfinx.fem.functionspace(mesh, Ve)
Pf = dolfinx.fem.functionspace(mesh, Pe)
Nf = dolfinx.fem.functionspace(mesh, Le)
Tf = dolfinx.fem.functionspace(mesh, Le)

# Define functions
v = dolfinx.fem.Function(Vf, name="v")
p = dolfinx.fem.Function(Pf, name="p")
n = dolfinx.fem.Function(Nf, name="n")
t = dolfinx.fem.Function(Tf, name="t")

vt = dolfinx.fem.Function(Vf, name="vt")
pt = dolfinx.fem.Function(Pf, name="pt")
nt = dolfinx.fem.Function(Nf, name="nt")
tt = dolfinx.fem.Function(Tf, name="tt")

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Vf, Pf, Nf, Tf))
δv, δp, δn, δt = δm

# Define state as (ordered) list of functions
m = [v, p, n, t]
mt = [vt, pt, nt, tt]
δm = [δv, δp, δn, δt]

# Create other functions
v_vector_o = dolfinx.fem.Function(Vf)
p_scalar_i = dolfinx.fem.Function(Pf)

# for output
vo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (2,))), name="v")
po = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)), name="p")
no = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)), name="n")
to = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)), name="t")

# Set up restriction
rdofsV = dolfiny.mesh.locate_dofs_topological(Vf, mesh_data.cell_tags, domain)
rdofsV = dolfiny.function.unroll_dofs(rdofsV, Vf.dofmap.bs)
rdofsP = dolfiny.mesh.locate_dofs_topological(Pf, mesh_data.cell_tags, domain)
rdofsN = dolfiny.mesh.locate_dofs_topological(Nf, mesh_data.facet_tags, ring_inner)
rdofsT = dolfiny.mesh.locate_dofs_topological(Tf, mesh_data.facet_tags, ring_inner)
restrc = dolfiny.restriction.Restriction([Vf, Pf, Nf, Tf], [rdofsV, rdofsP, rdofsN, rdofsT])

# Time integrator
odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=m, xt=mt, rho=0.8)


def D(v):
    """Rate of strain as function of v (velocity)."""
    return 0.5 * (ufl.grad(v).T + ufl.grad(v))


def J2(A):
    """Second (main) invariant J2 = (I_1)^2 - 2*(I_2) with I_1, I_2 principal invariants."""
    return 0.5 * ufl.inner(A, A)


def rJ2(A):
    """Square root of J2."""
    return ufl.sqrt(J2(A) + np.finfo(np.float64).eps)  # eps for AD


def T(v, p):
    """Constitutive relation for Bingham - Cauchy stress as a function of velocity and pressure."""
    # Deviatoric strain rate
    D_ = ufl.dev(D(v))  # == D(v) if div(v)=0
    # Second invariant
    rJ2_ = rJ2(D_)
    # Regularisation
    mu_effective = mu + tau_zero * 1.0 / (2.0 * (rJ2_ + tau_zero_regularisation))
    # Cauchy stress
    T = -p * ufl.Identity(2) + 2.0 * mu_effective * D_
    return T


# Helper
n_vec = ufl.FacetNormal(mesh)  # outward unit normal vector
t_vec = ufl.as_vector([n_vec[1], -n_vec[0]])  # tangent 2D

# Weak form (as one-form)
form = (
    ufl.inner(δv, rho * vt + rho * ufl.grad(v) * v) * dx
    + ufl.inner(ufl.grad(δv), T(v, p)) * dx
    + ufl.inner(δp, ufl.div(v)) * dx
    - ufl.inner(δv, n_vec) * n * ds(ring_inner)
    - ufl.inner(δv, t_vec) * t * ds(ring_inner)
    - δn * (v_n - ufl.inner(v, n_vec)) * ds(ring_inner)
    - δt * (v_t - ufl.inner(v, t_vec)) * ds(ring_inner)
    + dolfinx.fem.Constant(mesh, scalar(0.0)) * ufl.inner(δp, p) * dx
)  # Zero pressure block for BCs

# Overall form (as one-form)
form = odeint.discretise_in_time(form)
# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_data(mesh_data)
# Write initial state
dolfiny.interpolation.interpolate(v, vo)
dolfiny.interpolation.interpolate(p, po)
dolfiny.interpolation.interpolate(n, no)
dolfiny.interpolation.interpolate(t, to)
ofile.write_function(vo, time.value)
ofile.write_function(po, time.value)
ofile.write_function(no, time.value)
ofile.write_function(to, time.value)

# Options for PETSc backend
opts = PETSc.Options("bingham")  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix="bingham", restriction=restrc)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
ring_outer_dofs_Vf = dolfiny.mesh.locate_dofs_topological(Vf, mesh_data.facet_tags, ring_outer)
ring_inner_dofs_Pf = dolfiny.mesh.locate_dofs_topological(Pf, mesh_data.facet_tags, ring_outer)

# Process time steps
for time_step in range(1, nT + 1):
    dolfiny.utils.pprint(
        f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}"
    )

    # Stage next time step
    odeint.stage()

    # Update functions (taking up time.value)
    v_t.value = v_inner_(t=time.value)

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(v_vector_o, ring_outer_dofs_Vf),  # velocity ring_outer
        dolfinx.fem.dirichletbc(p_scalar_i, ring_inner_dofs_Pf),  # pressure ring_inner
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Update solution states for time integration
    odeint.update()

    # Write output
    dolfiny.interpolation.interpolate(v, vo)
    dolfiny.interpolation.interpolate(p, po)
    dolfiny.interpolation.interpolate(n, no)
    dolfiny.interpolation.interpolate(t, to)
    ofile.write_function(vo, time.value)
    ofile.write_function(po, time.value)
    ofile.write_function(no, time.value)
    ofile.write_function(to, time.value)

ofile.close()
