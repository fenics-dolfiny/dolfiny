#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_annulus_gmshapi as mg
import numpy as np
import sympy.physics.units as syu

import dolfiny
import dolfiny.utils
from dolfiny.units import Quantity

# Basic settings
name = "bingham_block"
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

# Fluid material parameters
tau_zero = dolfinx.fem.Constant(mesh, scalar(0.2))  # [kg/m/s^2]
tau_zero_regularisation = dolfinx.fem.Constant(mesh, scalar(1.0e-3))  # [-]

# Max inner ring velocity
v_inner_max = 0.1  # [m/s]

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


# Define helpers
def n_vector_(x, r=iR):
    return np.array([-x[0], -x[1]]) / r


def t_vector_(x, r=iR):
    return np.array([-x[1], x[0]]) / r


# Boundary velocity as expression
def v_vector_i_(x):
    return t_vector_(x) * v_inner_(t=time.value)


def v_vector_o_(x):
    return np.zeros((mesh.geometry.dim, x.shape[1]))


# Define elements
Ve = basix.ufl.element("P", mesh.basix_cell(), degree=2, shape=(mesh.geometry.dim,))
Pe = basix.ufl.element("P", mesh.basix_cell(), degree=1)

# Define function spaces
Vf = dolfinx.fem.functionspace(mesh, Ve)
Pf = dolfinx.fem.functionspace(mesh, Pe)

# Define functions
v = dolfinx.fem.Function(Vf, name="v")
p = dolfinx.fem.Function(Pf, name="p")

vt = dolfinx.fem.Function(Vf, name="vt")
pt = dolfinx.fem.Function(Pf, name="pt")

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Vf, Pf))
δv, δp = δm

# Define state and rate as (ordered) list of functions
m = [v, p]
mt = [vt, pt]

# Create other functions
v_vector_o = dolfinx.fem.Function(Vf)
v_vector_i = dolfinx.fem.Function(Vf)
p_scalar_i = dolfinx.fem.Function(Pf)

# for output
vo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (2,))), name="v")
po = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)), name="p")

# Time integrator
odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=m, xt=mt)

# Define reference quantities with units
t_ref = Quantity(mesh, 1, syu.second, "t_ref")
v_ref = Quantity(mesh, 1, syu.meter / syu.second, "v_ref")
l_ref = Quantity(mesh, 1, syu.meter, "l_ref")
rho = Quantity(mesh, 2, syu.kilogram / syu.meter**3, "rho")
mu = Quantity(mesh, 1, syu.kilogram / (syu.meter * syu.second), "mu")
tau_zero = Quantity(mesh, 0.2, syu.kilogram / (syu.meter * syu.second**2), "tau_zero")


def D(v):
    """Rate of strain as function of v (velocity)."""
    return 0.5 * (ufl.grad(v).T + ufl.grad(v))


def J2(A):
    """Second (main) invariant J2 = (I_1)^2 - 2*(I_2) with I_1, I_2 principal invariants."""
    return 0.5 * ufl.inner(A, A)


def rJ2(A):
    """Square root of J2."""
    return ufl.sqrt(J2(A) + (v_ref / l_ref) ** 2 * np.finfo(np.float64).eps)  # eps for AD


def T_mu(v, p):
    """Viscous part of Cauchy stress tensor."""
    # Deviatoric strain rate
    D_ = ufl.dev(D(v))  # == D(v) if div(v)=0
    # Cauchy stress (viscous part)
    T = -p * ufl.Identity(2) + 2.0 * mu * D_
    return T


def T_tau(v):
    """Yield stress part of Cauchy stress tensor for Bingham fluid."""
    # Deviatoric strain rate
    D_ = ufl.dev(D(v))  # == D(v) if div(v)=0
    # Second invariant
    rJ2_ = rJ2(D_)
    # Regularisation factor
    regularisation = 1.0 / (2.0 * (rJ2_ + (v_ref / l_ref) * tau_zero_regularisation))
    # Yield stress contribution
    T = 2.0 * tau_zero * regularisation * D_
    return T


def T(v, p):
    return T_mu(v, p) + T_tau(v)


quantities = [t_ref, v_ref, l_ref, rho, mu, tau_zero]
if MPI.COMM_WORLD.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

# Create mapping for dimensional transformation
p_ref = mu * v_ref / l_ref
mapping = {
    mesh.ufl_domain(): l_ref,  # Add mesh scaling to mapping
    v: v_ref * v,
    vt: v_ref / t_ref * vt,
    p: p_ref * p,
    δv: v_ref * δv,
    δp: p_ref * δp,
}

# Define terms as dictionary
terms = {
    "time": ufl.inner(δv, rho * vt) * dx,
    "conv": ufl.inner(δv, rho * ufl.grad(v) * v) * dx,
    "stress_mu": ufl.inner(ufl.grad(δv), T_mu(v, p)) * dx,
    "stress_tau": ufl.inner(ufl.grad(δv), T_tau(v)) * dx,
    "cont": ufl.inner(δp, ufl.div(v)) * dx,
    "press_diag": dolfinx.fem.Constant(mesh, scalar(0.0))
    * v_ref
    / l_ref
    / p_ref
    * ufl.inner(δp, p)
    * dx,
}

# Few dimensional sanity checks
dimsys = syu.si.SI.get_dimension_system()

assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(rho * vt, quantities, mapping),
    syu.mass / syu.length**3 * syu.length / syu.time**2,
)

assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(tau_zero, quantities, mapping),
    syu.mass / (syu.length * syu.time**2),
)

factorized = dolfiny.units.factorize(terms, quantities, mode="factorize", mapping=mapping)
assert isinstance(factorized, dict)

# Choose reference term for scaling
reference_term = "conv"
normalized = dolfiny.units.normalize(factorized, reference_term, quantities)

# Weak form (as one-form)
form = sum(normalized.values(), ufl.form.Zero())

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
ofile.write_function(vo, time.value)
ofile.write_function(po, time.value)

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
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix="bingham")

# Identify dofs of function spaces associated with tagged interfaces/boundaries
ring_outer_dofs_Vf = dolfiny.mesh.locate_dofs_topological(Vf, mesh_data.facet_tags, ring_outer)
ring_inner_dofs_Vf = dolfiny.mesh.locate_dofs_topological(Vf, mesh_data.facet_tags, ring_inner)
ring_inner_dofs_Pf = dolfiny.mesh.locate_dofs_topological(Pf, mesh_data.facet_tags, ring_inner)

# Process time steps
for time_step in range(1, nT + 1):
    dolfiny.utils.pprint(
        f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}"
    )

    # Stage next time step
    odeint.stage()

    # Update functions (taking up time.value)
    v_vector_o.interpolate(v_vector_o_)
    v_vector_i.interpolate(v_vector_i_)

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(v_vector_o, ring_outer_dofs_Vf),  # velocity ring_outer
        dolfinx.fem.dirichletbc(v_vector_i, ring_inner_dofs_Vf),  # velocity ring_inner
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
    ofile.write_function(vo, time.value)
    ofile.write_function(po, time.value)

ofile.close()
