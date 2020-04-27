#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import Constant, fem, FunctionSpace, Function, log
from dolfinx.io import XDMFFile
import ufl

from dolfiny.odeint import ODEInt
import dolfiny.snesblockproblem
import mesh_annulus_gmshapi as mg
import prepare_output as po

# Basic settings
name = "bingham"

# Init files and folders
po.init(name)

# Geometry and mesh parameters
iR = 1.0
oR = 2.0
nR = 10 * 4
nT = 7 * 4
x0 = 0.0
y0 = 0.0

# create the regular mesh of an annulus with given dimensions
labels = mg.mesh_annulus_gmshapi(name, iR, oR, nR, nT, x0, y0, do_quads=False, progression=1.1)

# read mesh, subdomains and interfaces
with XDMFFile(MPI.COMM_WORLD, name + ".xdmf", "r") as infile:
    mesh = infile.read_mesh("Grid")
    mesh.topology.create_connectivity_all()

with XDMFFile(MPI.COMM_WORLD, name + "_subdomains" + ".xdmf", "r") as infile:
    subdomains = infile.read_meshtags(mesh, "Grid")

with XDMFFile(MPI.COMM_WORLD, name + "_interfaces" + ".xdmf", "r") as infile:
    interfaces = infile.read_meshtags(mesh, "Grid")

inner = labels["ring_inner"]
outer = labels["ring_outer"]

# Fluid material parameters
rho = Constant(mesh, 2.0)  # [kg/m^3]
mu = Constant(mesh, 1.0)  # [kg/m/s]
tau_zero = Constant(mesh, 0.2)  # [kg/m/s^2]

# Max inner ring velocity
v_inner_max = 0.1  # [m/s]
# Stabilisation material (for tau_zero)
stab_material = Constant(mesh, 1.e-3)

# Global time
time = Constant(mesh, 0.0)  # [s]
# Time step size
dt = 0.1  # [s]
# Number of time steps
TS = 40

# Define measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Check geometry data
area_msh = fem.assemble_scalar(1 * dx)
area_ana = np.pi * (oR ** 2 - iR ** 2)
print("geometry subdomain area      = %4.3e (rel error = %4.3e)" %
      (area_msh, np.sqrt((area_msh - area_ana) ** 2 / area_ana ** 2)))
ring_msh = fem.assemble_scalar(1 * ds(outer))
ring_ana = 2.0 * np.pi * oR
print("geometry boundary outer ring = %4.3e (rel error = %4.3e)" %
      (ring_msh, np.sqrt((ring_msh - ring_ana) ** 2 / ring_ana ** 2)))
ring_msh = fem.assemble_scalar(1 * ds(inner))
ring_ana = 2.0 * np.pi * iR
print("geometry boundary inner ring = %4.3e (rel error = %4.3e)" %
      (ring_msh, np.sqrt((ring_msh - ring_ana) ** 2 / ring_ana ** 2)))


# Inner ring velocity
def v_inner_(t=0.0, vt=v_inner_max, g=5, a=1, b=3):
    return vt * 0.25 * (np.tanh(g * (t - a)) + 1) * (np.tanh(-g * (t - b)) + 1)


# Define helpers
def n_vector_(x, r=iR):
    return np.array([-x[0], - x[1]]) / r


def t_vector_(x, r=iR):
    return np.array([-x[1], x[0]]) / r


# Boundary velocity as expression
def v_vector_i_(x):
    return t_vector_(x) * v_inner_(t=time.value)


def v_vector_o_(x):
    return np.zeros((mesh.geometry.dim, x.shape[1]))


# Function spaces
Ve = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
Pe = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, Ve)
P = FunctionSpace(mesh, Pe)

# Define functions
v = Function(V, name="v")
p = Function(P, name="p")

δv = ufl.TestFunction(V)
δp = ufl.TestFunction(P)

# Create (zero) initial conditions
v0 = Function(V)
v0t = Function(V)
p0 = Function(P)
p0t = Function(P)

m = [v, p]
m0 = [v0, p0]
m0t = [v0t, p0t]

# Create other functions
v_vector_o = Function(V)
v_vector_i = Function(V)
p_scalar_i = Function(P)

n_vector = Function(V)
t_vector = Function(V)

# Time integrator
odeint = ODEInt(dt=dt)


def D(v):
    """Rate of strain as function of v (velocity)."""
    return 0.5 * (ufl.grad(v).T + ufl.grad(v))


def J2(A):
    """Second (main) invariant J2 = (I_1)^2 - 2*(I_2) with I_1, I_2 principal invariants."""
    return 0.5 * ufl.inner(A, A)


def rJ2(A):
    """Square root of J2."""
    return ufl.sqrt(J2(A) + np.finfo(np.float).eps)  # eps for AD


def T(v, p):
    """Constitutive relation for Bingham - Cauchy stress as a function of velocity and pressure."""
    # Deviatoric strain rate
    D_ = ufl.dev(D(v))  # == D(v) if div(v)=0
    # Second invariant
    rJ2_ = rJ2(D_)
    # Regularisation
    mu_effective = mu + tau_zero * 1.0 / (2. * (rJ2_ + stab_material))
    # Cauchy stress
    T = -p * ufl.Identity(2) + 2.0 * mu_effective * D_
    return T


# Weak form: time-rate-dependent components
def g(dmdt, time_instant):
    dvdt = dmdt[0]
    g = [ufl.inner(δv, rho * dvdt) * dx, 0 * dx]
    return g


# Weak form: time-rate-independent components
def f(m, time_instant):
    v = m[0]
    p = m[1]
    f = [ufl.inner(δv, rho * ufl.grad(v) * v) * dx
         + ufl.inner(ufl.grad(δv), T(v, p)) * dx,
         ufl.inner(δp, ufl.div(v)) * dx]
    return f


# Overall form
F = [g + f for g, f in zip(odeint.g_(g, m, m0, m0t), odeint.f_(f, m, m0))]

# output files
ofile = XDMFFile(MPI.COMM_WORLD, name + "_results.xdmf", "w")
ofile.write_mesh(mesh)

# record some values
array_time_instant = np.zeros(TS + 1)
array_shear_stress = np.zeros(TS + 1)
array_shear_gammad = np.zeros(TS + 1)
array_tangent_velocity = np.zeros(TS + 1)
array_tangent_traction = np.zeros(TS + 1)

opts = PETSc.Options()

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-10
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 500
opts["mat_mumps_icntl_24"] = 1

problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, opts=opts)

outerdofs_V = fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, interfaces.indices[np.where(interfaces.values == outer)[0]])
innerdofs_V = fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, interfaces.indices[np.where(interfaces.values == inner)[0]])
innerdofs_P = fem.locate_dofs_topological(
    P, mesh.topology.dim - 1, interfaces.indices[np.where(interfaces.values == inner)[0]])


# Process time steps
for i in range(TS + 1):

    # Set current time
    time.value = i * odeint.dt

    # Update functions
    v_vector_o.interpolate(v_vector_o_)
    v_vector_i.interpolate(v_vector_i_)
    n_vector.interpolate(n_vector_)
    t_vector.interpolate(t_vector_)

    print("Processing time instant = %4.3f in step %d " % (time.value, i), end="\n")

    # Set/update boundary conditions
    problem.bcs = [
        fem.DirichletBC(v_vector_o, outerdofs_V),  # velo outer
        fem.DirichletBC(v_vector_i, innerdofs_V),  # velo inner
        fem.DirichletBC(p_scalar_i, innerdofs_P),  # pressure inner
    ]

    # Solve nonlinear problem
    m = problem.solve()

    # Extract solution
    v_, p_ = m

    assert problem.snes.getKSP().getConvergedReason() > 0

    # Write output
    log.set_log_level(log.LogLevel.OFF)
    ofile.write_function(p_, float(time.value))
    ofile.write_function(v_, float(time.value))
    log.set_log_level(log.LogLevel.WARNING)

    # Extract and analyse data
    array_time_instant[i] = time.value
    array_shear_stress[i] = fem.assemble_scalar(
        rJ2(ufl.dev(T(v_, p_))) * ds(inner)) / fem.assemble_scalar(1.0 * ds(inner))
    array_shear_gammad[i] = fem.assemble_scalar(
        rJ2(ufl.dev(D(v_))) * ds(inner)) / fem.assemble_scalar(1.0 * ds(inner))
    array_tangent_velocity[i] = fem.assemble_scalar(
        ufl.sqrt(ufl.dot(v_, v_)) * ds(inner)) / fem.assemble_scalar(1.0 * ds(inner))
    t_ = ufl.dot(ufl.dev(T(v_, p_)), n_vector)
    array_tangent_traction[i] = fem.assemble_scalar(
        ufl.sqrt(ufl.dot(t_, t_)) * ds(inner)) / fem.assemble_scalar(1.0 * ds(inner))

    # Update solution states for time integration
    m0, m0t = odeint.update(m, m0, m0t)

ofile.close()

# Identify apparent yield stress tau_a and apparent viscosity mu_a by linear regression
threshold = 0.05  # minimum dot gamma
selection = np.greater(array_shear_gammad, threshold)
x = array_shear_gammad[selection]
y = array_shear_stress[selection]
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=-1)[0]  # determine linear fct param: y = m*x + c
mu_a = 0.5 * m
tau_a = c  # slope m = 2*mu_a
print("apparent viscosity    = %4.3e" % mu_a)
print("apparent yield stress = %4.3e" % tau_a)

# Plot shear stress - shear strain rate curve
fig, ax1 = plt.subplots()
color = "tab:green"
ax1.plot(array_shear_gammad, array_shear_stress, "o-", color=color)
ax1.set_xlabel(r"shear strain rate ${\dot\gamma} = \sqrt{J_2(D_{dev})}$ $[1/s]$", fontsize=12)
ax1.set_ylabel(r"shear wall stress ${\tau} = \sqrt{J_2(T_{dev})}$ $[N/m^2]$", fontsize=12, color=color)
ax1.grid(linewidth=0.5)
ax1.set_title(r"Co-axial cylinder rheometer: Bingham fluid", fontsize=12)
fig.tight_layout()
fig.savefig(name + "_shear.pdf")

# Plot values over time
fig, ax1 = plt.subplots()
color = "tab:red"
ax1.plot(array_time_instant, array_tangent_velocity, color=color)
ax1.set_xlabel(r"time $t$ $[s]$", fontsize=12)
ax1.set_xlim([0, dt * TS])
ax1.set_ylabel(r"boundary tangent velocity $|v|$ $[m/s]$", fontsize=12, color=color)
ax1.set_ylim([0, 0.1])
ax2 = ax1.twinx()
color = "tab:blue"
ax2.plot(array_time_instant, array_tangent_traction, color=color)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel(r"boundary tangent stress $|\lambda|$ $[N/m^2]$", fontsize=12, color=color)
ax2.set_ylim([-0.1, 1.0])
ax2.set_title(r"Co-axial cylinder rheometer: Bingham fluid", fontsize=12)
fig.tight_layout()
fig.savefig(name + "_time.pdf")