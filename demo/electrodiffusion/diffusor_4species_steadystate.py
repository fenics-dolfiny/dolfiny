#!/usr/bin/env python3

from collections import namedtuple

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
import mesh_diffusor_gmshapi as mg
import numpy as np
import plot_diffusor_pyvista as pl

import dolfiny

# references:
# https://doi.org/10.1524/zpch.1954.1.5_6.305
# https://doi.org/10.1006/jcis.1999.6145

# Basic settings
name = "diffusor_4species_steadystate"
comm = MPI.COMM_WORLD

# Create the mesh of the diffusor [m]
gmsh_model, tdim = mg.mesh_diffusor_gmshapi(name)

# Get mesh and meshtags
partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, partitioner=partitioner)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_one = interfaces_keys["surface_one"]
surface_two = interfaces_keys["surface_two"]
surface_outer = interfaces_keys["surface_outer"]
surface_inner = interfaces_keys["surface_inner"]

# System constants
ε_r = dolfinx.fem.Constant(mesh, scalar(80.0))  # [-] -- relative permittivity
ε_0 = dolfinx.fem.Constant(mesh, scalar(8.854187819e-12))  # [A^2 s^4 / kg / m^3] -- permittivity
e_0 = dolfinx.fem.Constant(mesh, scalar(1.602176634e-19))  # [A s] -- elementary electrical charge
F = dolfinx.fem.Constant(mesh, scalar(9.648533212e04))  # [A s / mol] -- Faraday constant
R = dolfinx.fem.Constant(mesh, scalar(8.314462618))  # [kg m^2 / s^2 / K / mol] -- gas constant
T = dolfinx.fem.Constant(mesh, scalar(300.0))  # [K] -- temperature

# System parameters
Ion = namedtuple("Ion", ["name", "z", "D", "a"])
species = [
    Ion(name="$ SO^{2-}_{4} $", z=-2, D=3.0e-10, a=6.0e-10),
    Ion(name="$ Mg^{2+}_{ } $", z=+2, D=3.0e-10, a=8.0e-10),
    Ion(name="$ Na^{ +}_{ } $", z=+1, D=3.0e-10, a=4.0e-10),
    Ion(name="$  K^{ +}_{ } $", z=+1, D=5.0e-10, a=3.0e-10),
]

z = dolfinx.fem.Constant(mesh, [scalar(s.z) for s in species])  # [-] -- valence number
D = dolfinx.fem.Constant(mesh, [scalar(s.D) for s in species])  # [m^2 / s] -- diffusivity
a = dolfinx.fem.Constant(mesh, [scalar(s.a) for s in species])  # [m] -- ion radius
n = len(species)  # number of species

β = 50  # [-] -- (optional) concentration scaling: low/high
c_1 = β * np.array([6, 3, 0, 0]) / 6  # [mmol / l] = [mol / m^3] -- concentrations left
c_2 = β * np.array([5, 0, 1, 3]) / 6  # [mmol / l] = [mol / m^3] -- concentrations right
c_i = β * np.array([3, 0, 0, 0]) / 6  # [mmol / l] = [mol / m^3] -- concentrations initial (domain)

assert c_1 @ z.value == c_2 @ z.value == c_i @ z.value, "Check electroneutrality constraint!"

φ_1 = scalar(0.0)  # [V] -- voltage left
φ_2 = R.value * T.value / F.value * 5  # [V] -- voltage right (RT / F * ln(ζ))

w = -ufl.dot(ufl.as_vector(c_1), z)  # [mol / m^3] -- fixed charge concentration (electroneutrality)

# Define elements
porder = 1  # ansatz order for physics
Ce = basix.ufl.element("P", mesh.basix_cell(), porder, shape=(n,))
Pe = basix.ufl.element("P", mesh.basix_cell(), porder)

# Define function spaces
Cf = dolfinx.fem.functionspace(mesh, Ce)
Pf = dolfinx.fem.functionspace(mesh, Pe)

# Define functions
c = dolfinx.fem.Function(Cf, name="c")  # [mol / m^3] -- concentration
φ = dolfinx.fem.Function(Pf, name="φ")  # [V] -- electrostatic potential

δc, δφ = ufl.TestFunctions(ufl.MixedFunctionSpace(Cf, Pf))

# Define state as (ordered) list of functions
m, δm = [c, φ], [δc, δφ]

# Create other functions: output / visualisation
vorder = mesh.geometry.cmap.degree
co = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (n,))), name="c")  # type: ignore
φo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="φ")  # type: ignore

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Expressions, (extended) Debye-Hückel model
A = ufl.sqrt(2) * F**2 * e_0 / (8 * ufl.pi) / ufl.sqrt((ε_0 * ε_r * R * T) ** 3)  # parameter A
B = ufl.sqrt(2) * F / ufl.sqrt(ε_0 * ε_r * R * T)  # parameter B
S = ufl.dot(ufl.elem_mult(z, z), c) / 2 + w * n / 2  # ionic strength [mol / m^3]

# Weak form: components (as one-form)
form = ufl.inner(ufl.grad(δφ), ε_r * ε_0 * ufl.grad(φ)) * dx
form -= δφ * F * (ufl.dot(z, c) + w) * dx

for δck, ck, zk, Dk, ak in zip(δc, c, z, D, a):  # add species
    # chemical activity coefficient [-]
    ln_γk = -A * ufl.sqrt(S) * zk**2 / (1 + B * ak * ufl.sqrt(S))
    # advective velocity [1 / m]
    vk = zk * F / (R * T) * ufl.grad(φ) + ufl.grad(ln_γk)
    # ion flux = (isotropic) diffusion + electro-migration [mol / s / m^2]
    fk = -Dk * (ufl.grad(ck) + vk * ck)

    form -= ufl.inner(ufl.grad(δck), fk) * dx

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
bcsdofs_Cf_one = dolfiny.mesh.locate_dofs_topological(Cf, interfaces, surface_one)
bcsdofs_Cf_two = dolfiny.mesh.locate_dofs_topological(Cf, interfaces, surface_two)
bcsdofs_Pf_one = dolfiny.mesh.locate_dofs_topological(Pf, interfaces, surface_one)
bcsdofs_Pf_two = dolfiny.mesh.locate_dofs_topological(Pf, interfaces, surface_two)

# Boundary conditions
bcs = [
    # concentrations
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, c_1), bcsdofs_Cf_one, Cf),
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, c_2), bcsdofs_Cf_two, Cf),
    # potential
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, φ_1), bcsdofs_Pf_one, Pf),
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, φ_2), bcsdofs_Pf_two, Pf),
]

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "none"
opts["snes_atol"] = 1.0e-13  # required to enforce SNES iteration 2
opts["snes_rtol"] = 1.0e-15  # required to enforce SNES iteration 2
opts["snes_stol"] = 1.0e-06
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, bcs, prefix=name)

# Interpolate initial state
dolfiny.interpolation.interpolate(dolfinx.fem.Constant(mesh, c_i), c)

# Solve nonlinear problem
problem.solve(u_init=m)

# Assert convergence of nonlinear solver
problem.status(verbose=True, error_on_failure=True)

# Interpolate for output
dolfiny.interpolation.interpolate(c, co)
dolfiny.interpolation.interpolate(φ, φo)

# Write to xdmf file -- open in Paraview with Xdmf3ReaderT
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(co)
    ofile.write_function(φo)

# Visualise
pl.plot_diffusor_pyvista(name)

# Evaluate solution on path
n_ = 100 + 1
r_ = 0.0025
ε_ = 10 ** -(np.finfo(mesh.geometry.x.dtype).precision // 2 - 1)
s_ = np.linspace(0.0 + ε_, 1.0 - ε_, n_)
x_ = np.vstack([np.zeros((n_,)), r_ * (1 - np.cos(s_ * np.pi / 2)), r_ * np.sin(s_ * np.pi / 2)])

c_ = dolfiny.function.evaluate(c, x_)
φ_ = dolfiny.function.evaluate(φ, x_)

# Plot (only on rank = 0)
if comm.rank == 0:
    title = "Ion transport ($Mg SO_4$ | $Na_2 SO_4 + K_2 SO_4$)"

    fig, ax1 = plt.subplots(dpi=400)
    ax1.set_title(f"{title}, steady-state", fontsize=12)
    ax1.set_xlabel("path coordinate $s$ $[-]$", fontsize=12)
    ax1.set_ylabel("concentration $c_k$ $[mol/m^3]$", fontsize=12)
    ax1.grid(linewidth=0.25)
    ax2 = ax1.twinx()
    ax2.set_ylabel("electrostatic potential $φ$ $[V]$", fontsize=12)

    ax1.plot(s_, c_, "-", label=[s.name for s in species])
    ax1.ticklabel_format(style="sci", scilimits=(0, -2), axis="x")
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, β))
    ax1.legend(loc="best")

    ax2.plot(s_, φ_, "-", color="black", alpha=0.5)
    ax2.set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
    ax2.set_ylim((0, 0.15))

    fig.tight_layout()
    fig.savefig(f"{name}.pdf")
