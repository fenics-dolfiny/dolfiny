#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
import mesh_iso6892_gmshapi as mg
import numpy as np

import dolfiny

# references:
# https://doi.org/10.1007/978-94-011-2860-5_66
# https://doi.org/10.1016/j.commatsci.2012.05.062
# https://doi.org/10.24355/dbbs.084-202112170722-0

# Basic settings
name = "solid_plasticity_monolithic"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
l0, d0 = 0.10, 0.02  # [m]
nr = 5
# Geometry and physics ansatz order
o, p = 1, 1

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mg.mesh_iso6892_gmshapi(name, l0, d0, nr, order=o)

# Create the mesh of the specimen with given dimensions and save as msh, then read into gmsh model
# mg.mesh_iso6892_gmshapi(name, l0, d0, nr, order=o, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

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
domain_gauge = subdomains_keys["domain_gauge"]
surface_1 = interfaces_keys["surface_grip_left"]
surface_2 = interfaces_keys["surface_grip_right"]

# Solid: material parameters
mu = dolfinx.fem.Constant(mesh, scalar(100.0))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
la = dolfinx.fem.Constant(mesh, scalar(10.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
Sy = dolfinx.fem.Constant(mesh, scalar(0.300))  # initial yield stress [GPa]
bh = dolfinx.fem.Constant(mesh, scalar(20.00))  # isotropic hardening: saturation rate  [-]
qh = dolfinx.fem.Constant(mesh, scalar(0.100))  # isotropic hardening: saturation value [GPa]
bb = dolfinx.fem.Constant(mesh, scalar(250.0))  # kinematic hardening: saturation rate  [-]
qb = dolfinx.fem.Constant(mesh, scalar(0.100))  # kinematic hardening: saturation value [GPa] (2/3)

# Solid: load parameters
μ = dolfinx.fem.Constant(mesh, scalar(1.0))  # load factor


def u_bar(x):
    return μ.value * np.array([l0 * 0.01 * np.sign(x[0]), 0.0 * x[1], 0.0 * x[2]])


# Define integration measures
quad_degree = p
dx = ufl.Measure(
    "dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": quad_degree}
)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
He = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=quad_degree)
Te = basix.ufl.blocked_element(He, shape=(mesh.geometry.dim, mesh.geometry.dim), symmetry=True)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Tf = dolfinx.fem.functionspace(mesh, Te)
Hf = dolfinx.fem.functionspace(mesh, He)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")  # displacement
P = dolfinx.fem.Function(Tf, name="P")  # plastic strain
h = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening
B = dolfinx.fem.Function(Tf, name="B")  # kinematic hardening

u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
P0 = dolfinx.fem.Function(Tf, name="P0")
h0 = dolfinx.fem.Function(Hf, name="h0")
B0 = dolfinx.fem.Function(Tf, name="B0")

S0 = dolfinx.fem.Function(Tf, name="S0")  # stress, previous converged solution (load step)

u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, defines state at boundary

Po = dolfinx.fem.Function(
    dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="P"
)  # for output
Bo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="B")
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="S")
ho = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0)), name="h")

# clone to be distinct from Tf
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Tf, Hf, Tf.clone()))
δu, δP, δh, δB = δm

# Define state and variation of state as (ordered) list of functions
m = [u, P, h, B]


def rJ2(A):
    """Square root of J2 invariant of tensor A"""
    J2 = 1 / 2 * ufl.inner(A, A)
    rJ2 = ufl.sqrt(J2)
    return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


# Configuration gradient
I = ufl.Identity(3)  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
E = 1 / 2 * (F.T * F - I)  # E = E(F), total Green-Lagrange strain
E_el = E - P  # E_el = E - P, elastic strain = total strain - plastic strain

# Stress
S = 2 * mu * E_el + la * ufl.tr(E_el) * I  # S = S(E_el), PK2, St.Venant-Kirchhoff

# Wrap variable around expression (for diff)
S, B, h = ufl.variable(S), ufl.variable(B), ufl.variable(h)

# Yield function
f = ufl.sqrt(3) * rJ2(ufl.dev(S - B)) - (Sy + h)  # von Mises criterion (J2), with hardening

# Plastic potential
g = f

# Derivative of plastic potential wrt stress
dgdS = ufl.diff(g, S)

# Total differential of yield function, used for checks only
df = (
    +ufl.inner(ufl.diff(f, S), S - S0)
    + ufl.inner(ufl.diff(f, h), h - h0)
    + ufl.inner(ufl.diff(f, B), B - B0)
)

# Unwrap expression from variable
S, B, h = S.expression(), B.expression(), h.expression()

# Variation of Green-Lagrange strain
δE = ufl.derivative(E, m, δm)

# Plastic multiplier (J2 plasticity: closed-form solution for return-map)
dλ = ufl.max_value(f, 0)  # ppos = MacAuley bracket

# Weak form (as one-form)
form = (
    ufl.inner(δE, S) * dx
    + ufl.inner(δP, (P - P0) - dλ * dgdS) * dx
    + ufl.inner(δh, (h - h0) - dλ * bh * (qh * 1.00 - h)) * dx
    + ufl.inner(δB, (B - B0) - dλ * bb * (qb * dgdS - B)) * dx
)

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "l2"
opts["snes_linesearch_max_it"] = 4
opts["snes_linesearch_monitor"] = ""
opts["snes_atol"] = 1.0e-09
opts["snes_rtol"] = 1.0e-06
opts["snes_stol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"  # NOTE: this monolithic formulation is not symmetric
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_1_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_1)
surface_2_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_2)

# Book-keeping of results
results: dict[str, list[float]] = {"E": [], "S": [], "P": [], "μ": []}

# Set up load steps
K = 10  # number of steps per load phase
Z = 2  # number of cycles
load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
cycle = np.concatenate((load, unload, -load, -unload))
cycles = np.concatenate([cycle] * Z)

# Process load steps
for step, factor in enumerate(cycles):
    # Set current load factor
    μ.value = factor

    dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    # Update values for given boundary displacement
    u_.interpolate(u_bar)

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(u_, surface_1_dofs_Uf),  # disp left
        dolfinx.fem.dirichletbc(u_, surface_2_dofs_Uf),  # disp right
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Post-process data
    dxg = dx(domain_gauge)
    V = dolfiny.expression.assemble(1.0, dxg)
    n = ufl.as_vector([1, 0, 0])
    results["E"].append(dolfiny.expression.assemble(ufl.dot(E * n, n), dxg) / V)
    results["S"].append(dolfiny.expression.assemble(ufl.dot(S * n, n), dxg) / V)
    results["P"].append(dolfiny.expression.assemble(ufl.dot(P * n, n), dxg) / V)
    results["μ"].append(factor)

    # Basic consistency checks
    assert dolfiny.expression.assemble(dλ * df, dxg) / V < 1.0e-03, "|| dλ*df || != 0.0"
    assert dolfiny.expression.assemble(dλ * f, dxg) / V < 1.0e-05, "|| dλ*df || != 0.0"

    # Fix: 2nd order tetrahedron
    # mesh.geometry.cmap.non_affine_atol = 1.0e-8
    # mesh.geometry.cmap.non_affine_max_its = 20

    # Write output
    ofile.write_function(u, step)

    # Interpolate and write output
    dolfiny.interpolation.interpolate(P, Po)
    dolfiny.interpolation.interpolate(B, Bo)
    dolfiny.interpolation.interpolate(S, So)
    dolfiny.interpolation.interpolate(h, ho)
    ofile.write_function(Po, step)
    ofile.write_function(Bo, step)
    ofile.write_function(So, step)
    ofile.write_function(ho, step)

    # Store stress state
    dolfiny.interpolation.interpolate(S, S0)

    # Store primal states
    for source, target in zip([u, P, h, B], [u0, P0, h0, B0]):
        with source.x.petsc_vec.localForm() as locs, target.x.petsc_vec.localForm() as loct:
            locs.copy(loct)

ofile.close()

# Post-process results

fig, ax1 = plt.subplots()
ax1.set_title("Rate-independent plasticity: $J_2$, monolithic formulation, 3D", fontsize=12)
ax1.set_xlabel(r"volume-averaged strain $\frac{1}{V}\int n^T E n \, dV$ [-]", fontsize=12)
ax1.set_ylabel(r"volume-averaged stress $\frac{1}{V}\int n^T S n \, dV$ [GPa]", fontsize=12)
ax1.grid(linewidth=0.25)
fig.tight_layout()

E = np.array(results["E"])
S = np.array(results["S"])

# stress-strain curve
ax1.plot(
    E,
    S,
    color="tab:blue",
    linestyle="-",
    linewidth=1.0,
    markersize=4.0,
    marker=".",
    label=r"$S-E$ curve",
)

ax1.legend(loc="lower right")
ax1.ticklabel_format(style="sci", scilimits=(-2, -2), axis="x")
fig.savefig(f"{name}.pdf")
