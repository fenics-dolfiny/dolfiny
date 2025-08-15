#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_curve3d_gmshapi as mg
import numpy as np
import postprocess_matplotlib as pp
import sympy.physics.units as syu

import dolfiny
from dolfiny.units import Quantity

# Basic settings
name = "beam_curved_finitestrain_bstar_reissner"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # beam length
nodes = 8 * 4  # number of nodes
p = 2  # physics: polynomial order
q = 2  # geometry: polynomial order

# Create the regular mesh of a curve with given dimensions
gmsh_model, tdim = mg.mesh_curve3d_gmshapi(name, shape="f_arc", L=L, nL=nodes, order=q)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
beg = interfaces_keys["beg"]
end = interfaces_keys["end"]

# Structure: section geometry
b = 1.0
h = L / 500
area = Quantity(mesh, b * h, syu.meter**2, "A")
I = Quantity(mesh, b * h**3 / 12, syu.meter**4, "I")  # noqa: E741

# Structure: material parameters
n = 0  # [-] Poisson ratio
E = Quantity(mesh, 100, syu.mega * syu.pascal, "E")
la = Quantity(mesh, E.scale * n / (1 + n) / (1 - 2 * n), syu.mega * syu.pascal, "lambda")
mu = Quantity(mesh, E.scale / 2 / (1 + n), syu.mega * syu.pascal, "mu")

# Structure: shear correction factor, see Cowper (1966)
sc_fac = 10 * (1 + n) / (12 + 11 * n)

# Reference quantities for dimensional analysis
L_ref = Quantity(mesh, L, syu.meter, "L_ref")  # Reference length
F_ref = Quantity(mesh, 1, syu.newton, "F_ref")  # Reference force


def s_lambda(e):
    e = ufl.variable(e)
    W = la / 2 * e**2
    s = ufl.diff(W, e)
    return s


def s_mu(e):
    e = ufl.variable(e)
    W = mu * e * e
    s = ufl.diff(W, e)
    return s


# Structure: load parameters
μ = dolfinx.fem.Constant(mesh, scalar(1.0))  # load factor

p_x = μ * F_ref / L_ref * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
p_z = μ * F_ref / L_ref * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
m_y = μ * F_ref * L_ref / L_ref * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))

F_x = μ * (2.0 * np.pi / L_ref) ** 2 * E * I * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
F_z = μ * (0.5 * np.pi / L_ref) ** 2 * E * I * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
M_y = μ * (2.0 * np.pi / L_ref) ** 1 * E * I * 1

dimsys = syu.si.SI.get_dimension_system()

assert dimsys.equivalent_dims(dolfiny.units.get_dimension(F_x, [L_ref, E, I]), syu.force)

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), degree=p)
We = basix.ufl.element("P", mesh.basix_cell(), degree=p)
Re = basix.ufl.element("P", mesh.basix_cell(), degree=p)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Wf = dolfinx.fem.functionspace(mesh, We)
Rf = dolfinx.fem.functionspace(mesh, Re)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
w = dolfinx.fem.Function(Wf, name="w")
r = dolfinx.fem.Function(Rf, name="r")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions
w_ = dolfinx.fem.Function(Wf, name="w_")
r_ = dolfinx.fem.Function(Rf, name="r_")

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Wf, Rf))
δu, δw, δr = δm

# Define state as (ordered) list of functions
m = [u, w, r]

# GEOMETRY -------------------------------------------------------------------
# Coordinates of undeformed configuration
x0 = ufl.SpatialCoordinate(mesh)

# Jacobi matrix of map reference -> undeformed
J0 = ufl.geometry.Jacobian(mesh)
# Tangent basis
gs = J0[:, 0]
gη = ufl.as_vector([0, 1, 0])  # unit vector e_y (assume curve in x-z plane)
gξ = ufl.cross(gs, gη)  # unit normal vector (gdim x 1)
# Unit tangent basis
gs /= ufl.sqrt(ufl.dot(gs, gs))
gη /= ufl.sqrt(ufl.dot(gη, gη))
gξ /= ufl.sqrt(ufl.dot(gξ, gξ))

# Contravariant basis
K0 = ufl.geometry.JacobianInverse(mesh).T  # type: ignore
# Curvature tensor (tdim x tdim)
B0 = -ufl.dot(ufl.dot(ufl.grad(gξ), J0).T, K0)  # = ufl.dot(gξ, ufl.dot(ufl.grad(K0), J0))
# ----------------------------------------------------------------------------


def GRAD(u):
    """DERIVATIVE with respect to arc-length coordinate s of straight reference configuration:
    du/ds = du/dx * dx/dr * dr/ds
    """
    return ufl.dot(ufl.grad(u), J0[:, 0]) * 1 / ufl.geometry.JacobianDeterminant(mesh)


# Undeformed configuration: stretch (at the principal axis)
λ0 = ufl.sqrt(ufl.dot(GRAD(x0), GRAD(x0)))  # from geometry (!= 1)
# Undeformed configuration: curvature
κ0 = -B0[0, 0]  # from curvature tensor B0i

# Deformed configuration: stretch components (at the principal axis)
λs = (1.0 + GRAD(x0[0]) * GRAD(u) + GRAD(x0[2]) * GRAD(w)) * ufl.cos(r) + (
    GRAD(x0[2]) * GRAD(u) - GRAD(x0[0]) * GRAD(w)
) * ufl.sin(r)
λξ = (1.0 + GRAD(x0[0]) * GRAD(u) + GRAD(x0[2]) * GRAD(w)) * ufl.sin(r) - (
    GRAD(x0[2]) * GRAD(u) - GRAD(x0[0]) * GRAD(w)
) * ufl.cos(r)
# Deformed configuration: curvature
κ = GRAD(r)

# Reissner strains (total): determined by deformation kinematics
ε_total = λs - λ0
γ_total = λξ
κ_total = κ

# Reissner strains (elastic): e_total = e_elast + e_presc
ε = ε_elast = ε_total
γ = γ_elast = γ_total
κ = κ_elast = κ_total

# Variation of elastic Green-Lagrange strains
δε = ufl.derivative(ε, m, δm)
δγ = ufl.derivative(γ, m, δm)
δκ = ufl.derivative(κ, m, δm)

# Stress resultants
N_bulk = s_lambda(ε) * area
N_shear = s_mu(ε) * area

T_bulk = s_lambda(γ) * area * sc_fac
T_shear = s_mu(γ) * area * sc_fac

M_bulk = s_lambda(κ) * I
M_shear = s_mu(κ) * I

# Partial selective reduced integration of membrane/shear virtual work, see Arnold/Brezzi (1997)
A = dolfinx.fem.functionspace(mesh, ("DP", 0))
α = dolfinx.fem.Function(A)
dolfiny.interpolation.interpolate(h**2 / ufl.JacobianDeterminant(mesh), α)

# Define mapping for dimensional analysis
mapping = {
    mesh.ufl_domain(): L_ref,
    u: L_ref * u,
    w: L_ref * w,
    r: r,  # rotation is dimensionless (radians)
    δu: L_ref * δu,
    δw: L_ref * δw,
    δr: δr,
}

quantities = [area, I, L_ref, F_ref, mu, la, E]

if comm.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

# Define beam terms for dimensional analysis
terms = {
    "membrane_lambda": -ufl.inner(δε, N_bulk) * α * dx,
    "membrane_lambda_reduced": -ufl.inner(δε, N_bulk)
    * (1 - α)
    * dx(metadata={"quadrature_degree": p * (p - 1)}),
    "shear_lambda": -ufl.inner(δγ, T_bulk) * α * dx,
    "shear_lambda_reduced": -ufl.inner(δγ, T_bulk)
    * (1 - α)
    * dx(metadata={"quadrature_degree": p * (p - 1)}),
    "bending_lambda": -ufl.inner(δκ, M_bulk) * dx,
    "membrane_mu": -ufl.inner(δε, N_shear) * α * dx,
    "membrane_mu_reduced": -ufl.inner(δε, N_shear)
    * (1 - α)
    * dx(metadata={"quadrature_degree": p * (p - 1)}),
    "shear_mu": -ufl.inner(δγ, T_shear) * α * dx,
    "shear_mu_reduced": -ufl.inner(δγ, T_shear)
    * (1 - α)
    * dx(metadata={"quadrature_degree": p * (p - 1)}),
    "bending_mu": -ufl.inner(δκ, M_shear) * dx,
    "body_x": δu * p_x * dx,
    "body_z": δw * p_z * dx,
    "body_m": δr * m_y * dx,
    "point_x": δu * F_x * ds(end),
    "point_z": δw * F_z * ds(end),
    "point_m": δr * M_y * ds(end),
}

factorized = dolfiny.units.factorize(terms, quantities, mode="check", mapping=mapping)
assert isinstance(factorized, dict)
reference_term = "bending_mu"
normalized = dolfiny.units.normalize(factorized, reference_term, quantities)
form = sum(normalized.values(), ufl.form.Zero())

# Optional: linearise weak form
# form = dolfiny.expression.linearise(form, m)  # linearise around zero state

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
if q <= 2:
    ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options("beam")  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-07
opts["snes_rtol"] = 1.0e-07
opts["snes_stol"] = 1.0e-06
opts["snes_max_it"] = 60
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix="beam")

# Identify dofs of function spaces associated with tagged interfaces/boundaries
beg_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, beg)
beg_dofs_Wf = dolfiny.mesh.locate_dofs_topological(Wf, interfaces, beg)
beg_dofs_Rf = dolfiny.mesh.locate_dofs_topological(Rf, interfaces, beg)

# Create custom plotter (via matplotlib)
plotter = pp.Plotter(
    f"{name}.pdf", r"finite strain beam (1st order shear, displacement-based, on $\mathcal{B}_{*}$)"
)

# Create vector function space and vector function for writing the displacement vector
Z = dolfinx.fem.functionspace(mesh, ("P", p, (mesh.geometry.dim,)))
z = dolfinx.fem.Function(Z)

# Process load steps
for load_factor in np.linspace(0, 1, num=20 + 1):
    # Set current load factor
    μ.value = load_factor

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(u_, beg_dofs_Uf),  # u beg
        dolfinx.fem.dirichletbc(w_, beg_dofs_Wf),  # w beg
        dolfinx.fem.dirichletbc(r_, beg_dofs_Rf),  # r beg
    ]

    dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Add to plot
    if comm.size == 1:
        plotter.add(mesh, q, m, μ)

    # Write output
    if q <= 2:
        dolfiny.interpolation.interpolate(ufl.as_vector([u, 0, w]), z)
        ofile.write_function(z, float(μ.value))

ofile.close()
