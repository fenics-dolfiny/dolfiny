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

import dolfiny

# Basic settings
name = "beam_curved_finitestrain_bzero"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # beam length
nodes = 8 * 4  # number of nodes
p = 2  # physics: polynomial order
q = 2  # geometry: polynomial order

# Create the regular mesh of a curve with given dimensions
gmsh_model, tdim = mg.mesh_curve3d_gmshapi(name, shape="f_arc", L=L, nL=nodes, order=q)

# Get mesh and meshtags
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
beg = mesh_data.physical_groups["beg"].tag
end = mesh_data.physical_groups["end"].tag

# Structure: section geometry
b = 1.0  # [m]
h = L / 500  # [m]
area = b * h  # [m^2]
I = b * h**3 / 12  # [m^4]  # noqa: E741

# Structure: material parameters
n = 0  # [-] Poisson ratio
E = 1.0e8  # [N/m^2] elasticity modulus
lamé_λ = E * n / (1 + n) / (1 - 2 * n)  # Lamé constant λ
lamé_μ = E / 2 / (1 + n)  # Lamé constant μ


def S(E):
    """
    Stress as function of strain from strain energy function
    """
    E = ufl.variable(E)
    W = lamé_μ * ufl.inner(E, E) + lamé_λ / 2 * ufl.tr(E) ** 2  # Saint-Venant Kirchhoff
    S = ufl.diff(W, E)
    return S


# Structure: shear correction factor, see Cowper (1966)
sc_fac = 10 * (1 + n) / (12 + 11 * n)

# Structure: load parameters
μ = dolfinx.fem.Constant(mesh, scalar(1.0))  # load factor

p_x = μ * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
p_z = μ * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))
m_y = μ * dolfinx.fem.Constant(mesh, scalar(1.0 * 0))

F_x = μ * dolfinx.fem.Constant(mesh, scalar((2.0 * np.pi / L) ** 2 * E * I * 0))  # prescr F_x: 2, 4
F_z = μ * dolfinx.fem.Constant(mesh, scalar((0.5 * np.pi / L) ** 2 * E * I * 0))  # prescr F_z: 4, 8
M_y = μ * dolfinx.fem.Constant(mesh, scalar((2.0 * np.pi / L) ** 1 * E * I * 1))  # prescr M_y: 1, 2

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

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
gξ = ufl.cross(gs, gη)  # normal vector (gdim x 1)
# Unit tangent basis
gs /= ufl.sqrt(ufl.dot(gs, gs))
gη /= ufl.sqrt(ufl.dot(gη, gη))
gξ /= ufl.sqrt(ufl.dot(gξ, gξ))

# ----------------------------------------------------------------------------

# Orthogonal projection operator (assumes sufficient geometry approximation)
P = ufl.Identity(mesh.geometry.dim) - ufl.outer(gξ, gξ)

# Thickness variable
X = dolfinx.fem.functionspace(mesh, ("DP", q))
ξ = dolfinx.fem.Function(X, name="ξ")

# Undeformed configuration: director d0 and placement b0
d0 = gξ  # normal of manifold mesh
b0 = x0 + ξ * d0

# Deformed configuration: director d and placement b, assumed kinematics, uses rotation matrix
d = ufl.as_matrix([[ufl.cos(r), 0, ufl.sin(r)], [0, 1, 0], [-ufl.sin(r), 0, ufl.cos(r)]]) * d0
b = x0 + ufl.as_vector([u, 0, w]) + ξ * d

# Configuration gradient, undeformed configuration
J0 = ufl.grad(b0) - ufl.outer(d0, d0)  # = P * ufl.grad(x0) + ufl.grad(ξ * d0)
J0 = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(J0)
J0 = ufl.algorithms.apply_derivatives.apply_derivatives(J0)
J0 = ufl.replace(J0, {ufl.grad(ξ): d0})

# Configuration gradient, deformed configuration
J = ufl.grad(b) - ufl.outer(
    d0, d0
)  # = P * ufl.grad(x0) + ufl.grad(ufl.as_vector([u, 0, w]) + ξ * d)
J = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(J)
J = ufl.algorithms.apply_derivatives.apply_derivatives(J)
J = ufl.replace(J, {ufl.grad(ξ): d0})

# Green-Lagrange strains (total): determined by deformation kinematics
E_total = 1 / 2 * (J.T * J - J0.T * J0)

# Green-Lagrange strains (elastic): E_total = E_elast + E_presc
E = E_elast = E_total

# Membrane strain
Em = P * ufl.replace(E, {ξ: 0.0}) * P

# Bending strain
Eb = ufl.diff(E, ξ)
Eb = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(Eb)
Eb = ufl.algorithms.apply_derivatives.apply_derivatives(Eb)
Eb = P * ufl.replace(Eb, {ξ: 0.0}) * P

# Shear strain
Es = ufl.replace(E, {ξ: 0.0}) - P * ufl.replace(E, {ξ: 0.0}) * P

# Variation of elastic Green-Lagrange strains
δEm = ufl.derivative(Em, m, δm)
δEs = ufl.derivative(Es, m, δm)
δEb = ufl.derivative(Eb, m, δm)

# Stress resultant tensors
N = S(Em) * area
T = S(Es) * area * sc_fac
M = S(Eb) * I

# Partial selective reduced integration of membrane/shear virtual work, see Arnold/Brezzi (1997)
A = dolfinx.fem.functionspace(mesh, ("DP", 0))
α = dolfinx.fem.Function(A)
dolfiny.interpolation.interpolate(h**2 / ufl.JacobianDeterminant(mesh), α)

# Weak form: components (as one-form)
form = (
    -ufl.inner(δEm, N) * α * dx
    - ufl.inner(δEm, N) * (1 - α) * dx(metadata={"quadrature_degree": p * (p - 1)})
    - ufl.inner(δEs, T) * α * dx
    - ufl.inner(δEs, T) * (1 - α) * dx(metadata={"quadrature_degree": p * (p - 1)})
    - ufl.inner(δEb, M) * dx
    + δu * p_x * dx
    + δw * p_z * dx
    + δr * m_y * dx
    + δu * F_x * ds(end)
    + δw * F_z * ds(end)
    + δr * M_y * ds(end)
)

# Optional: linearise weak form
# form = dolfiny.expression.linearise(form, m)  # linearise around zero state

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
if q <= 2:
    ofile.write_mesh_data(mesh_data)

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
opts["mat_mumps_cntl_1"] = 0.0

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix="beam")

# Identify dofs of function spaces associated with tagged interfaces/boundaries
beg_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, mesh_data.facet_tags, beg)
beg_dofs_Wf = dolfiny.mesh.locate_dofs_topological(Wf, mesh_data.facet_tags, beg)
beg_dofs_Rf = dolfiny.mesh.locate_dofs_topological(Rf, mesh_data.facet_tags, beg)

# Create custom plotter (via matplotlib)
plotter = pp.Plotter(
    f"{name}.pdf", r"finite strain beam (1st order shear, displacement-based, on $\mathcal{B}_{0}$)"
)

# Create vector function space and vector function for writing the displacement vector
Z = dolfinx.fem.functionspace(mesh, ("CG", p, (mesh.geometry.dim,)))
z = dolfinx.fem.Function(Z)

# Process load steps
for factor in np.linspace(0, 1, num=20 + 1):
    # Set current load factor
    μ.value = factor

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
        ofile.write_function(z, μ.value)

ofile.close()
