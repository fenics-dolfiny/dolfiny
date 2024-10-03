#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import hdivdiv
import mesh_spanner_gmshapi as mg
import plot_spanner_pyvista as pl

import dolfiny

# Basic settings
name = "solid_tdnns_3d_spanner"
comm = MPI.COMM_WORLD

# Create the mesh of a spanner [m]
gmsh_model, tdim = mg.mesh_spanner_gmshapi(name)

# Get mesh and meshtags
partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, partitioner=partitioner)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_flats = interfaces_keys["surface_flats"]
surface_crown = interfaces_keys["surface_crown"]
surface_other = interfaces_keys["surface_other"]

# Solid material parameters, steel S235: E=210 [GPa], nue=0.30 [-], fy = 0.235 [GPa]
mu = dolfinx.fem.Constant(mesh, scalar(81.0))  # GPa
la = dolfinx.fem.Constant(mesh, scalar(121.0))  # GPa

# Load
g = dolfinx.fem.Constant(mesh, [0.0, 0.0, 0.0])  # volume force vector
t = dolfinx.fem.Constant(mesh, [5.0e-4, 0.0, 0.0])  # boundary stress vector

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)
dS = ufl.Measure("dS", domain=mesh)

# Define elements
Ue = basix.ufl.element("N2E", mesh.basix_cell(), 2)
Se = hdivdiv.create_custom_hdivdiv(mesh.basix_cell(), 2, verbose=not comm.rank)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
S = dolfinx.fem.Function(Sf, name="S")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions
S_ = dolfinx.fem.Function(Sf, name="S_")  # boundary conditions
Sn = dolfinx.fem.Function(Sf, name="Sn")  # boundary conditions, crown

δu = ufl.TestFunction(Uf)
δS = ufl.TestFunction(Sf)

# Define state as (ordered) list of functions
m, δm = [u, S], [δu, δS]

# Create other functions: output / visualisation
vorder = mesh.geometry.cmap.degree
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3, 3), True)), name="S")
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3,))), name="u")
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")


# Strain, kinematically
def Eu(u):
    return ufl.sym(ufl.grad(u))  # linear


# Strain from stress, inverse constitutive
def Es(S):
    return 1 / (2 * mu) * S - la / (2 * mu * (3 * la + 2 * mu)) * ufl.tr(S) * ufl.Identity(3)


# Stress measure, von Mises
def s(S):
    return ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S)))


# Mesh facet normal
n = ufl.FacetNormal(mesh)


# Duality pairing operator
def b(S, u):
    return (
        -ufl.inner(S, Eu(u)) * dx
        + ufl.dot(ufl.dot(S, n), n) * ufl.dot(u, n) * ds
        + ufl.dot(ufl.dot(S("+"), n("+")), n("+")) * ufl.jump(u, n) * dS
    )


# Projector, vector, tangential
def tangential(a):
    return a - ufl.dot(a, n) * n


# Projector, matrix, normal-normal
def normalnormal(A):
    return ufl.dot(ufl.dot(A, n), n)


# Form, dual mixed form, TD-NNS, duality pairing
form = (
    +ufl.inner(δS, Es(S)) * dx
    + b(δS, u)
    + b(S, δu)
    + ufl.dot(δu, g) * dx
    + ufl.dot(tangential(δu), tangential(t)) * ds(surface_crown)
)

form += (
    dolfinx.fem.Constant(mesh, scalar(0.0)) * ufl.inner(δu, u) * dx
)  # ensure zero block diagonal for bc

# Overall form (as list of forms)
forms = dolfiny.function.extract_blocks(form, δm)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_flats_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_flats)
surface_crown_dofs_Sf = dolfiny.mesh.locate_dofs_topological(Sf, interfaces, surface_crown)
surface_other_dofs_Sf = dolfiny.mesh.locate_dofs_topological(Sf, interfaces, surface_other)

# Provide normal-normal stress (boundary stress vector) at crown via projection
dolfiny.projection.project_codimension(ufl.dot(t, n), Sn, normalnormal, interfaces, surface_crown)

# Define boundary conditions
bcs = [
    dolfinx.fem.dirichletbc(u_, surface_flats_dofs_Uf),  # u_t zero on flats (tangentially)
    dolfinx.fem.dirichletbc(S_, surface_other_dofs_Sf),  # t_n zero on other (stress-free)
    dolfinx.fem.dirichletbc(Sn, surface_crown_dofs_Sf),  # t_n non-zero on crown
]

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 0.99
opts["snes_max_it"] = 1
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, bcs, prefix=name)

# Solve problem
problem.solve()

# Assert convergence of solver
problem.status(verbose=True, error_on_failure=False)

# Assert symmetry of operator
assert dolfiny.la.is_symmetric(problem.J)

# Interpolate
dolfiny.interpolation.interpolate(u, uo)
dolfiny.interpolation.interpolate(S, So)
dolfiny.interpolation.interpolate(s(S), so)

# Write results to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(uo)
    ofile.write_function(So)
    ofile.write_function(so)

# Visualise
pl.plot_spanner_pyvista(name)
