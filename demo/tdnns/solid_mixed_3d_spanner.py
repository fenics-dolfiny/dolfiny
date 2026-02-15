#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_spanner_gmshapi as mg
import plot_spanner_pyvista as pl

import dolfiny

# Basic settings
name = "solid_mixed_3d_spanner"
comm = MPI.COMM_WORLD

# Create the mesh of a spanner [m]
gmsh_model, tdim = mg.mesh_spanner_gmshapi(name)

# Get mesh and meshtags
partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none, 2)  # type: ignore
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0, partitioner=partitioner)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
surface_flats = mesh_data.physical_groups["surface_flats"].tag
surface_crown = mesh_data.physical_groups["surface_crown"].tag

# Solid material parameters, steel S235: E=210 [GPa], nue=0.30 [-], fy = 0.235 [GPa]
mu = dolfinx.fem.Constant(mesh, scalar(81.0))  # GPa
la = dolfinx.fem.Constant(mesh, scalar(121.0))  # GPa

# Load
g = dolfinx.fem.Constant(mesh, [0.0, 0.0, 0.0])  # volume force vector
t = dolfinx.fem.Constant(mesh, [5.0e-4, 0.0, 0.0])  # boundary stress vector

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(3,))
Se = basix.ufl.element("Regge", mesh.basix_cell(), 1)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
S = dolfinx.fem.Function(Sf, name="S")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions
S_ = dolfinx.fem.Function(Sf, name="S_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Sf))
δu, δS = δm

# Define state as (ordered) list of functions
m = [u, S]

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


# Form, dual mixed form
form = (
    -ufl.inner(Eu(δu), S) * dx
    + ufl.inner(δu, g) * dx
    + ufl.dot(δu, t) * ds(surface_crown)
    + ufl.inner(δS, Es(S) - Eu(u)) * dx
)

form += (
    dolfinx.fem.Constant(mesh, scalar(0.0)) * ufl.inner(δu, u) * dx
)  # ensure zero block diagonal for bc

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_flats_dofs_Uf = dolfiny.mesh.locate_dofs_topological(
    Uf, mesh_data.facet_tags, surface_flats
)

# Define boundary conditions
bcs = [
    dolfinx.fem.dirichletbc(u_, surface_flats_dofs_Uf),  # u zero on flats
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
problem = dolfiny.snesproblem.SNESProblem(forms, m, bcs, prefix=name)

# Solve problem
problem.solve()

# Assert convergence of solver
problem.status(verbose=True, error_on_failure=True)

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
