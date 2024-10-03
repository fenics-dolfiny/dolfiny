#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_box_onboard as mg
import plot_box_pyvista as pl

import dolfiny

# Basic settings
name = "solid_stressonly"
comm = MPI.COMM_WORLD


# Discretisation settings
e = 8  # elements in each direction
p = 2  # ansatz order

# Get mesh and meshtags
mesh, mts = mg.mesh_box_onboard(e=e, do_quads=True)

# Merge meshtags, see `boundary_keys` for identifiers of outer faces
boundary, boundary_keys = dolfiny.mesh.merge_meshtags(mesh, mts, mesh.topology.dim - 1)

# Material parameters
E = dolfinx.fem.Constant(mesh, scalar(200.0))  # [GPa]
nu = dolfinx.fem.Constant(mesh, scalar(0.25))  # [-]
la = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

# Stabilisation factor
ω = dolfinx.fem.Constant(mesh, scalar(1.01))


def strain_from_displm(u):
    return ufl.sym(ufl.grad(u))


def stress_from_strain(e):
    return 2 * mu * e + la * ufl.tr(e) * ufl.Identity(3)


def strain_from_stress(s):
    return 1 / (2 * mu) * s - la / (2 * mu * (3 * la + 2 * mu)) * ufl.tr(s) * ufl.Identity(3)


# Manufactured solution
x = ufl.SpatialCoordinate(mesh)  # spatial coordinates
n = ufl.FacetNormal(mesh)  # normal

u0_expr = ufl.as_vector([ufl.sin(x[0] + x[1]), ufl.sin(x[1] + x[2]), ufl.sin(x[2] + x[0])])
u0_expr /= 2000

E0_expr = strain_from_displm(u0_expr)
S0_expr = stress_from_strain(E0_expr)


def b(S):
    return -ufl.div(S)  # volume force vector


def T(S):
    return (
        (1 + nu) * ufl.dot(ufl.grad(S), n)
        + ufl.outer(ufl.grad(ufl.tr(S)), n)
        + ufl.inner(ufl.div(S), n) * ufl.Identity(3)
        + (1 + nu) * ω * ufl.outer(ufl.div(S), n)
    )  # boundary stress tensor


# Load
b0 = b(S0_expr)  # volume force vector
T0 = T(S0_expr)  # boundary stress tensor

# Define integration measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundary)

# Define elements
Se = basix.ufl.element("P", mesh.basix_cell(), p, shape=(3, 3), symmetry=True)

# Define function spaces
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
S = dolfinx.fem.Function(Sf, name="S")
S0 = dolfinx.fem.Function(Sf, name="S0")
δS = ufl.TestFunction(Sf)

# Define state as (ordered) list of functions
m, δm = [S], [δS]

# Create other functions: output / visualisation
vorder = mesh.geometry.cmap.degree
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3, 3), True)), name="S")
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3,))), name="u")
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")

# Boundaries (via mesh tags)
dirichlet = [0, 2, 4]  # faces = {x0 = xmin, x1 = xmin, x2 = xmin}
neumann = list(set(boundary_keys.values()) - set(dirichlet))  # complement to dirichlet

# Form, stress-based, see Eq. 6.14 in https://doi.org/10.1016/j.ijsolstr.2024.112808
form = (
    (1 + nu) * ufl.inner(ufl.grad(δS), ufl.grad(S)) * dx
    - (1 + nu) * 2 * ufl.inner(δS, ufl.sym(ufl.grad(b0))) * dx
    + ufl.inner(ufl.div(δS), ufl.grad(ufl.tr(S))) * dx
    + ufl.inner(ufl.grad(ufl.tr(δS)), ufl.div(S)) * dx
    - (1 + nu**2) / (1 - nu) * ufl.tr(δS) * ufl.div(b0) * dx
    + (1 + nu) * ω * ufl.inner(ufl.div(δS), ufl.div(S)) * dx
    - (1 + nu) * ω * ufl.inner(δS, ufl.sym(ufl.grad(b0))) * dx
    - sum(ufl.inner(δS, T0) * ds(k) for k in neumann)
)

# Overall form (as list of forms)
forms = dolfiny.function.extract_blocks(form, δm)

# Interpolate expression
dolfiny.interpolation.interpolate(S0_expr, S0)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
bcsdofs_Sf = dolfiny.mesh.locate_dofs_topological(Sf, boundary, dirichlet)
bcs = [dolfinx.fem.dirichletbc(S0, bcsdofs_Sf)]

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 0.99
opts["snes_max_it"] = 1
opts["ksp_type"] = "cg"
opts["ksp_atol"] = 1.0e-12
opts["ksp_rtol"] = 1.0e-10
opts["pc_type"] = "bjacobi"

# Create nonlinear problem: SNES (convenience choice here)
problem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, bcs, prefix=name)

# Solve problem
problem.solve()

# Assert convergence of solver
problem.status(verbose=True, error_on_failure=False)

# Assert symmetry of operator
assert dolfiny.la.is_symmetric(problem.J)

# Compute L2 relative stress error
error_expr = ufl.sqrt(ufl.inner(S - S0_expr, S - S0_expr)) / ufl.sqrt(ufl.inner(S0_expr, S0_expr))
error_comp = dolfiny.expression.assemble(error_expr, dx)
dolfiny.utils.pprint(f"* L2 relative stress error = {error_comp:.3e}")

# Define scalar measure (output)
s_title, s_range, s = (
    "von Mises stress [GPa]",
    [0.0, 0.16],
    ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S))),
    # "mean stress [GPa]",
    # [-0.16, +0.16],
    # ufl.tr(S) / 3,
    # "L2 relative stress error",
    # None,
    # ufl.sqrt(ufl.inner(S - S0_expr, S - S0_expr)) / ufl.sqrt(ufl.inner(S0_expr, S0_expr)),
)

# Interpolate
dolfiny.interpolation.interpolate(u0_expr, uo)
dolfiny.interpolation.interpolate(S, So)
dolfiny.interpolation.interpolate(s, so)

# Write results to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(uo)
    ofile.write_function(So)
    ofile.write_function(so)

# Visualise
pl.plot_box_pyvista(name, options=dict(s_title=s_title, s_range=s_range))
