#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_tube3d_gmshapi as mg
import numpy as np
import plot_tube3d_pyvista as pl

import dolfiny

# Basic settings
name = "solid_elasticity_classic"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
r, t, h = 0.04, 0.01, 0.1  # [m]
nr, nt, nh = 16, 5, 8

# Create the regular mesh of a tube with given dimensions
gmsh_model, tdim = mg.mesh_tube3d_gmshapi(name, r, t, h, nr, nt, nh, do_quads=True, order=2)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_lower = interfaces_keys["surface_lower"]
surface_upper = interfaces_keys["surface_upper"]

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces, metadata={"quadrature_degree": 4})

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf))
(δu,) = δm

# Define state as (ordered) list of functions
m = [u]

# Functions for output / visualisation
vorder = mesh.geometry.cmap.degree
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3,))), name="u")  # type: ignore
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")  # type: ignore

# Kinematics
F = ufl.Identity(3) + ufl.grad(u)

# Strain measure: Cauchy strain tensor
C = F.T * F
C = ufl.variable(C)

# Variation of strain measure
δC = ufl.derivative(C, m, δm)

# Elasticity parameters
E = dolfinx.fem.Constant(mesh, scalar(1.0))  # [MPa]
nu = dolfinx.fem.Constant(mesh, scalar(0.4))  # [-]
mu = E / (2 * (1 + nu))  # type: ignore
la = E * nu / ((1 + nu) * (1 - 2 * nu))  # type: ignore

# Define boundary stress vector (torque at upper face)
x0 = ufl.SpatialCoordinate(mesh)
n0 = ufl.FacetNormal(mesh)
λ = dolfinx.fem.Constant(mesh, scalar(0.0))
t = ufl.cross(x0 - ufl.as_vector([0.0, 0.0, h]), n0) * 4 * λ  # [N/m^2]


def strain_energy(i1, i2, i3):
    """Strain energy function
    i1, i2, i3: principal invariants of the Cauchy-Green tensor
    """
    # Determinant of configuration gradient F
    J = ufl.sqrt(i3)
    #
    # Classical St. Venant-Kirchhoff
    # Ψ = la / 8 * (i1 - 3) ** 2 + mu / 4 * ((i1 - 3) ** 2 + 4 * (i1 - 3) - 2 * (i2 - 3))
    # Modified St. Venant-Kirchhoff
    # Ψ = la / 2 * (ufl.ln(J)) ** 2 + mu / 4 * ((i1 - 3) ** 2 + 4 * (i1 - 3) - 2 * (i2 - 3))
    # Compressible neo-Hooke
    Ψ = mu / 2 * (i1 - 3 - 2 * ufl.ln(J)) + la / 2 * (J - 1) ** 2
    # Compressible Mooney-Rivlin (beta = 0)
    # Ψ = mu / 4 * (i1 - 3) + mu / 4 * (i2 - 3) - mu * ufl.ln(J) + la / 2 * (J - 1) ** 2
    #
    return Ψ


# Invariants (of C)
i1, i2, i3 = dolfiny.invariants.invariants_principal(C)
# Material model (isotropic)
Ψ = strain_energy(i1, i2, i3)
# Stress measure
S = 2 * ufl.diff(Ψ, C)

# von Mises stress (output)
svm = ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S)))

# Weak form
form = -0.5 * ufl.inner(δC, S) * dx + ufl.inner(δu, t) * ds(surface_upper)

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

# FFCx options
jit_options = dict(cffi_extra_compile_args=["-g0"])

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix=name, jit_options=jit_options)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
b_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_lower)

# Set/update boundary conditions
problem.bcs = [
    dolfinx.fem.dirichletbc(u_, b_dofs_Uf),  # u lower face
]

# Apply external force via load stepping
for λk in np.linspace(0.0, 1.0, 10 + 1)[1:]:
    # Set load factor
    λ.value = λk
    dolfiny.utils.pprint(f"\n*** Load factor λ = {λk:.4f} \n")

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Assert symmetry of operator
    assert dolfiny.la.is_symmetric(problem.J)

# Interpolate for output purposes
dolfiny.interpolation.interpolate(u, uo)
dolfiny.interpolation.interpolate(svm, so)

# Write results to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(uo)
    ofile.write_function(so)

# Visualise
pl.plot_tube3d_pyvista(name)
