#!/usr/bin/env python3

import argparse

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_tube3d_gmshapi as mg
import numpy as np
import plot_tube3d_pyvista as pl
import sympy.physics.units as syu

import dolfiny
from dolfiny.units import Quantity

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Solid elasticity with classic or spectral formulation"
)
parser.add_argument(
    "--formulation",
    choices=["classic", "spectral"],
    default="spectral",
    help="Choose strain formulation: classic (Cauchy strain) or spectral (principal stretches)",
)
args = parser.parse_args()

# Basic settings
name = f"solid_elasticity_{args.formulation}"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
r, t, h = 0.4, 0.1, 1
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
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3,))), name="u")
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")  # for output

# Kinematics
F = ufl.Identity(3) + ufl.grad(u)

# Strain measure: Cauchy strain tensor
C = F.T * F
C = ufl.variable(C)

# Variation of strain measure
δC = ufl.derivative(C, m, δm)

# Formulation-specific strain measures
if args.formulation == "spectral":
    # Strain measure: from Cauchy strain tensor to squares of principal stretches
    c, _ = dolfiny.invariants.eigenstate(C)  # spectral decomposition of C
    c = ufl.as_vector(c)  # squares of principal stretches
    c = ufl.variable(c)
    # Variation of strain measure (squares of principal stretches)
    δc = ufl.derivative(c, m, δm)

# Elasticity parameters
nu = 0.4
E = Quantity(mesh, 1, syu.mega * syu.pascal, "E")  # Young's modulus
mu = Quantity(mesh, E.scale / (2 * (1 + nu)), syu.mega * syu.pascal, "μ")  # shear modulus
λ = Quantity(
    mesh, E.scale * nu / ((1 + nu) * (1 - 2 * nu)), syu.mega * syu.pascal, "λ"
)  # Lamé constant
kappa = Quantity(mesh, λ.scale + 2 / 3 * mu.scale, syu.mega * syu.pascal, "κ")  # Lamé constant

l_ref = Quantity(mesh, 0.1, syu.meter, "l_ref")
t_ref = Quantity(mesh, 1.0, syu.mega * syu.pascal, "t_ref")

# Define boundary stress vector (torque at upper face)
x0 = ufl.SpatialCoordinate(mesh)
n0 = ufl.FacetNormal(mesh)
load_factor = dolfinx.fem.Constant(mesh, scalar(0.0))
d = x0 - l_ref * h * ufl.as_vector([0.0, 0.0, 1.0]) / l_ref
t = ufl.cross(d, n0) * 4 * t_ref * load_factor
t = ufl.as_vector([0.0, 0.0, 1.0]) * t_ref * load_factor


def strain_energy_bulk(i1, i2, i3):
    J = ufl.sqrt(i3)
    return kappa / 2 * (J - 1) ** 2


def strain_energy_shear(i1, i2, i3):
    J = ufl.sqrt(i3)
    return mu / 2 * (i1 - 3 - 2 * ufl.ln(J))


# Invariants (of C)
i1, i2, i3 = dolfiny.invariants.invariants_principal(C)
# Stress measures
S_bulk = 2 * ufl.diff(strain_energy_bulk(i1, i2, i3), C)
S_shear = 2 * ufl.diff(strain_energy_shear(i1, i2, i3), C)

quantities = [l_ref, t_ref, mu, kappa]
quantities = [mu, kappa, l_ref, t_ref]
if comm.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

mapping = {
    mesh.ufl_domain(): l_ref,
    u: l_ref * u,
    δu: l_ref * δu,
}

terms = {
    "int_bulk": -1 / 2 * ufl.inner(δC, S_bulk) * dx,
    "int_shear": -1 / 2 * ufl.inner(δC, S_shear) * dx,
    "external": ufl.inner(δu, t) * ds(surface_upper),
}
factorized = dolfiny.units.factorize(terms, quantities, mode="factorize", mapping=mapping)
assert isinstance(factorized, dict)

dimsys = syu.si.SI.get_dimension_system()
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(terms["int_bulk"], quantities, mapping),
    syu.energy,
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(strain_energy_bulk(i1, i2, i3), quantities, mapping),
    syu.energy * syu.length**-3,
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(S_shear, quantities, mapping),
    syu.pressure,
)

reference_term = "int_bulk"
ref_factor = factorized[reference_term].factor

normalized = dolfiny.units.normalize(factorized, reference_term, quantities)
form = sum(normalized.values(), ufl.form.Zero())

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Options for PETSc backend (using the same options for both formulations)
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_cntl_1"] = 0.0

# FFCx options (formulation-specific)
if args.formulation == "spectral":
    # ARM64-specific optimizations for spectral formulation
    jit_options = dict(
        cffi_extra_compile_args=[
            "-fdisable-rtl-combine",
            "-fno-schedule-insns",
            "-fno-schedule-insns2",
        ]
    )
else:
    # Standard options for classic formulation
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
for lf in np.linspace(0.0, 1.0, 10 + 1)[1:]:
    # Set load factor
    load_factor.value = lf
    dolfiny.utils.pprint(f"\n*** Load factor = {lf:.4f} ({args.formulation} formulation) \n")

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Assert symmetry of operator
    assert dolfiny.la.is_symmetric(problem.J)

# Interpolate for output purposes
dolfiny.interpolation.interpolate(u, uo)

# von Mises stress (output)
S = S_bulk + S_shear
svm = dolfiny.units.transform(ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S))), mapping)
dolfiny.interpolation.interpolate(svm, so)

# Write results to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(uo)
    ofile.write_function(so)

# Visualise
pl.plot_tube3d_pyvista(name)
