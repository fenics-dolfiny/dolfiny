# %% [markdown]
# # Dimensional Analysis of the Navier-Stokes Equations

# %%
import argparse

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import numpy as np
import sympy.physics.units as syu

import dolfiny
import dolfiny.ufl_utils
from dolfiny.units import Quantity

# Parse command line arguments
parser = argparse.ArgumentParser(description="Navier-Stokes dimensional analysis")
parser.add_argument(
    "--p-ref", type=float, default=5000.0, help="Reference pressure value (default: 5000.0)"
)
args = parser.parse_args()

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
Ve = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.topology.dim,))
Pe = basix.ufl.element("P", mesh.basix_cell(), 1)
Vf = dolfinx.fem.functionspace(mesh, Ve)
Pf = dolfinx.fem.functionspace(mesh, Pe)

v = dolfinx.fem.Function(Vf, name="v")
v0 = dolfinx.fem.Function(Vf, name="v0")
p = dolfinx.fem.Function(Pf, name="p")
δv, δp = ufl.TestFunctions(ufl.MixedFunctionSpace(Vf, Pf))

b = dolfinx.fem.Constant(mesh, (0.0, -1.0))
n = dolfinx.fem.Constant(mesh, scalar(1))  # for time step dt = t_ref / n

# %%
# Define reference quantities with units
nu = Quantity(mesh, 1000, syu.millimeter**2 / syu.second, "nu")
rho = Quantity(mesh, 5000, syu.kilogram / syu.m**3, "rho")
l_ref = Quantity(mesh, 1, syu.meter, "l_ref")
t_ref = Quantity(mesh, 1 / 60, syu.minute, "t_ref")
v_ref = Quantity(mesh, 1, syu.meter / syu.second, "v_ref")
p_ref = Quantity(mesh, args.p_ref, syu.pascal, "p_ref")
g_ref = Quantity(mesh, 10, syu.meter / syu.second**2, "g_ref")
# quantities = [nu, rho, l_ref, t_ref, v_ref, p_ref, g_ref]
quantities = [v_ref, l_ref, rho, nu, g_ref, p_ref, t_ref]  # order -> 1 / Re, Fr, Eu, St

# %%
if MPI.COMM_WORLD.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

# %%
mapping = {
    mesh.ufl_domain(): l_ref,
    v: v_ref * v,
    v0: v_ref * v0,
    p: p_ref * p,
    δv: v_ref * δv,
    δp: p_ref * δp,
}


# %%
def D(u_expr):
    """Strain rate tensor."""
    return ufl.sym(ufl.grad(u_expr))


terms = {
    "unsteady": ufl.inner(δv, rho * (v - v0) / (t_ref / n)) * ufl.dx,
    "convection": ufl.inner(δv, rho * ufl.dot(v, ufl.grad(v))) * ufl.dx,
    "viscous": ufl.inner(D(δv), 2 * rho * nu * D(v)) * ufl.dx,
    "pressure": -ufl.inner(ufl.div(δv), p) * ufl.dx,
    "force": -ufl.inner(δv, rho * g_ref * b) * ufl.dx,
    "incompressibility": δp * ufl.div(v) * ufl.dx,
    "press_diag": dolfinx.fem.Constant(mesh, scalar(0.0))
    * v_ref
    / l_ref
    / p_ref
    * ufl.inner(δp, p)
    * ufl.dx,
}

# %%

# Few dimensional sanity checks
dimsys = syu.si.SI.get_dimension_system()
assert dimsys.equivalent_dims(dolfiny.units.get_dimension(D(v), quantities, mapping), 1 / syu.time)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(rho * (v - v0) / (t_ref / n), quantities, mapping),
    syu.mass / syu.length**3 * syu.length / syu.time**2,
)
assert dolfiny.units.get_dimension(D(v), quantities, mapping) == 1 / syu.time

form = sum(terms.values())
form_dim = dolfiny.units.get_dimension(form, quantities, mapping)
assert dimsys.equivalent_dims(form_dim, syu.power / syu.length)

convective_dim = dolfiny.units.get_dimension(rho * ufl.dot(v, ufl.grad(v)), quantities, mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(convective_dim, syu.force / syu.length**3)

terms_fact = dolfiny.units.factorize(terms, quantities, mode="factorize", mapping=mapping)
assert isinstance(terms_fact, dict)

reference_term = "convection"
terms_norm = dolfiny.units.normalize(terms_fact, reference_term, quantities)

# %%
form_nondimensional = sum(terms_norm.values(), ufl.form.Zero())

# %%


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


# No-slip condition on boundaries where x = 0, x = 1, and y = 0
noslip = np.zeros(mesh.geometry.dim, dtype=scalar)
facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, noslip_boundary)
bc0 = dolfinx.fem.dirichletbc(noslip, dolfinx.fem.locate_dofs_topological(Vf, 1, facets), Vf)

# Driving (lid) velocity condition on top boundary (y = 1)
lid_velocity = dolfinx.fem.Function(Vf)
lid_velocity.interpolate(lid_velocity_expression)
facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lid)
bc1 = dolfinx.fem.dirichletbc(lid_velocity, dolfinx.fem.locate_dofs_topological(Vf, 1, facets))

# Pin pressure to zero at single dof
dof0 = dolfinx.fem.locate_dofs_geometrical(
    Pf, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
)
bc2 = dolfinx.fem.dirichletbc(scalar(0.0), dof0, Pf)

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1, bc2]

form_nondimensional_blocks = ufl.extract_blocks(form_nondimensional)  # type: ignore

opts = PETSc.Options("ns")  # type: ignore[attr-defined]
opts["snes_max_it"] = 5
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

problem_gmres = dolfiny.snesproblem.SNESProblem(
    form_nondimensional_blocks, [v, p], bcs=bcs, prefix="ns", nest=False
)

(u_gmres, p_gmres) = problem_gmres.solve()
