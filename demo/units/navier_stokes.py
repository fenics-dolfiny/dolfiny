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
p_ref = Quantity(mesh, 5000, syu.pascal, "p_ref")
g_ref = Quantity(mesh, 10, syu.meter / syu.second**2, "g_ref")
quantities = [nu, rho, l_ref, t_ref, v_ref, p_ref, g_ref]
# quantities = [v_ref, l_ref, rho, nu, g_ref, p_ref, t_ref]  # order -> 1 / Re, Fr, Eu, St

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
bc2 = dolfinx.fem.dirichletbc(
    scalar(0.0), dolfinx.fem.locate_dofs_topological(Pf, 0, np.array([0], dtype=np.int32)), Pf
)

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1, bc2]

form_nondimensional_blocks = ufl.extract_blocks(form_nondimensional)

# Set up options for GMRES
opts = PETSc.Options("ns")  # type: ignore[attr-defined]
opts["snes_max_it"] = 1
opts["ksp_type"] = "gmres"
opts["ksp_gmres_restart"] = 200
opts["ksp_gmres_modifiedgramschmidt"] = True
opts["ksp_max_it"] = 10000
opts["pc_type"] = "none"
opts["ksp_rtol"] = 1e-14

problem_gmres = dolfiny.snesproblem.SNESProblem(
    form_nondimensional_blocks, [v, p], bcs=bcs, prefix="ns", nest=True
)
problem_gmres.verbose["ksp"] = False
problem_gmres.verbose["snes"] = False


def compute_condition_number(J):
    J_aij = J.copy()
    J_aij.convert("aij")
    A_scipy = dolfiny.la.petsc_to_scipy(J_aij)
    return np.linalg.cond(A_scipy.todense())


conds = []
nums_its = []
ps = np.linspace(args.p_ref / 10, args.p_ref * 2, 50)
eus = ps / (rho.value * v_ref.value**2)
for p0 in ps:
    p_ref.scale = p0
    v.x.array[:] = 0.0
    p.x.array[:] = 0.0
    (u_gmres, p_gmres) = problem_gmres.solve()

    cond = compute_condition_number(problem_gmres.J)
    num_its = problem_gmres.snes.ksp.getIterationNumber()
    conds.append(cond)
    nums_its.append(num_its)
    dolfiny.utils.pprint(
        f"GMRES: p_ref = {p0:.2f} Pa, iterations = {num_its}, condition number = {cond:.4g}"
    )

np.savetxt(
    "cond_ns.txt",
    np.column_stack((eus, conds, nums_its)),
    header="Euler number, Condition number, Iterations",
    fmt="%g %g %d",
)
