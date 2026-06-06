from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import sympy as sy
import sympy.physics.units as syu

from dolfiny.units import (
    Quantity,
    buckingham_pi_analysis,
    expand,
    factorize,
    get_dimension,
    normalize,
    transform,
)

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError("This minimal Helmholtz demo must be run with a single MPI rank.")

print("Step 1: quantities and problem data")
mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
V = dolfinx.fem.functionspace(mesh, ("P", 1))

u = dolfinx.fem.Function(V, name="u")
rhs = dolfinx.fem.Function(V, name="rhs")
v = ufl.TestFunction(V)

ell = Quantity(mesh, 1.0, syu.meter, "ell")
kappa = Quantity(mesh, 2.0, 1 / syu.meter, "kappa")
u_ref = Quantity(mesh, 3.0, syu.kelvin, "u_ref")

quantities = [ell, kappa, u_ref]
assert str(ell) == "ell"
assert str(kappa) == "kappa"
assert str(u_ref) == "u_ref"

print("Step 2: weak form and mapping")
dx = ufl.dx(domain=mesh)
terms = {
    "mass": ufl.inner(u, v) * dx,
    "diffusion": (1 / kappa**2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx,
    "rhs": rhs * v * dx,
}
mapping = {
    mesh.ufl_domain(): ell,
    u: u_ref * u,
    rhs: u_ref * rhs,
    v: u_ref * v,
}

transformed_terms = transform(terms, mapping)
assert isinstance(transformed_terms, dict)
assert set(transformed_terms) == {"mass", "diffusion", "rhs"}

print("Step 3: dimensional checks")
dimsys = syu.si.SI.get_dimension_system()
mass_dim = get_dimension(transformed_terms["mass"], quantities)
diffusion_dim = get_dimension(transformed_terms["diffusion"], quantities)
rhs_dim = get_dimension(transformed_terms["rhs"], quantities)

assert dimsys.equivalent_dims(mass_dim, diffusion_dim)
assert dimsys.equivalent_dims(mass_dim, rhs_dim)
assert dimsys.equivalent_dims(mass_dim, u_ref.dimension**2 * ell.dimension**2)

print("Step 4: factorization and normalization")
factorized_terms = factorize(transformed_terms, quantities, mode="factorize")
assert np.allclose(factorized_terms["mass"].factor, np.array([2.0, 0.0, 2.0]))
assert np.allclose(factorized_terms["diffusion"].factor, np.array([0.0, -2.0, 2.0]))
assert np.allclose(factorized_terms["rhs"].factor, np.array([2.0, 0.0, 2.0]))

normalized_terms = normalize(factorized_terms, "mass", quantities)
normalized_diffusion = factorize(normalized_terms["diffusion"], quantities, mode="factorize")
assert np.allclose(normalized_diffusion.factor, np.array([-2.0, -2.0, 0.0]))

print("Step 5: Buckingham Pi analysis")
dim_matrix, base_dims, pi_groups = buckingham_pi_analysis(quantities)
assert dim_matrix.shape == (len(base_dims), len(quantities))
assert len(pi_groups) == 1

pi_expr = sy.simplify(expand(list(pi_groups[0]), [q.symbol for q in quantities]))
assert sy.simplify(pi_expr - ell.symbol * kappa.symbol) == 0

print("Helmholtz demo completed successfully.")

