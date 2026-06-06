"""
+--------------------------------------------+
| Minimal Helmholtz dimensional demo         |
+--------------------------------------------+

PDE
---

    -∇ · (κ^-2 ∇u) = f

Units
-----

    [x]     = l = L
    [u]     = u_ref = Θ
    [rhs]   = u_ref = Θ
    [κ]     = l^-1 = L^-1
"""

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


mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
V = dolfinx.fem.functionspace(mesh, ("P", 1))

u = dolfinx.fem.Function(V, name="u")
f = dolfinx.fem.Function(V, name="source")
v = ufl.TestFunction(V)

kappa = Quantity(mesh, 2.0, 1 / syu.meter, "kappa")

l_ref = Quantity(mesh, 1.0, syu.meter, "l_ref")
u_ref = Quantity(mesh, 3.0, syu.kelvin, "u_ref")

quantities = [l_ref, kappa, u_ref] # TODO: reorder?

term_source = f * v * ufl.dx
term_diff = (1 / kappa**2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
terms = {
    "diffusion": term_diff,
    "source": term_source,
}
mapping = {
    mesh.ufl_domain(): l_ref,
    u: u_ref * u,
    f: u_ref * f,
    v: u_ref * v,
}

transformed_terms = transform(terms, mapping)
assert transformed_terms.keys() == { "diffusion", "source"}

dimsys = syu.si.SI.get_dimension_system()
diffusion_dim = get_dimension(transformed_terms["diffusion"], quantities)
rhs_dim = get_dimension(transformed_terms["source"], quantities)

# factorize
factorized_terms = factorize(transformed_terms, quantities, mode="check")
assert np.allclose(factorized_terms["diffusion"].factor, np.array([0.0, -2.0, 2.0]))
assert np.allclose(factorized_terms["source"].factor, np.array([2.0, 0.0, 2.0]))

# buckingham pi
dim_matrix, base_dims, pi_groups = buckingham_pi_analysis(quantities)
assert dim_matrix.shape == (len(base_dims), len(quantities))
assert len(pi_groups) == 1

pi_expr = sy.simplify(expand(list(pi_groups[0]), [q.symbol for q in quantities]))
assert sy.simplify(pi_expr - l_ref.symbol * kappa.symbol) == 0
