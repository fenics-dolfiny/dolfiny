"""
+--------------------------------------------+
| Poisson dimensional demo                   |
+--------------------------------------------+

PDE
---

    -∇ · (κ ∇u) = f

Units
-----

    [x]     = l_ref = L
    [u]     = u_ref = Θ
    [κ]     = κ_ref = W / (m K)
    [f]     = f_ref = W / m³

Dimensional relation
--------------------

    [f] = [κ] [u] / [x]²

or equivalently

    f_ref = κ_ref u_ref / l_ref²

Dimensionless group
-------------------

    Π = κ_ref u_ref / (f_ref l_ref²)

or its inverse, depending on normalization.
"""

from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import sympy as sy
import sympy.physics.units as syu

import dolfiny
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

T = dolfinx.fem.Function(V, name="T")
f = dolfinx.fem.Function(V, name="source")
v = ufl.TestFunction(V)

kappa = Quantity(mesh, 1.0, syu.W / (syu.K * syu.m), "kappa")
l_ref = Quantity(mesh, 1.0, syu.m, "l_ref")
T_ref = Quantity(mesh, 1.0, syu.K, "T_ref")
f_ref = Quantity(mesh, 1.0, syu.W / syu.m**3, "f_ref")

terms = {
    "source": f * v * ufl.dx,
    "diss": ufl.inner(kappa * ufl.grad(T), ufl.grad(v)) * ufl.dx,
}
mapping = {
    mesh.ufl_domain(): l_ref,
    T: T_ref * T,
    f: f_ref * f,
    v: T_ref * v,
}

quantities = dolfiny.units.collect_quantities(sum(terms.values(), ufl.form.Zero()), mapping=mapping)
assert set(quantities) == {T_ref, f_ref, l_ref, kappa}

# Buckingham Pi analysis
_, _, pi_groups = dolfiny.units.buckingham_pi_analysis(quantities)

assert len(pi_groups) == 1
pi_expr = sy.simplify(expand(list(pi_groups[0]), [q.symbol for q in quantities]))
pi_expected = kappa.symbol * T_ref.symbol / (f_ref.symbol * l_ref.symbol**2)
assert sy.simplify(pi_expr / pi_expected - 1) == 0 or sy.simplify(pi_expr * pi_expected - 1) == 0

# Dimensional consistency using mapping
diffusion_dim = get_dimension(terms["diss"], quantities, mapping=mapping)
rhs_dim = get_dimension(terms["source"], quantities, mapping=mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(diffusion_dim, rhs_dim)

factorized = factorize(terms, quantities, mode="factorize", mapping=mapping)
normalized = normalize(factorized, "source", quantities)
