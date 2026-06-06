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
import dolfiny
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

kappa = Quantity(mesh, 1.0, 1 / syu.meter, "kappa")
l_ref = Quantity(mesh, 1.0, syu.meter, "l_ref")
u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")

terms = {
    "diffusion": f * v * ufl.dx,
    "source": (1 / kappa**2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
}
mapping = {
    mesh.ufl_domain(): l_ref,
    u: u_ref * u,
    f: u_ref * f,
    v: u_ref * v,
}

quantities = dolfiny.units.collect_quantities(sum(terms.values(), ufl.form.Zero()), mapping=mapping)
assert set(quantities) == {l_ref, kappa, u_ref}

# Buckingham Pi analysis
_, _, pi_groups = dolfiny.units.buckingham_pi_analysis(quantities)

assert len(pi_groups) == 1
pi_expr = sy.simplify(expand(list(pi_groups[0]), [q.symbol for q in quantities]))
assert sy.simplify(pi_expr - l_ref.symbol * kappa.symbol) == 0

# Dimensional consistency using mapping
diffusion_dim = get_dimension(terms["diffusion"], quantities, mapping=mapping)
rhs_dim = get_dimension(terms["source"], quantities, mapping=mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(diffusion_dim, rhs_dim)

factorized = factorize(terms, quantities, mode="factorize", mapping=mapping)
normalized = normalize(factorized, "source", quantities)
