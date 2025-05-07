# %% [markdown]
# # Obstacle problem
# This demo showcases how to solve obstacle problems with PETSc TAO's optimziation algorithms
# with the Dolfiny 🐬 interface.
# Both linear and non-linear formulations are demonstrated.

# ## Obstacle problems
# Let us take a surface $\mathcal{F} \subset \mathbb{R}^3$ which is parametrized by a regular
# paramatetrization $\Psi$ over $B \subset \mathbb{R}^2$, i.e. $\mathcal{F} = \Psi(B)$.
# The *suface area* of $\mathcal{F}$ is then given by
# $$
#   \int_\mathcal{F} 1 \, \text{d} x
#   = \int_{\Psi(B)} 1 \, \text{d} x
#   = \int_{B} | \det ( D\Psi(x) ) | \, \text{d} x
#   = \int_{B} \left| \frac{\partial \Psi}{\partial x} \times \frac{\partial \Psi}{\partial y} \right| \, \text{d} x.
# $$
# Let us now choose the coordinates $\Psi_1 = (x, y, g(x, y))$ and $\Psi_2 = (x, y, )$ TODO: continue
#
# ## Minimizing Dirichlet Energy
# TODO: another option where obstacle problems show up naturely
# TODO: change to smooth lower bound
# %%

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl
from dolfinx import fem, io, mesh

import numpy as np

import dolfiny
import dolfiny.taoblockproblem

# import dolfiny

n = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)

V = fem.functionspace(msh, ("Lagrange", 1))

boundary_facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[1], 1.0),
)
boundary_dofs = fem.locate_dofs_topological(
    V,
    msh.topology.dim - 1,
    boundary_facets,
)
bc = fem.dirichletbc(value=0.0, dofs=boundary_dofs, V=V)

u = fem.Function(V, name="u")

# linearised surface area: (Dirichlet energy)
F = ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
# non-linear surface area:
F = ufl.sqrt(1 + ufl.inner(ufl.grad(u), ufl.grad(u))) * ufl.dx

opts = PETSc.Options("obstacle")
opts["tao_type"] = "bnls"
opts["tao_ls_monitor"] = ""
opts["tao_max_it"] = "1000"
# opts["tao_monitor"] = ""

problem = dolfiny.taoblockproblem.TAOBlockProblem(F, [u], [bc], prefix="obstacle")

ub = fem.Function(V, name="upper_bound")
ub.x.array[:] = 1e4
bc.set(ub.x.array, alpha=1)

lb = fem.Function(V, name="lower_bound")
x = ufl.SpatialCoordinate(msh)
h = ufl.conditional((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 <= 0.2**2, 1, 0)
lb.interpolate(fem.Expression(h, V.element.interpolation_points))
bc.set(lb.x.array, alpha=1)
bc.set(lb.x.array, alpha=1)

# TODO: move into problem
problem.tao.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)

(u,) = problem.solve()

with io.XDMFFile(msh.comm, "out_obstacle/data.xdmf", "w") as file:
    file.write_mesh(msh)

    # file.write_function(interpolation_h)
    file.write_function(u)

# %%
