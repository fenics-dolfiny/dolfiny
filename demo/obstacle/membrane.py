# %% [markdown]
# # Obstacle Problems with 🐬
#
# This demo showcases how to solve obstacle problems using the Dolfiny interface to PETSc/TAO.
# Both linear and non-linear formulations are considered.
#
# ## The Surface Area Functional
#
# Let us consider a surface $\mathcal{F} \subset \mathbb{R}^3$ which is parametrized by a regular
# paramatetrization $\Psi$ over $B \subset \mathbb{R}^2$, i.e. $\mathcal{F} = \Psi(B)$.
# We assume, $\Psi$ can be expressed as a height map over the cartesian coordinates
#
# $$
#   \Psi(x,y) = (x, y, g(x, y)).
# $$
#
# The *surface area* of $\mathcal{F}$ is then given by
#
# $$
# \begin{align*}
#     S(g)
#     &= \int_\mathcal{F} 1 \, \text{d} x \\
#     &= \int_{\Psi(B)} 1 \, \text{d} x \\
#     &= \int_{B} | \det ( D\Psi(x) ) | \, \text{d} x \\
#     &= \int_{B} \left|
#         \frac{\partial \Psi}{\partial x} \times \frac{\partial \Psi}{\partial y}
#        \right| \, \text{d} x \\
#     &= \int_{B} \left|
#         \begin{bmatrix} 1, 0, \frac{\partial g}{\partial x} \end{bmatrix} \times
#         \begin{bmatrix} 0, 1, \frac{\partial g}{\partial y} \end{bmatrix}
#        \right| \, \text{d} x \\
#     &= \int_{B} \sqrt{ \left( \frac{\partial g}{\partial x}\right)^2 +
#         \left(\frac{\partial g}{\partial y}\right)^2 + 1 } \, \text{d} x \\
#     &= \int_{B} \sqrt{|\nabla g|^2 + 1} \, \text{d} x.
# \end{align*}
# $$
#
# Especially, the surface area is a nonlinear functional in $g$.
# To circumvent the additional complexity of non-linear problems, often the linearized version is
# considered.
# The second order Taylor expansion of $\sqrt{x^2 + 1}$ around $x = 0$ reads
#
# $$
# \begin{align*}
#     \sqrt{x^2 + 1}
#     &= \sqrt{0^2 + 1} + \frac{0}{\sqrt{0^2 + 1}} x
#         + \frac{1}{2} \frac{1-0^2}{\left( 0^2 + 1 \right)^\frac{3}{2}} x^2 + \mathcal{O} (x^3) \\
#     &= 1 + \frac{1}{2} x^2 + \mathcal{O} (x^3).
# \end{align*}
# $$
#
# Therefore the *linearized surface area* is given by
#
# $$
#     S_\text{linear}(g) = \int_{B} 1 + \frac{1}{2} |\nabla g|^2 \, \text{d} x.
# $$
#
# ## The Obstacle Problem
#
# The *obstacle problem* is the task of finding a surface of minimal area spanning over some
# *obstacle* $\phi: B \to \mathbb{R}$
#
# $$
#    \min_g S(g) \quad \text{ subject to } g \geq \phi \quad \text{(pointwise)}.
# $$
#
# ```{note}
# When using the linear surface area $S_\text{linear}$ the obstacle problem is equivalent to
# minimizing Dirichlet energy of $g$ under the same constraint
#
# $$
#     \min_g \int_B |\nabla g|^2 \, \text{d}x
#     \quad \text{ subject to }
#     g \geq \phi \quad \text{(pointwise)}.
# $$
#
# ```
#
# For this demonstration we choose $B = (-1,1)^2$, and
#
# $$
# \phi(x, y) = \sin \left(3\pi \sqrt{x^2 + y^2} \right) \left( 1-x^2 \right) \left( 1-y^2 \right).
# $$

# %%
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl

import numpy as np
import pyvista

import dolfiny
import dolfiny.taoproblem

comm = MPI.COMM_WORLD

msh = dolfinx.mesh.create_rectangle(comm, [[-1, -1], [1, 1]], [128, 128])

V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))  # type: ignore

msh.topology.create_connectivity(1, 2)
boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(np.array(0.0), dofs=boundary_dofs, V=V)

x = ufl.SpatialCoordinate(msh)
φ = ufl.sin(np.pi * 3 * ufl.sqrt(x[0] ** 2 + x[1] ** 2)) * (1 - x[0] ** 2) * (1 - x[1] ** 2)

lb = dolfinx.fem.Function(V, name="Phi")
lb.interpolate(dolfinx.fem.Expression(φ, V.element.interpolation_points))

opts = PETSc.Options("obstacle")  # type: ignore
opts["tao_type"] = "bnls"
opts["tao_max_it"] = "100"

g_linear = dolfinx.fem.Function(V, name="g_linear")
S_linear = (1 + 0.5 * ufl.inner(ufl.grad(g_linear), ufl.grad(g_linear))) * ufl.dx
linear_problem = dolfiny.taoproblem.TAOProblem(
    S_linear, [g_linear], [bc], lb=[lb], prefix="obstacle"
)
linear_problem.solve()

g = dolfinx.fem.Function(V, name="g")
S = ufl.sqrt(1 + ufl.inner(ufl.grad(g), ufl.grad(g))) * ufl.dx
problem = dolfiny.taoproblem.TAOProblem(S, [g], [bc], lb=[lb], prefix="obstacle")
problem.solve()

dolfiny.utils.pprint(f"S_linear(g) = {linear_problem.tao.getFunctionValue():.4f}")
dolfiny.utils.pprint(f"S(g)        = {problem.tao.getFunctionValue():.4f}")


# %%
pyvista.set_jupyter_backend("static")
pv_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh, 2))


def plot_deflected(f):
    if comm.size != 1:
        return

    # Extend to 3D: u = [0, 0, f]
    u = np.zeros((V.dofmap.index_map.size_local, 3))
    u[:, 2:3] = f.x.array.reshape(-1, 1)
    pv_grid.point_data[f.name] = u

    plotter = pyvista.Plotter()
    plotter.image_scale = 4

    warped = pv_grid.warp_by_vector(f.name, factor=1.0)
    warped.active_scalars_name = f.name
    sargs = dict(
        width=0.8,
        position_x=0.1,
        title=f.name,
        font_family="courier",
        fmt="%1.2f",
        color="black",
    )
    plotter.add_mesh(
        warped,
        component=2,
        scalar_bar_args=sargs,
        specular=0.25,
        specular_power=5,
        smooth_shading=True,
        split_sharp_edges=True,
        cmap="coolwarm",
    )

    plotter.camera.elevation -= 20
    plotter.show()


# %%
plot_deflected(lb)

# %%
plot_deflected(g_linear)

# %%
plot_deflected(g)

# %%
with dolfinx.io.VTXWriter(V.mesh.comm, "direct.bp", lb, "bp4") as file:
    file.write(0.0)
with dolfinx.io.VTXWriter(V.mesh.comm, "g.bp", g, "bp4") as file:
    file.write(0.0)
with dolfinx.io.VTXWriter(V.mesh.comm, "g_linear.bp", g_linear, "bp4") as file:
    file.write(0.0)

# TODO: add CI relevant checking here - converged?, error norms?...

# %%
