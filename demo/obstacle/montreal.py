# %% [markdown]
# # Minimal surfaces in Architecture
#
# ```{note}
#   This demo builds on top of the [previous](https://dolfiny.uni.lu/obstacle/membrane) example.
# ```
#
# We will consider the German pavillon at the Expo '67 as an example of minimal surface area used as
# an architectural design methodology.
# Pioneered by Otto Frei, the idea originates from the experimentation with soap bubbles and the
# idea, that the formed *minimal surfaces* should also be structurally relevant.
# The shapes, of the soap bubbles, minimise surface area, and thus are a great example of the
# previously discussed obstacle problems.
#
# ```{figure} https://upload.wikimedia.org/wikipedia/commons/4/4b/Pavillon_de_l%27Allemagne_%282%29.jpg
# :label: img-pavillon
# :align: center
# German Pavillon at Expo '67 in Montreal. René Lavigne,
# [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons.
# ```
# %%
from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import dolfinx.io.gmshio
import ufl

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista

import dolfiny

comm = MPI.COMM_WORLD
# %% [markdown]
# ## Geometry of the Pavillon
# The geometrical data, used in the following, is taken from the great visualizations in
# {cite:p}`Lopez2015`.
# We model the pavillon based on the floorplan described by a list points connected with lines in
# between.
# The original structure relied on a curved boundary representation, however this is hard to
# reparameterize without detailed description of the required parameters.
# Therefore, we stick to a piecewise linear boundary representation here.
#
# Additionally, the tent like structure is held in place by support trusses, for which we define the
# base positions and the corresponding lengths.
#
# Notably the structure did not stand on even grounds which results in the pinned points
# (vertices of the floorplan) and the truss bases having a non-zero $z$-component at times.
#
# The support trusses are split into two groups.
# They either hold up the tent like structure, or pull it down.
# This corresponds to them acting as lower, *positive*,  or upper, *negative* bounds for the
# underlying obstacle problem.
#
# %% tags=["hide-input"]
geometry = np.array(
    [
        [0, 0, 13],
        [5.04, -9.1, -1.8],
        [20.95, -5.82, 3],
        [41.07, -24.15, 3],
        [68.22, -28.99, 3],
        [90.04, -19.52, 3],
        [108.99, -28.34, 3],
        [119.15, -17.88, 3],
        [124.75, -10.99, 3],
        [124.8, -3.63, 3],
        [124.94, 4.28, 3],
        [124.94, 15.10, 3],
        [119.47, 25.12, 3],
        [124.43, 39.66, 3],
        [123.83, 53.1, 3],
        [110.67, 58.64, 3],
        [87.42, 53.18, 3],
        [75.72, 66.73, 0],
        [56.37, 69.7, 0],
        [49.44, 73.74, 0],
        [31.77, 71.45, 0],
        [27.52, 64.24, 0],
        [35.25, 56.11, 0],
        [28.49, 42.08, 0],
        [32.02, 24.79, 0],
        [23.04, 10.36, -1.8],
        [12.15, 15.29, -1.8],
        [0.18, 11.88, -1.8],
    ],
    dtype=np.float64,
)

supports = np.array(
    [
        [0, 0, -1],
        [65.06, -6.4, 0],
        [102.9, 2.34, 0],
        [54.55, 19.54, 0],
        [86.3, 19.54, 0],
        [38.81, 35.41, -3],
        [65.06, 35.41, 0],
        [102.86, 35.41, 0],
        [38.81, 61.16, 0],
        [65.05, 58.62, 0],
    ],
    dtype=np.float64,
)

support_heights = np.array([14, 38, 27, 3, 3, 10, 16, 22, 23, 15], dtype=np.float64)

support_positive = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=np.bool)

# plt.rcParams["font.family"] = "courier"
plt.rcParams["figure.dpi"] = 400

plt.title("Floor Plan (in meters)")
plt.fill(geometry[:, 0], geometry[:, 1], color="lightgray")
plt.plot(
    geometry[:, 0],
    geometry[:, 1],
    "o",
    marker="s",
    markersize=5,
    color="black",
    label="Geometry",
)
plt.plot(
    *zip(*supports[:, :-1][np.where(support_positive)]),
    "o",
    marker="o",
    color="green",
    label="Support (positive)",
)
plt.plot(
    *zip(*supports[:, :-1][np.where(~support_positive)]),
    "o",
    marker="o",
    color="yellow",
    label="Support (negative)",
)

plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# %% tags=["hide-input"]
pyvista.global_theme.font.family = "courier"
pyvista.global_theme.font.size = 40
pyvista.global_theme.font.title_size = 100

plotter = pyvista.Plotter()  # window_size=[3840, 2160])

geometry_proj = geometry.copy()
geometry_proj[:, 2] = 0
floor_plan = pyvista.PolyData(
    geometry_proj,
    np.append(geometry.shape[0], np.append(np.arange(0, geometry.shape[0]), [0])),
).triangulate()
plotter.add_mesh(floor_plan, opacity=0.5, label="Floor Plan (top-down view)")
plotter.add_mesh(pyvista.PolyData(geometry), color="black", point_size=10, label="Fixtures")
width = 1
support_boxes = pyvista.MultiBlock(
    [
        pyvista.Box(
            (
                p[0] - 0.5 * width,
                p[0] + 0.5 * width,
                p[1] - 0.5 * width,
                p[1] + 0.5 * width,
                p[2],
                p[2] + h,
            )
        )
        for p, h in zip(supports, support_heights)
    ]
)

plotter.add_mesh(support_boxes.combine().extract_surface(), color="green", label="Support")
plotter.add_legend()
plotter.show_axes()
plotter.camera.zoom(1.4)
# plotter.show()

# %% [markdown]
# ## Meshing
#
# We will rely on `GMSH` for the generation of the mesh and generate a 2D mesh in the $x-y$-plane
# of the floorplan, which is then deflected into $z$ direction.
# There is one additional geometrical nuance we need to take into account before we can begin to
# setup the mesh.
# As visible in the [image](#img-pavillon) the surface does not span all the way up to the truss
# tops, but rather ends below the top.
# We will take this into account by introducing circular cavities of radius $2\,m$ for the model of
# the membrane at the positions of the support trusses.
#
# ```{note}
#   One of the supports has a base point which resides on the boundary of the geometry.
#   Introducing a circular cavity here, means we need to change the boundary representation of the
#   outer geometry.
#   This case is dealt with during mesh generation separately from the other cavities, which 'just'
#   introduce holes.
# ```
#
# We end up with the following computational mesh.
# %% tags=["hide-input"]
support_radius = 2.0
p_s = support_radius * geometry[-1, :-1] / np.linalg.norm(geometry[-1, :-1])
p_e = support_radius * geometry[1, :-1] / np.linalg.norm(geometry[1, :-1])


def mesh_pavillon() -> dolfinx.io.gmshio.MeshData:
    if comm.rank > 0:
        return dolfinx.io.gmshio.model_to_mesh(None, comm, 0, gdim=2)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # No output
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 3e-2)  # h

    points = [gmsh.model.occ.addPoint(x, y, 0) for x, y, _ in geometry]
    point_start = gmsh.model.occ.addPoint(p_s[0], p_s[1], 0)
    point_center = gmsh.model.occ.addPoint(0, 0, 0)
    point_end = gmsh.model.occ.addPoint(p_e[0], p_e[1], 0)

    segments = [
        gmsh.model.occ.addCircleArc(point_start, point_center, point_end),
        gmsh.model.occ.addLine(point_end, points[1], tag=0),
    ]
    segments += [
        gmsh.model.occ.addLine(points[i], points[i + 1]) for i in range(1, len(points) - 1)
    ]
    segments += [gmsh.model.occ.addLine(len(points), point_start)]

    outer_loop = gmsh.model.occ.addCurveLoop(segments)
    support_circles = [gmsh.model.occ.addCircle(p[0], p[1], 0, support_radius) for p in supports]
    support_base = [gmsh.model.occ.addCurveLoop([c]) for c in support_circles]
    surface = gmsh.model.occ.addPlaneSurface(
        [outer_loop, *[support_base[i] for i in range(1, len(support_base))]]
    )

    gmsh.model.occ.synchronize()

    # remove arc for physical group
    gmsh.model.addPhysicalGroup(1, segments[1:], tag=100)
    gmsh.model.addPhysicalGroup(1, [segments[0]], tag=101)

    for i in range(1, len(support_circles)):
        gmsh.model.addPhysicalGroup(1, [support_circles[i]], tag=101 + i)

    gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    # Optionally store for debugging
    # gmsh.write("pavillon.msh")
    # gmsh.fltk.run()

    return dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)


mesh_data = mesh_pavillon()

mesh = mesh_data.mesh
facet_tags = mesh_data.facet_tags

pyvista.set_jupyter_backend("static")
pv_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, 2))
plotter = pyvista.Plotter(window_size=[3840, 2160])
plotter.add_mesh(pv_grid, show_edges=True)
plotter.camera.zoom(1.6)
plotter.add_axes()
plotter.set_position((10, -200, 200))
plotter.camera.Roll(-10)
plotter.camera.Pitch(-2)
plotter.show()

# %% [markdown]
# ## Obstacle problem and bounds
#
# The construction of the lower bound $\phi$ and upper bound $\psi$ are given by the associated
# heights of the support trusses.
# If the support truss is a positive one, the cavity will be acting as a lower bound of value
# $h_i + z_i$, for height/length of the truss $h_i$ and base $z$-coordinate $x_i$.
# Else, in the case of a negative one, it results in a lower bound with same value construction.
#
# The computation of the Dirichlet boundary data is not as straightforward.
# We only have point data (on the boundary) available and need to extend this to the whole boundary.
# Choosing to linearly interpolate between the two closes boundary vertices, we achieve this by
# defining the boundary as a stand alone mesh and non matchingly interpolating into the deflection
# function space.
#
# %%
V = dolfinx.fem.functionspace(mesh, ("P", 1))

φ = dolfinx.fem.Function(V, name="lb")
φ.x.array[:] = -PETSc.INFINITY  # type: ignore

ψ = dolfinx.fem.Function(V, name="ub")
ψ.x.array[:] = PETSc.INFINITY  # type: ignore

for i in range(len(supports)):
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facet_tags.find(101 + i))  # type: ignore
    if support_positive[i]:
        φ.x.array[dofs] = supports[i, 2] + support_heights[i]
    else:
        ψ.x.array[dofs] = supports[i, 2] + support_heights[i]


def fixed_boundary_data() -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
    # Creates boundary data on boundary mesh.
    # Note: small vertex/cell count, but non-scaling for-loops.
    cells = []
    for i in range(geometry.shape[0]):
        cells += [[i, (i + 1)]]
    values = geometry[:, 2]
    values = np.append(values, geometry[0, 2])

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(2,)))

    boundary_geometry = np.append([p_e], geometry[1:, :-1], axis=0)
    boundary_geometry = np.append(boundary_geometry, [p_s], axis=0)
    boundary_mesh = (
        dolfinx.mesh.create_mesh(comm, np.array(cells, dtype=np.int64), boundary_geometry, element)
        if comm.rank == 0
        else dolfinx.mesh.create_mesh(
            comm, np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float64), element
        )
    )

    V_bdy = dolfinx.fem.functionspace(boundary_mesh, ("CG", 1))
    h = dolfinx.fem.Function(V_bdy)

    for i in range(boundary_geometry.shape[0]):
        for j in range(boundary_mesh.geometry.x.shape[0]):
            if np.isclose(boundary_geometry[i, 0], boundary_mesh.geometry.x[j, 0]) and np.isclose(
                boundary_geometry[i, 1], boundary_mesh.geometry.x[j, 1]
            ):
                h.x.array[j] = values[i]
                break

    h_interpolated = dolfinx.fem.Function(V)
    num_cells_on_proc = (
        V.mesh.topology.index_map(2).size_local + V.mesh.topology.index_map(2).num_ghosts
    )
    V_cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = dolfinx.fem.create_interpolation_data(V, V_bdy, V_cells, padding=1e-12)

    h_interpolated.interpolate_nonmatching(h, V_cells, interpolation_data)
    return h, h_interpolated


h, h_interpolated = fixed_boundary_data()
bc = dolfinx.fem.dirichletbc(
    h_interpolated,
    dolfinx.fem.locate_dofs_topological(V, 1, facet_tags.find(100)),  # type: ignore
)


# %% [markdown]
# As before we rely on the surace area functional.
# In this setting the obstacles are non-smooth (as functions over the whole domain), thus we require
# additionally some form of regularisation {cite:p}`Caffarelli1998`.
# We will use a $H^1$-seminorm here.
# Our complete functional reads
#
# $$
#   F(g) = \int_{\Omega} \sqrt{|\nabla g|^2 + 1} \, \text{d} x
#        + \alpha ||\nabla g||_{L^2(\Omega)}^2.
# $$
#
# ```{note}
#   To verify we do not over-regularize we compare the optimal deflection against the optimal
#   (unregularized) linearized surface area minimizer.
# ```
#
# %%
def S(f):
    return ufl.sqrt(1 + ufl.inner(ufl.grad(f), ufl.grad(f))) * ufl.dx


def S_linear(f):
    return (1 + 0.5 * ufl.inner(ufl.grad(f), ufl.grad(f))) * ufl.dx


def R(f):
    alpha = 5e-2
    return alpha * ufl.inner(ufl.grad(f), ufl.grad(f)) * ufl.dx


opts = PETSc.Options()  # type: ignore
opts["tao_type"] = "bnls"
opts["tao_max_it"] = "1000"
opts["tao_gatol"] = 0
opts["tao_gttol"] = 0
opts["tao_grtol"] = 1e-8

g_linear = dolfinx.fem.Function(V, name="g-linear")
tao_linear = dolfiny.taoproblem.TAOProblem(S_linear(g_linear), [g_linear], [bc], lb=[φ], ub=[ψ])
tao_linear.solve()

g = dolfinx.fem.Function(V, name="g")
tao = dolfiny.taoproblem.TAOProblem(S(g) + R(g), [g], [bc], lb=[φ], ub=[ψ])
tao.solve()

# %% tags=["hide-input"]
for problem in [tao_linear, tao]:
    if problem.tao.getConvergedReason() <= 0:
        raise RuntimeError("Optimisation did not converge.")

if np.any(g_linear.x.array < φ.x.array) or np.any(g.x.array < φ.x.array):
    raise RuntimeError("Lower bound violated.")

if np.any(g_linear.x.array > ψ.x.array) or np.any(g.x.array > ψ.x.array):
    raise RuntimeError("Upper bound violated.")

if comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(S(g)))) >= comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(S(g_linear)))
):
    raise RuntimeError("Nonlinear solution is less accurate than linear.")


def plot(f):
    pv_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, 2))
    surface = np.zeros((V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, 3))
    surface[:, 2:3] = f.x.array.reshape(-1, 1)
    pv_grid.point_data[f.name] = surface
    pv_grid.warp_by_vector(f.name, inplace=True)
    plotter = pyvista.Plotter(window_size=[3840, 2160])
    sargs = dict(
        height=0.8,
        position_y=0.1,
        title="",
        font_family="courier",
        title_font_size=100,
        label_font_size=40,
        fmt="%1.2f",
        color="black",
        vertical=True,
    )
    plotter.add_mesh(
        pv_grid,
        scalars=f.name,
        specular=0.1,
        specular_power=5,
        scalar_bar_args=sargs,
        smooth_shading=True,
        split_sharp_edges=True,
        cmap="coolwarm",
    )
    plotter.camera.zoom(1.6)
    plotter.add_axes()
    plotter.set_position((10, -200, 200))
    plotter.camera.Roll(-10)
    plotter.camera.Pitch(-2)
    plotter.show()


# %% [markdown]
# ## The surface
#
# We end up with the following surface, that resembles the real world structure quite well.
#
# %%
plot(g)

# %% tags=["hide-input"]
for f in [h, h_interpolated, g_linear, g, φ, ψ]:
    with dolfinx.io.VTXWriter(V.mesh.comm, f"montreal/{f.name}.bp", f, "bp4") as file:
        file.write(0.0)
