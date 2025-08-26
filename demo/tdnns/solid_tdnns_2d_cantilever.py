#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import hdivdiv
import numpy as np

import dolfiny

# Basic settings
name = "solid_tdnns_2d_cantilever"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
ox = [0.0, -0.05]
rx = [1.0, +0.05]
nn = [10, 4]

# Get mesh and meshtags
mesh, tdim = dolfinx.mesh.create_rectangle(comm, [ox, rx], nn), 2
mts = {}
for d in range(mesh.geometry.dim):
    for k, kx in enumerate((ox[d], rx[d])):
        facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[d], kx))
        mt = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, 2 * d + k)
        mts[f"face_x{d}={'min' if k == 0 else 'max'}"] = mt

mesh.topology.create_connectivity(tdim - 1, tdim)

# Get merged MeshTags for each codimension
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_left = interfaces_keys["face_x0=min"]
surface_right = interfaces_keys["face_x0=max"]
surface_bottom = interfaces_keys["face_x1=min"]
surface_top = interfaces_keys["face_x1=max"]

# Solid material parameters
mu = dolfinx.fem.Constant(mesh, scalar(100.0))  # GPa
la = dolfinx.fem.Constant(mesh, scalar(150.0))  # GPa

# Impressed action
g = dolfinx.fem.Constant(mesh, [0.0, 0.0])  # volume force vector
t = dolfinx.fem.Constant(mesh, [0.0, 0.1])  # tangential boundary stress vector

# Define integration measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)
dS = ufl.Measure("dS", domain=mesh)

# Define elements
Ue = basix.ufl.element("N2E", mesh.basix_cell(), 2)
Se = hdivdiv.create_custom_hdivdiv(mesh.basix_cell(), 2, verbose=not comm.rank)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
S = dolfinx.fem.Function(Sf, name="S")

u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions
S_ = dolfinx.fem.Function(Sf, name="S_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Sf))
δu, δS = δm

# Define state as (ordered) list of functions
m = [u, S]

# Create other functions: output / visualisation
vorder = mesh.geometry.cmap.degree
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (2, 2), True)), name="S")
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (2,))), name="u")
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")


# Strain, kinematically
def Eu(u):
    return ufl.sym(ufl.grad(u))  # linear


# Strain, inverse constitutive
def Es(S):
    return 1 / (2 * mu) * S - la / (2 * mu * (2 * la + 2 * mu)) * ufl.tr(S) * ufl.Identity(2)


# Mesh facet normal
n = ufl.FacetNormal(mesh)

# von Mises stress (output), FIXME: adapt to plane strain
s = ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S)))


# Duality pairing operator
def b(S, u):
    return (
        -ufl.inner(S, Eu(u)) * dx
        + ufl.dot(ufl.dot(S, n), n) * ufl.dot(u, n) * ds
        + ufl.dot(ufl.dot(S("+"), n("+")), n("+")) * ufl.jump(u, n) * dS
    )


# Form, dual mixed form, TD-NNS, duality pairing
form = (
    +ufl.inner(δS, Es(S)) * dx
    + b(δS, u)
    + b(S, δu)
    + ufl.dot(δu, g) * dx
    + ufl.dot(δu, t) * ds(surface_right)
)

form += (
    dolfinx.fem.Constant(mesh, scalar(0.0)) * ufl.inner(δu, u) * dx
)  # ensure zero block diagonal for bc

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_left)
surface_rest_dofs_Sf = dolfiny.mesh.locate_dofs_topological(
    Sf, interfaces, [surface_right, surface_top, surface_bottom]
)

# Define boundary conditions
bcs = [
    dolfinx.fem.dirichletbc(u_, surface_left_dofs_Uf),  # u_t left (tangentially fixed)
    dolfinx.fem.dirichletbc(S_, surface_rest_dofs_Sf),  # S_nn rest (stress-free)
]

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-08
opts["snes_rtol"] = 1.0e-06
opts["snes_max_it"] = 1
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 200  # percentage of max. memory increase during numerical phase
opts["mat_mumps_icntl_24"] = 1

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, bcs, prefix=name)

# Solve problem
problem.solve()

# Assert convergence of solver
problem.status(verbose=True, error_on_failure=True)

# Assert symmetry of operator
assert dolfiny.la.is_symmetric(problem.J)

# Check properties of solution
x = ufl.SpatialCoordinate(mesh)

ref_tuple = ufl.dot(ufl.dot(S, n), n), ds
exp_tuple = 0.0, ds
assert np.isclose(dolfiny.expression.assemble(*ref_tuple), dolfiny.expression.assemble(*exp_tuple))
ref_tuple = ufl.jump(ufl.dot(ufl.dot(S, n), n)), dS
exp_tuple = 0.0, dS
assert np.isclose(dolfiny.expression.assemble(*ref_tuple), dolfiny.expression.assemble(*exp_tuple))
ref_tuple = +ufl.dot(S, n) - ufl.dot(ufl.dot(S, n), n) * n, ds(surface_left)
exp_tuple = -ufl.dot(S, n) - ufl.dot(ufl.dot(S, n), n) * n, ds(surface_right)
assert np.isclose(
    dolfiny.expression.assemble(*ref_tuple), dolfiny.expression.assemble(*exp_tuple), atol=1e-03
).all()
ref_tuple = ufl.dot(ufl.dot(S, n), n) * x[1], ds(surface_left)
exp_tuple = (t[0] * x[1] - t[1] * x[0]), ds(surface_right)
assert np.isclose(
    dolfiny.expression.assemble(*ref_tuple), dolfiny.expression.assemble(*exp_tuple), atol=1e-03
)

# Write results to file
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
ofile.write_mesh_meshtags(mesh)
dolfiny.interpolation.interpolate(u, uo)
dolfiny.interpolation.interpolate(S, So)
dolfiny.interpolation.interpolate(s, so)
ofile.write_function(uo)
ofile.write_function(So)
ofile.write_function(so)
ofile.close()

# Read results and plot using pyvista (all in serial)
if comm.rank == 0:
    import pyvista

    class Xdmf3Reader(pyvista.XdmfReader):
        _vtk_module_name = "vtkIOXdmf3"
        _vtk_class_name = "vtkXdmf3Reader"

    reader = Xdmf3Reader(path=f"./{name}.xdmf")
    multiblock = reader.read()

    grid = multiblock[-1]
    grid.point_data["u"] = multiblock[0].point_data["u"]
    grid.point_data["S"] = multiblock[1].point_data["S"]
    grid.point_data["s"] = multiblock[2].point_data["s"]

    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)

    grid_warped = grid.warp_by_vector("u", factor=1.0)

    subdivision_levels = 4 if not grid.get_cell(0).is_linear else 0

    plotter.add_mesh(
        grid_warped.extract_surface(nonlinear_subdivision=subdivision_levels),
        scalars="s",
        scalar_bar_args={"title": "von Mises stress"},
        n_colors=10,
    )

    plotter.add_mesh(
        grid_warped.separate_cells()
        .extract_surface(nonlinear_subdivision=subdivision_levels)
        .extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=dolfiny.pyvista.pixels // 500,
    )
    plotter.add_mesh(
        grid,
        color="black",
        style="wireframe",
        opacity=0.2,
        line_width=dolfiny.pyvista.pixels // 500,
    )

    plotter.show_axes()
    plotter.view_xy()

    plotter.screenshot(f"{name}.png", transparent_background=False)
