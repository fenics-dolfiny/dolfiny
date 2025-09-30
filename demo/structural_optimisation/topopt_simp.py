# %% [markdown]
# # Topology optimisation
#
# This demo showcases the mother of all topology optimisation problems: compliance minimisation of a
# **s**olid **i**sotropic **m**aterial with **p**enalisation (SIMP) regularised with a Helmholtz
# filter.
#
# In particular this demo emphasizes
# 1. ... TODO
# 1. ~implementation of a linear elastic truss model with ufl,~
# 2. ~creation of (braced) truss meshes, and~
# 3. ~the interface to PETSc/TAO for otimisation solvers.~
#
# %%
import argparse

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem.petsc
import ufl

import numpy as np
import pyvista as pv

import dolfiny

# %% tags=["hide-input"]
output = False
parser = argparse.ArgumentParser(description="Truss sizing demo")
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["conlin", "mma"],
    default="mma",
    help="Choose optimisation algorithm",
)
args, _unknown = parser.parse_known_args()

# %% [markdown]
# ## Computational domain
#
# The computational domain is given by $\Omega = (0, 2) \times (0,1) \subset \mathbb{R}^2$ and is
# discretized by $(2n, n)$ quadrilateral elements.
# %%
tdim = 2
n = 50
comm = MPI.COMM_WORLD
mesh = (
    dolfinx.mesh.create_rectangle(
        comm, [[0, 0], [2, 1]], (2 * n, n), cell_type=dolfinx.mesh.CellType.quadrilateral
    )
    if tdim == 2
    else dolfinx.mesh.create_box(
        comm, [[0, 0, 0], [2, 1, 1]], (2 * n, n, n), cell_type=dolfinx.mesh.CellType.hexahedron
    )
)

# %% [markdown]
# ## Function spaces
#
# TODO:
# %%
V_u = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (tdim,)))
V_ρ = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
V_ρ_f = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

ρ = dolfinx.fem.Function(V_ρ, name="density")
ρ_f = dolfinx.fem.Function(V_ρ_f, name="density-filtered")
u = dolfinx.fem.Function(V_u, name="u")

ρ_min = np.float64(1e-9)
penalty = 3

# ASTM A-36 / EN S235J2 steel
E0 = 2.11e11  # Pa
E = (ρ_min + (1 - ρ_min) * ρ_f**penalty) * E0
nu = 0.29  # dimensionless


def ε(u):  # strain
    return ufl.sym(ufl.grad(u))


def σ(u):  # stress
    # Lamé parameters λ and μ
    λ = E * nu / ((1 + nu) * (1 - 2 * nu))
    μ = E / (2 * (1 + nu))
    return λ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * μ * ε(u)


def on_rhs(x):
    mask = np.isclose(x[0], 2.0) & np.greater_equal(x[1], 0.45) & np.less_equal(x[1], 0.55)
    if tdim == 3:
        mask &= np.greater_equal(x[2], 0.45) & np.less_equal(x[2], 0.55)
    return mask


facets_rhs = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, on_rhs)
facet_tag = dolfinx.mesh.meshtags(mesh, tdim - 1, np.unique(facets_rhs), 1)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)


def compliance(u):
    load = -8.6e4  # Nw
    f = ufl.as_vector((0, load) if tdim == 2 else (0, load, 0))
    return ufl.inner(f, u) * ds(1)


def elastic_energy(u):
    E = 1 / 2 * ufl.inner(σ(u), ε(u)) * ufl.dx
    E -= compliance(u)
    return E


fixed_entities = dolfinx.mesh.locate_entities_boundary(
    mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0)
)
fixed_dofs = dolfinx.fem.locate_dofs_topological(V_u, tdim - 1, fixed_entities)
bc_u = dolfinx.fem.dirichletbc(np.zeros(tdim, dtype=ScalarType), fixed_dofs, V_u)

# State solver
a = ufl.derivative(ufl.derivative(elastic_energy(u), u), u)
L = -ufl.derivative(elastic_energy(u), u)
L = ufl.replace(L, {u: ufl.as_vector((0, 0) if tdim == 2 else (0, 0, 0))})

elas_prob = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    bcs=[bc_u],
    u=u,
    petsc_options=(
        {
            "ksp_error_if_not_converged": "True",
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
        }
        if tdim == 2
        else {
            # Combination of https://github.com/FEniCS/performance-test and https://doi.org/10.1007/s00158-020-02618-z
            "ksp_error_if_not_converged": True,
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-8,
            "pc_type": "gamg",
            "pc_gamg_type": "agg",
            "pc_gamg_agg_nsmooths": 1,
            "pc_gamg_threshold": 0.001,
            "mg_levels_esteig_ksp_type": "cg",
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_chebyshev_esteig_steps": 50,
            "mg_levels_pc_type": "sor",
            "matptap_via": "scalable",
            "pc_gamg_coarse_eq_limit": 1000,
        }
    ),
    petsc_options_prefix="elasticity_ksp",
)

r = 0.45 * ufl.CellDiameter(mesh)  # factor 1-3
u_f, v_f = ufl.TrialFunction(V_ρ_f), ufl.TestFunction(V_ρ_f)
a_filter = dolfinx.fem.form(
    r**2 * ufl.inner(ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx + u_f * v_f * ufl.dx
)
L_filter_ρ = dolfinx.fem.form(ρ * v_f * ufl.dx)
s = dolfinx.fem.Function(V_ρ_f, name="s")
L_filter_s = dolfinx.fem.form(s * v_f * ufl.dx)

A_filter = dolfinx.fem.petsc.create_matrix(a_filter)
dolfinx.fem.petsc.assemble_matrix(A_filter, a_filter)
A_filter.assemble()

b_filter = dolfinx.fem.petsc.create_vector(V_ρ_f)

opts = PETSc.Options("filter")  # type: ignore
opts["ksp_type"] = "cg"
opts["pc_type"] = "jacobi"
opts["ksp_error_if_not_converged"] = True

filter_ksp = PETSc.KSP().create()  # type: ignore
filter_ksp.setOptionsPrefix("filter")
filter_ksp.setFromOptions()
filter_ksp.setOperators(A_filter)


def apply_filter(rhs, f) -> None:
    """Compute filtered f from rhs."""

    with b_filter.localForm() as b_local:
        b_local.set(0.0)

    dolfinx.fem.petsc.assemble_vector(b_filter, rhs)
    b_filter.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
    b_filter.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    filter_ksp.solve(b_filter, f.x.petsc_vec)
    f.x.petsc_vec.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore


J_form = dolfinx.fem.form(compliance(u))
DJ_form = dolfinx.fem.form(-ufl.derivative(elastic_energy(u), ρ_f))

mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx))
)
volume_fraction = ρ / mesh_volume * ufl.dx
max_volume_fraction = 0.4 if tdim == 2 else 0.1

g = volume_fraction <= max_volume_fraction

ρ.x.array[:] = max_volume_fraction
ρ_f.interpolate(ρ)

apply_filter(L_filter_ρ, ρ_f)
elas_prob.solve()

c0 = comm.allreduce(dolfinx.fem.assemble_scalar(J_form))

J_scale = 1 / c0  # normalize
J_scale *= 10  # target objective range


@dolfiny.taoproblem.sync_functions([ρ])
def J(tao, _):
    apply_filter(L_filter_ρ, ρ_f)

    # Compute displacement from filtered density.
    elas_prob.solve()

    return comm.allreduce(dolfinx.fem.assemble_scalar(J_form)) * J_scale


Dρ = dolfinx.fem.Function(V_ρ_f)
z = dolfinx.fem.Function(V_ρ_f, name="z")
tmpDG0 = dolfinx.fem.Function(V_ρ)


@dolfiny.taoproblem.sync_functions([ρ])
def DJ(tao, _, G):
    # TODO: surely not necessary?

    # Compute filtered density from density.
    # apply_filter()

    # Compute displacement from filtered denstity.
    # elas_prob.solve()

    # Assemble variation (w.r.t. filtered density).
    with s.x.petsc_vec.localForm() as local:
        local.set(0.0)
    dolfinx.fem.petsc.assemble_vector(s.x.petsc_vec, DJ_form)
    s.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    s.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Apply adjoint to DJ/s -> z.
    apply_filter(L_filter_s, z)

    # Interpolate/project z into DG0.
    tmpDG0.interpolate(z)

    # Copy to G.
    tmpDG0.x.petsc_vec.copy(G)
    G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    G.scale(J_scale)


opts = PETSc.Options()  # type: ignore
opts["tao_type"] = "python"
opts["tao_monitor"] = ""
opts["tao_max_it"] = (max_it := 100)
if args.algorithm == "conlin":
    opts["tao_python_type"] = "dolfiny.conlin.CONLIN"
    opts["tao_conlin_subsolver_tao_monitor"] = ""
else:  # mma
    opts["tao_python_type"] = "dolfiny.mma.MMA"
    opts["tao_mma_move_limit"] = 0.01
    opts["tao_mma_subsolver_tao_monitor"] = ""

problem = dolfiny.taoproblem.TAOProblem(
    J, [ρ], J=(DJ, ρ.x.petsc_vec.copy()), h=[g], lb=ρ_min, ub=np.float64(1)
)

if comm.size == 1:
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))
    plotter.open_gif("topopt_simp.gif", fps=5)
    pv_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    pv_grid.cell_data[ρ.name] = ρ.x.array
    plotter.add_mesh(pv_grid, scalars=ρ.name, clim=[ρ_min, 1], cmap="coolwarm")
    text = plotter.add_text("")
    plotter.view_xy()


def monitor(tao):
    it = tao.getIterationNumber()
    if comm.size == 1:
        text.SetText(2, f"Iteration {it}")
        pv_grid.cell_data[ρ.name] = ρ.x.array
        plotter.render()
        plotter.write_frame()
    if output:
        for f in (ρ, ρ_f, u):
            with dolfinx.io.XDMFFile(comm, f"topopt_simp/{f.name}_{it}.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(f, it)


problem.tao.setMonitor(monitor)
problem.solve()
problem.tao.view()

if comm.size == 1:
    plotter.close()

# %%[markdown]
# ```{image} topopt_simp.gif
# :alt: Optimisation iterations
# :align: center
# ```

for f in (ρ, ρ_f, u):
    with dolfinx.io.XDMFFile(comm, f"topopt_simp/{f.name}.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(f)
