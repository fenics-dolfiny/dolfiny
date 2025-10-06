# %% [markdown]
# # Topology optimisation
#
# This demo showcases the mother of all topology optimisation problems: compliance minimisation of a
# **s**olid **i**sotropic **m**aterial with **p**enalisation (SIMP), regularised with a Helmholtz
# filter.
#
# In particular this demo emphasizes
# 1. the use of custom optimisation solvers and
# 2. multi-step adjoint computations.
#
# %% tags=["hide-input"]
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
# For the topological dimension $d \in \{ 2, 3\}$ the computational domain is given by
# $\Omega = (0, 2)^{d-1} \times (0,1) \subset \mathbb{R}^d$ and is
# discretized by $(2n, n)$ quadrilateral or $(2n, n, n)$ hexahedral elements respectively, yielding
# a tesseltation $\mathcal{T}$.
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
# ## Problem formulation
#
# We define three function spaces associated with *density* $\rho$, *filtered-density* $\hat{\rho}$,
# and displacement $u$:
# 1. $V_\rho = \mathcal{P}_0 (\mathcal{T})$
# 2. $V_{\hat{\rho}} = \mathcal{P}_1 (\mathcal{T})$
# 3. $V_u = \mathcal{P}_1^3 (\mathcal{T})$.
#
# ```{note}
#   Later we will introduce a filter, i.e. a smoothing operation, which produces for a given $\rho$
#   a filtered/regularised $\hat{\rho}$.
#   Therefore the space $V_{\hat{\rho}}$ has higher regularity than $V_\rho$.
# ```
# %%
V_u = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (tdim,)))
V_ρ = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
V_ρ_f = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

ρ = dolfinx.fem.Function(V_ρ, name="density")
ρ_f = dolfinx.fem.Function(V_ρ_f, name="density-filtered")
u = dolfinx.fem.Function(V_u, name="u")

# %% [markdown]
# ## State problem (elasticity)
#
# The next step is to define the elasticity problem.
# We consider a linear isotropic material model, together with classic SIMP penalisation
# https://doi.org/10.1016/0045-7825(88)90086-2, which defines the Young's modulus E as
# $$
#   E(\hat{\rho}) = (\rho_\text{min} + (1-\rho_\text{min}) \hat{\rho}^p) E_0
# $$
# where $E_0$ is Young's modulus of the solid material (associated with the phase $\rho=1$), and
# $p > 1$ is the *penalty* factor.
#
# ```{note}
#   The penalty factor is a critical parameter to the problem formulation.
#   It ensures, with increasing value, that intermediate densities $\rho \in (\rho_\text{min}, 1)$
#   are avoided in the final design.
#   At the same time, for $p$ larger the problem becomes more and more non-linear and harder to
#   solve.
# ```
#
# As boundary conditions we fix the $x=0$ plane of the design and apply a constant force $f$
# (Neumann boundary condition) on the center of the facet at $x=2$.
# %%
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
# %% [markdown]
# ## Filtering
#
# We use a Helmholtz filter on the density field, first introduced by
# https://doi.org/10.1002/nme.3072 in the context of topology optimisation.
#
# In short, this boils down to solving for a given density $\rho$ a Helmholtz equation, yielding the
# filtered-density $\hat{\rho}$
#
# $$
#   \int_\Omega r^2 \nabla \hat{\rho} \cdot \nabla \tau + \hat{\rho} \tau \ \text{d}x
#   = \int_\Omega \rho \tau \ \text{d}x
#   \qquad \forall \tau \in V_{\hat{\rho}}.
# $$
#
# $r$ is a parameter that controls the filter radius, we choose $r$ to be dependent on the local
# cell diameter.
#
# ```{note}
#   The filter radius $r$ will be a constant for a uniformly refined mesh, but will capture
#   correctly the different scales of locally refined meshes.
# ```
#
# Since the Helmholtz equation is self-adjoint and we need to evaluate the adjoint of it for the
# gradient computation later on, we set up the solver to allow for handling of generic right hand
# sides.
# Thus we only have one linear solver and operator matrix stored for both forward and adjoint
# problem.
#
# %%
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


# %% [markdown]
# ## Optimisation problem
#
# With the state and filtering problems defined we can define the objective and gradient of the
# (reduced) optimisation problem.
#
# The objective, to be minimised, is *compliance*
#
# $$
#   \int_\Omega f \cdot u \ \text{d}x.
# $$
#
# We constrain the density to lower and upper bounds
#
# $$
#   \rho_\text{min} \leq \rho \leq 1,
# $$
#
# and the volume of the design to a volume fraction $V_f \in (0, 1)$
#
# $$
#   \frac{1}{\text{Vol} (\Omega)} \int_\Omega \rho \ \text{d}x \leq V_f.
# $$
#
# The the optimisation problem is stated in reduced form in $\rho$.
# So, $\hat{\rho}$ and $u$ only appear as intermediates.
# Gradients are then computed through the adjoint formulation.
#
# %%

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


# %% [markdown]
# ## Custom optimisation routines
#
# For the optimisation we rely on our custom implementations of the Method of Moving Asymptotes
# (MMA) https://doi.org/10.1002/nme.1620240207 or Convex Linearisation (CONLIN) https://doi.org/10.1007/BF01637664.
#
# %%
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

# %% tags=["hide-input"]
if comm.size == 1:
    plotter = pv.Plotter(off_screen=False, window_size=(1024, int(512 * 1.4)))
    plotter.open_gif("topopt_simp.gif", fps=5)
    pv_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    pv_grid.cell_data[ρ.name] = ρ.x.array
    plotter.add_mesh(pv_grid, scalars=ρ.name, clim=[ρ_min, 1], cmap="coolwarm")
    text = plotter.add_text("")
    plotter.view_xy()
    plotter.camera.zoom(1.5)

    plotter_f = pv.Plotter(off_screen=True, window_size=(1024, int(512 * 1.4)))
    plotter_f.open_gif("topopt_simp_filtered.gif", fps=5)
    pv_grid.point_data[ρ_f.name] = ρ_f.x.array
    plotter_f.add_mesh(pv_grid, scalars=ρ_f.name, clim=[ρ_min, 1], cmap="coolwarm")
    text_f = plotter_f.add_text("")
    plotter_f.view_xy()
    plotter_f.camera.zoom(1.5)


def monitor(tao):
    it = tao.getIterationNumber()
    if comm.size == 1:
        text.SetText(0, f"Iteration {it}")
        pv_grid.cell_data[ρ.name] = ρ.x.array
        plotter.render()
        plotter.write_frame()

        text_f.SetText(0, f"Iteration {it}")
        pv_grid.point_data[ρ_f.name] = ρ_f.x.array
        plotter_f.render()
        plotter_f.write_frame()

    if not output:
        return

    with dolfinx.io.XDMFFile(comm, f"topopt_simp/data_{it}.xdmf", "w") as file:
        file.write_mesh(mesh)
        for f in (ρ, ρ_f, u):
            file.write_function(f, it)


# %% tags=["hide-output"]
problem.tao.setMonitor(monitor)
problem.solve()

# %% tags=["hide-input"]
if comm.size == 1:
    plotter.close()
    plotter_f.close()

with dolfinx.io.XDMFFile(comm, "topopt_simp/data.xdmf", "w") as file:
    file.write_mesh(mesh)
    for f in (ρ, ρ_f, u):
        file.write_function(f)

# %% [markdown]
# ## Results
#
# We plot the {ref}`density<gif-density>` and {ref}`filtered-density<gif-filtered-density>` across
# MMA iterations.
#
#
# The density field drives for a 'black/white' design, i.e. for a discrete split into phase
# ($\rho=1$) and void ($\rho=0$).
# This results (due to the nature of discretization) in heavy stair-casing to occur along the phase
# interface.
#
#
# In contrast the filtered-density (computed from the corresponding density field in every
# iteration) smears our the interface and thus will never result in a 'black/white' design.
#
# ```{image} topopt_simp.gif
# :alt: Optimisation iterations
# :align: center
# :name: gif-density
#
# Density $\rho$ across throughout the MMA iterations.
#
# ```
#
# ```{image} topopt_simp_filtered.gif
# :alt: Optimisation iterations
# :align: center
# :name: gif-filtered-density
#
# Filtered-density $\hat{\rho}$ across throughout the MMA iterations.
#  ```
#
# ```{important}
#   We want to stress that both outputs are relevant for the interpretation of the obtained result.
#   Having pure phases in the density field is what one is naturally interested in, there is no
#   immediate physical interpretation of an intermediate density.
#   However no one will manufacture a voxelized design, the filtered-density allows for the
# extracion of smoothed designs (by considering iso-contours of the density field).
# ```
