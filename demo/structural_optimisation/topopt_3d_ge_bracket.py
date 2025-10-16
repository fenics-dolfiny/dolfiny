# %% [markdown]
# # Topology optimisation of a 3D jet engine bracket
#
# This demo solves the classic industrial optimisation problem of a General Electric (GE) jet engine
# bracket under multiple load cases, using the **S**olid **I**sotropic **M**aterial with
# **P**enalisation (SIMP), regularised with a Helmholtz filter.
#
# In particular this demo emphasises
# 1. working with a `STEP` geometry,
# 2. multiple load cases,
# 3. multi-step adjoint computations, and
# 4. the use of custom optimisation solvers.
#
# The
# [GE jet engine bracket challenge](https://grabcad.com/challenges/ge-jet-engine-bracket-challenge)
# was a competition hosted by GrabCAD in 2013. The goal was to design a bracket that connects an
# engine to the frame of an aircraft, minimising its weight while remaining in the elastic regime
# under a set of load cases.
#
# Geometry in the `STEP` file format is provided on the GrabCAD challenge page. We've rotated the
# geometry to align the faces with coordinate axes and fragmented the geometry into multiple volumes
# to disable the optimisation in the bolt areas. This preprocessing was done in
# [`FreeCAD`](https://www.freecad.org/) and is available at the dolfiny github repository
# [here](https://github.com/fenics-dolfiny/dolfiny/tree/main/demo/structural_optimisation/ge_bracket_rotated_nonds.step).
#
# The load cases are defined on the GrabCAD challenge page referenced above, but we include a
# summary here.
#
# ```{figure} https://d2t1xqejof9utc.cloudfront.net/competition_pics/pics/489/large.png?1370977814
# :alt: GE bracket
# :label: fig:ge_bracket
# :width: 400px
# :align: center
# GE jet engine bracket challenge geometry and load cases
# (source: https://grabcad.com/challenges/ge-jet-engine-bracket-challenge).
# ```
#
# We start by importing the necessary modules, reading in the geometry, and generating a mesh with
# `gmsh`. Unfortunately, boolean operations in `FreeCAD` do not always yield valid geometries, so we
# need to call `removeAllDuplicates` before meshing. Otherwise, this results in disconnected
# volumes. Removing duplicates renumbers the faces, so we colour the faces in FreeCAD and extract
# the face tags based on the colour.
#
#
# ```{figure} ge_bracket_colored.png
# :alt: GE bracket STEP with colours
# :label: fig:ge_bracket_step
# :width: 400px
# :align: center
# Coloured `STEP` file pre-processed in `FreeCAD`, used to identify physical groups.
# ```
#
# There are five physical groups:
# - `pin_left` (green faces)
# - `pin_right` (blue faces)
# - `bolt_faces` (red faces)
# - `volume` (the main volume for the optimisation)
# - `bolts` (the bolt volumes where the density is fixed to 1).
#
# The maximum mesh size is set to 3 mm, resulting in approximately 113k tetrahedral elements and 20k
# vertices. To resolve the curved geometry, we've set `Mesh.MeshSizeFromCurvature = 1`, which
# ensures that the mesh is refined based on the local curvature. More precisely, we have at least 8
# elements per $2\pi$. We also use `Netgen` mesh optimisation to improve the mesh quality.
# Additionally, STEP files usually have dimensions in mm, so we ask `gmsh` to interpret the geometry
# in metres by setting `Geometry.OCCTargetUnit = M`.

# %% tags=["hide-output"]
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import ufl

import gmsh
import numpy as np
import pyvista
import sympy.physics.units as syu

import dolfiny
from dolfiny.units import Quantity

comm = MPI.COMM_WORLD

if comm.rank == 0:
    if not gmsh.isInitialized():
        gmsh.initialize()

    # Convert length units of `CAD` geometry to meters.
    gmsh.option.set_string("Geometry.OCCTargetUnit", "M")

    gmsh.open("ge_bracket_rotated_nonds.step")
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    _, _, surfaces, volumes = (gmsh.model.occ.get_entities(d) for d in range(4))

    colors = {
        "red": (255, 0, 0, 255),
        "green": (0, 255, 0, 255),
        "blue": (0, 0, 255, 255),
        "yellow": (255, 255, 0, 255),
    }

    def extract_tags_by_color(a, color):
        return list(ai[1] for ai in a if gmsh.model.get_color(*ai) == color)

    gmsh.model.addPhysicalGroup(3, [1], name="volume")
    gmsh.model.addPhysicalGroup(3, [2, 3, 4, 5], name="bolts")

    gmsh.model.addPhysicalGroup(
        2, extract_tags_by_color(surfaces, colors["green"]), name="pin_left"
    )
    gmsh.model.addPhysicalGroup(
        2, extract_tags_by_color(surfaces, colors["blue"]), name="pin_right"
    )
    gmsh.model.addPhysicalGroup(
        2, extract_tags_by_color(surfaces, colors["red"]), name="bolt_faces"
    )

    gmsh.option.setNumber("Mesh.MeshSizeMax", 3e-3)  # in meters
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)

model = gmsh.model if comm.rank == 0 else None

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, rank=0)
mesh = mesh_data.mesh

pin_left_tag = mesh_data.physical_groups["pin_left"].tag
pin_right_tag = mesh_data.physical_groups["pin_right"].tag
bolt_faces_tags = mesh_data.physical_groups["bolt_faces"].tag

volume_tag = mesh_data.physical_groups["volume"].tag
bolts_tag = mesh_data.physical_groups["bolts"].tag

# %%
if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_mesh(
        grid, show_edges=True, color="white", line_width=dolfiny.pyvista.pixels // 1000
    )
    plotter.show_axes()
    plotter.camera.elevation = 30
    plotter.show()
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ## State problem (linear elasticity)
#
# The next step is to define the elasticity problem. We consider a linear isotropic material model,
# together with the classic SIMP penalisation https://doi.org/10.1007/BF01650949, which interpolates
# the Young's modulus $E$ as $$ E(\hat{\rho}) = \hat{\rho}^p E_0 $$ where $E_0$ is Young's modulus
# of the solid material (associated with the phase $\rho=1$), and $p > 1$ is the *penalty* factor.
# In this demo we set $p=3$.
#
# ```{note}
#   The SIMP penalty factor is an important parameter in the problem formulation.
#   With increasing value, it ensures that intermediate densities $\rho \in (\rho_\text{min}, 1)$
#   are avoided in the final design. It also makes the optimisation problem non-convex.
# ```
#
# Lower bound on the density, i.e. the density of the void phase $\rho_\text{min} = 10^{-3} \leq
# \rho$ is enforced during the optimisation (see below). The lower bound guarantees well-posedness
# of the elasticity problem. Young's modulus of the solid phase follows the material specification
# of Ti-6Al-4V, which is a common aerospace alloy. We've taken the approximate value $E_0 = 110
# \text{ GPa}$ from https://en.wikipedia.org/wiki/Ti-6Al-4V. The Poisson's ratio is set to $\nu =
# 0.31$.
#
# We derive the $i$-th state problem also as a minimisation problem, where the total potential
# energy is minimised. The total potential energy is defined as
#
# $$ \min_{u_i} \Pi(u_i, \hat \rho) = \min_{u_i} \int_\Omega \frac{1}{2} \sigma(u_i, \hat \rho) :
#   \epsilon(u_i) \, d\Omega
#   - \sum_i \int_{\Gamma_i} f_i \cdot u_i \, d\Gamma $$
#
# where $\sigma(u_i, \hat \rho) = \lambda(\hat \rho) \text{ tr}(\epsilon(u_i)) + 2 \mu(\hat \rho)
# \epsilon(u_i)$ is the stress tensor, $\epsilon$ is the small strain tensor, $f_i$ are the applied
# forces on the boundary parts $\Gamma_i$, and $u_i$ is the displacement field for the $i$-th load
# condition. The first term is the elastic energy stored in the deformed body, and the second term
# is the work done by the external forces, which at equilibrium coincides with the compliance.

# %%
tdim = mesh.topology.dim
V_u = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (tdim,)))
V_ρ = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
V_ρ_f = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
ρ = dolfinx.fem.Function(V_ρ, name="density")
ρ_f = dolfinx.fem.Function(V_ρ_f, name="density-filtered")
ρ_f.x.array[:] = 1.0

ρ_min = np.float64(1e-3)
penalty = 3

E0 = Quantity(mesh, 110, syu.giga * syu.pascal, "E0")  # Ti-6Al-4V
E = ρ_f**penalty * E0
nu = 0.31


def ε(u):  # strain
    return ufl.sym(ufl.grad(u))


def σ(u):  # stress
    # Lamé parameters λ and μ
    λ = E * nu / ((1 + nu) * (1 - 2 * nu))
    μ = E / (2 * (1 + nu))
    return λ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * μ * ε(u)


ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)


def compliance(load_condition):
    u = load_condition["u"]
    return sum(
        [ufl.inner(f, u) * m for f, m in zip(load_condition["force"], load_condition["measure"])]
    )


def elastic_energy(load_condition):
    u = load_condition["u"]
    E = 1 / 2 * ufl.inner(σ(u), ε(u)) * dx
    E -= compliance(load_condition)
    return E


assert isinstance(mesh_data.facet_tags, dolfinx.mesh.MeshTags)
assert isinstance(mesh_data.cell_tags, dolfinx.mesh.MeshTags)

fixed_entities = mesh_data.facet_tags.find(bolt_faces_tags)
fixed_dofs = dolfinx.fem.locate_dofs_topological(V_u, tdim - 1, fixed_entities)
bc_u = dolfinx.fem.dirichletbc(np.zeros(tdim, dtype=ScalarType), fixed_dofs, V_u)

V_ρ_f_bolt_dofs = dolfinx.fem.locate_dofs_topological(
    V_ρ_f, tdim, mesh_data.cell_tags.find(mesh_data.physical_groups["bolts"].tag)
)

# %% [markdown]
# In the challenge specification there are four load cases defined. We need to prepare the
# elasticity problem for each load case and store the associated linear problem. This is achieved by
# defining a dictionary with the load case name as the key, and the load vectors, measure, the
# function to store the solution, and later the `LinearProblem`.
#
# The first load case is a vertical upward force of $8000 \text{ lbs}$ on the pin faces. The Neumann
# boundary force in the weak form is the force per area, so we need to compute the area of the pin
# faces. In addition, we convert pounds-force to newtons.
#
# The second and third load cases are horizontal and 42 degree angled forces of approximately $8500
# \text{ lbs}$ and $9500 \text{ lbs}$, respectively.
#
# The fourth load case is a torsional load, which we model as a couple of forces in opposite
# directions of $5000 \text{ lbs} \cdot \text{in}$ on the pin faces. We assume a lever arm of $0.01
# \text{ m}$, which we measured as the approximate distance from the centre of the pin holes to the
# axis around which the torsion acts (midpoint between the pin holes). This couple of forces is the
# reason why we need to store the force and measure as lists, to allow for multiple Neumann
# conditions per load case.
#
# ```{note}
#   Even though the state problem is linear (wrt. load cases),
#   we need to solve it for each load case separately, since the total objective
#   is not linear in the load cases. In other words, the superposition principle
#   does not hold for the compliance objective.
# ```
#
# Dirichlet boundary conditions are the same for all load cases and correspond to fixing the bolt
# faces.
#
# We can use convenient unit handling provided by the `dolfiny.units.Quantity` class to convert the
# imperial units to SI units. This will happen automatically, since when we create a `Quantity`, the
# value gets internally converted to base SI units.
#
# ```{note}
#   The representation of the load cases with the above forces is very crude here.
#   The GrabCAD challenge explicitly states that the right and left pin holes are connected
#   with an infinitely stiff pin.
#   These conditions can be enforced using e.g. [`dolfinx_mpc`](https://github.com/jorgensd/dolfinx_mpc),
#   but we avoid this complexity in this demo.
# ```


# %%
pin_area = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1.0 * ds((pin_left_tag, pin_right_tag), domain=mesh))
    )
)

g = Quantity(mesh, 9.81, syu.meter / syu.second**2, "g")
torsion_lever_arm = Quantity(mesh, 0.01, syu.meter, "torsion_lever_arm")
pin_area = Quantity(mesh, pin_area, syu.meter**2, "pin_area")

F1 = Quantity(mesh, 8000, syu.pound, "F1")
F2 = Quantity(mesh, 8500, syu.pound, "F2")
F3 = Quantity(mesh, 9500, syu.pound, "F3")
F4 = Quantity(mesh, 5000, syu.pound * syu.inch, "F4")

load_conditions = {
    "vertical_up": {
        "force": [F1 * g / pin_area * ufl.as_vector((0, 0, 1))],
        "measure": [ds((pin_left_tag, pin_right_tag))],
        "u": dolfinx.fem.Function(V_u),
    },
    "horizontal_out": {
        "force": [F2 * g / pin_area * ufl.as_vector((0, -1, 0))],
        "measure": [ds((pin_left_tag, pin_right_tag))],
        "u": dolfinx.fem.Function(V_u),
    },
    "42deg_vertical_out": {
        "force": [
            F3 * g / pin_area * ufl.as_vector((0, -np.sin(np.deg2rad(42)), np.cos(np.deg2rad(42))))
        ],
        "measure": [ds((pin_left_tag, pin_right_tag))],
        "u": dolfinx.fem.Function(V_u),
    },
    "torsion": {
        "force": [
            F4 / torsion_lever_arm * g / pin_area / 2 * ufl.as_vector((0, -1, 0)),
            F4 / torsion_lever_arm * g / pin_area / 2 * ufl.as_vector((0, 1, 0)),
        ],
        "measure": [ds(pin_left_tag), ds(pin_right_tag)],
        "u": dolfinx.fem.Function(V_u),
    },
}


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity

    Copied from https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_elasticity.py
    """

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [dolfinx.la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    dolfinx.la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm) for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


ns = build_nullspace(V_u)

for lc in load_conditions.values():
    u_lc = lc["u"]
    a = ufl.derivative(
        ufl.derivative(elastic_energy(lc), u_lc),
        u_lc,
    )
    L = -ufl.derivative(elastic_energy(lc), u_lc)
    L = ufl.replace(L, {u_lc: ufl.as_vector((0, 0, 0))})

    elas_prob = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc_u],
        u=u_lc,  # type: ignore
        petsc_options=(
            {
                # Combination of https://github.com/FEniCS/performance-test and https://doi.org/10.1007/s00158-020-02618-z
                "ksp_error_if_not_converged": True,
                "ksp_type": "cg",
                "ksp_rtol": 1.0e-6,
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "pc_gamg_agg_nsmooths": 1,
                "pc_gamg_threshold": 0.001,
                "mg_levels_esteig_ksp_type": "cg",
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_ksp_chebyshev_esteig_steps": 50,
                "mg_levels_pc_type": "sor",
                "pc_gamg_coarse_eq_limit": 1000,
            }
        ),
        petsc_options_prefix="elasticity_ksp",
    )
    elas_prob._A.setNearNullSpace(ns)
    lc["linear_problem"] = elas_prob  # type: ignore

    J_form = dolfinx.fem.form(compliance(lc))
    DJ_form = dolfinx.fem.form(-ufl.derivative(elastic_energy(lc), ρ_f))
    lc["J_form"] = J_form
    lc["DJ_form"] = DJ_form

# %% [markdown]
# We can solve one of the load cases to visualise the displacement field. Below is the displacement
# field for the torsional load case, converted to mm and scaled by a factor of 0.2.

# %%
# Solve all load cases
for name in load_conditions.keys():
    load_conditions[name]["linear_problem"].solve()  # type: ignore

# Plot subplots for all load cases
if comm.size == 1:
    plotter = pyvista.Plotter(theme=dolfiny.pyvista.theme, shape=(2, 2))

    for i, name in enumerate(load_conditions.keys()):
        plotter.subplot(i // 2, i % 2)
        u_lc = load_conditions[name]["u"]
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_lc.function_space.mesh))  # type: ignore

        assert isinstance(u_lc, dolfinx.fem.Function)
        grid.point_data["u"] = u_lc.x.array.reshape((-1, 3)) * 1000  # displacement in mm
        grid_warped = grid.warp_by_vector("u", factor=0.2)

        plotter.add_mesh(
            grid_warped,
            show_edges=True,
            n_colors=10,
            scalar_bar_args={"title": "Displacement [mm]"},
        )
        plotter.add_text(
            name,
            font_size=dolfiny.pyvista.pixels // 100,
            font="courier",
            position="lower_edge",
        )
        plotter.camera.elevation = -15
        plotter.show_axes()

    plotter.show()
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ## Filtering
#
# We use a Helmholtz filter on the density field, first introduced by
# https://doi.org/10.1002/nme.3072 in the context of topology optimisation.
#
# In short, for a given density $\rho$, we solve a (positive definite) Helmholtz equation, yielding
# the filtered density $\hat{\rho}$:
#
# $$ \int_\Omega r^2 \nabla \hat{\rho} \cdot \nabla \tau
#   + \hat{\rho} \tau \ \text{d}x
#   + \int_\Gamma r \hat{\rho} \tau \ \text{d}s = \int_\Omega \rho \tau \ \text{d}x \qquad \forall
#   \tau \in V_{\hat{\rho}}. $$
#
# Here $r$ is a parameter that controls the filter radius, we choose $r$ to be dependent on the
# maximum cell diameter. In addition to the volumetric term, we also include a boundary term, which
# acts like a penalisation towards zero Neumann conditions, i.e., it prevents the density from
# sticking to the boundary, see https://doi.org/10.1007/s00158-020-02556-w. The boundary $\Gamma$ is
# here defined as all boundary facets except those with Dirichlet and Neumann conditions applied.
#
# Since the Helmholtz equation is self-adjoint and we need to evaluate its adjoint for the gradient
# computation later on, we set up the solver to allow for handling of generic right-hand sides. Thus
# we only have one linear solver and one operator matrix stored for both the forward and adjoint
# problems.
#
# In addition to the filter, we need to enforce that the density equals 1 in the bolt volumes. This
# is achieved by setting the corresponding degrees of freedom in the filtered density function to 1
# after applying the filter. We also set the unfiltered density $\rho$ to 1 in the bolt volumes when
# assembling the right-hand side of the filter problem. Setting the density to 1 in the bolt volumes
# is important from a practical point of view, as the bolts need to be in contact with solid
# material.

# %%
num_cells = mesh.topology.index_map(tdim).size_local
h = mesh.h(tdim, np.arange(0, num_cells))
hmax = comm.allreduce(h.max(), MPI.MAX)

r = 0.5 * hmax
u_f, v_f = ufl.TrialFunction(V_ρ_f), ufl.TestFunction(V_ρ_f)
a_filter = dolfinx.fem.form(
    r**2 * ufl.inner(ufl.grad(u_f), ufl.grad(v_f)) * dx
    + u_f * v_f * dx
    + r * ufl.inner(u_f, v_f) * ds
    - r * ufl.inner(u_f, v_f) * ds((pin_left_tag, pin_right_tag, bolt_faces_tags))
)
L_filter_ρ = dolfinx.fem.form(ρ * v_f * dx(volume_tag) + 1.0 * v_f * dx(bolts_tag))
s = dolfinx.fem.Function(V_ρ_f, name="s")
L_filter_s = dolfinx.fem.form(s * v_f * dx)

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
# With the state and filtering problems defined, we can define the objective and gradient of the
# (reduced) optimisation problem.
#
# The objective, to be minimised, is the *total compliance for all load cases*:
#
# $$ \min_{\hat \rho} \sum_i \int_{\Gamma_i} f_i \cdot u_i \ \text{d}\Gamma, $$
#
# which is the sum of the compliance for each load case. This choice (i.e., to sum the individual
# compliances) is one of many possible options for how to account for multiple load cases. Other
# options include minimising the maximum compliance or minimising a weighted sum of compliances. For
# simplicity, we follow the sum approach, as it is linear (wrt. load cases), and thus the adjoint
# problems can be solved independently.
#
# ```{note}
#   Minimisation of multiple load cases is a task of so-called multi-objective optimisation.
#   Multiple objectives can be combined into a single objective with a process called
#   scalarisation, see e.g.
#   [Convex Optimization, ch. 4.7.4](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
# ````
#
# We constrain the density to lower and upper bounds:
#
# $$ \rho_\text{min} \leq \rho \leq 1, $$
#
# and constrain the volume of the design to a volume fraction $V_f \in (0, 1)$:
#
# $$ \frac{1}{\text{Vol} (\Omega)} \int_\Omega \rho \ \text{d}x \leq V_f. $$
#
# The optimisation problem is stated in reduced form in $\rho$. So $\hat{\rho}$ and $u$ only appear
# as intermediates. Gradients are then computed through the adjoint formulation. Since the total
# compliance is the sum of the compliance for each load case, the gradient is also the sum of the
# gradients for each load case.
#
# The volume fraction in this demo was chosen as 0.3 (30% of the active part of the mesh). Note that
# the GrabCAD challenge does not specify any value for the volume fraction. The goal is to produce
# shapes as light as possible under stress limit constraints. This would require a different
# optimisation setup and is out of the scope of this demo.

# %% tags=["hide-output"]
mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * dx(volume_tag)))
)
volume_fraction = ρ / mesh_volume * dx(volume_tag)
max_volume_fraction = 0.3

g = volume_fraction <= max_volume_fraction

ρ.x.array[:] = max_volume_fraction
ρ_f.interpolate(ρ)

apply_filter(L_filter_ρ, ρ_f)
ρ_f.x.array[V_ρ_f_bolt_dofs] = 1.0


@dolfiny.taoproblem.sync_functions([ρ])
def J(_tao, _):
    apply_filter(L_filter_ρ, ρ_f)
    ρ_f.x.array[V_ρ_f_bolt_dofs] = 1.0
    ρ_f.x.array[:] = np.clip(ρ_f.x.array, ρ_min, 1.0)

    total = 0.0
    for lc_name, lc in load_conditions.items():
        lc["linear_problem"].solve()

        comp = comm.allreduce(dolfinx.fem.assemble_scalar(lc["J_form"]))
        if comm.rank == 0:
            print(f"Objective ({lc_name}): {comp:.4g}")

        total += comp
    return total


Dρ = dolfinx.fem.Function(V_ρ_f)
z = dolfinx.fem.Function(V_ρ_f, name="z")
tmpDG0 = dolfinx.fem.Function(V_ρ)
s_lc_vec = s.x.petsc_vec.copy()  # vector to store the gradient contrib. from each load case


@dolfiny.taoproblem.sync_functions([ρ])
def DJ(_tao, _, G):
    with s.x.petsc_vec.localForm() as s_local, s_lc_vec.localForm() as s_lc_local:
        s_local.set(0.0)
        s_lc_local.set(0.0)

    for _lc_name, lc in load_conditions.items():
        dolfinx.fem.petsc.assemble_vector(s_lc_vec, lc["DJ_form"])
        s_lc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        s_lc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with s.x.petsc_vec.localForm() as s_local, s_lc_vec.localForm() as s_lc_local:
            s_local += s_lc_local
            s_lc_local.set(0.0)

    # Apply adjoint to DJ/s -> z.
    apply_filter(L_filter_s, z)
    z.x.array[V_ρ_f_bolt_dofs] = 0.0

    # Interpolate/project z into DG0.
    tmpDG0.interpolate(z)

    # Copy to G.
    tmpDG0.x.petsc_vec.copy(G)
    G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


opts = PETSc.Options()  # type: ignore
opts["tao_type"] = "python"
opts["tao_monitor"] = ""
opts["tao_max_it"] = (max_it := 50)

opts["tao_python_type"] = "dolfiny.mma.MMA"
opts["tao_mma_move_limit"] = 0.05
opts["tao_mma_subsolver_tao_monitor"] = ""

problem = dolfiny.taoproblem.TAOProblem(
    J, [ρ], J=(DJ, ρ.x.petsc_vec.copy()), h=[g], lb=ρ_min, ub=np.float64(1)
)
problem.solve()

with dolfinx.io.XDMFFile(comm, "ge_bracket/result.xdmf", "w") as file:
    file.write_mesh(mesh)
    for f in (ρ, ρ_f, load_conditions["vertical_up"]["u"]):
        file.write_function(f)

# %% [markdown]
# ## Post-processing
# Once the optimisation is done, we can visualise the results. We use `pyvista` for the
# post-processing. Below is the filtered density field, clipped at 0.5 to visualise the solid
# structure.

# %%
if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(ρ_f.function_space.mesh))

    grid.point_data["density-filtered"] = ρ_f.x.array
    grid_clipped = grid.clip_scalar(scalars="density-filtered", invert=False, value=0.5)

    plotter = pyvista.Plotter(
        off_screen=True,
        theme=dolfiny.pyvista.theme,
        shape=(1, 2),
        window_size=(dolfiny.pyvista.pixels, dolfiny.pyvista.pixels // 2),
    )

    plotter.add_mesh(grid_clipped, color="white")
    plotter.show_axes()
    plotter.camera.elevation = 30

    plotter.subplot(0, 1)
    plotter.add_mesh(grid_clipped, color="white")
    plotter.show_axes()
    plotter.view_xz()
    plotter.camera.azimuth = 180

    plotter.show()
    plotter.close()
    plotter.deep_clean()

# %%
