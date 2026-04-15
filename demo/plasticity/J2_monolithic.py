# %% [markdown]
# # Rate-independent monolithic J2 plasticity
#
# This demo solves a quasi-static finite-strain boundary value problem in 3D with
# monolithic $J_2$ (von Mises) plasticity, isotropic- and kinematic hardening.
# The unknown displacement $u$, plastic strain tensor $P$, and hardening variables $h$ and $B$
# are solved simultaneously in one coupled nonlinear system.
# The constitutive setting is in the spirit of unified cyclic-plasticity models for metals,
# see {cite:t}`Tirpitz1992`.
#
# In particular, this demo emphasizes:
# - monolithic solution of a multi-field problem in one coupled nonlinear system,
# - quadrature-element discretisation of plastic strain and hardening variables, and
# - cyclic finite-strain plasticity with both isotropic and kinematic hardening.
#
# ---
#
# ## Model
#
# The kinematics and additive split are
# $$
#   F = I + \nabla u, \qquad E = \tfrac{1}{2}(F^T F - I) = E_{\mathrm{el}} + P.
# $$
# Here $I$ is the identity tensor, $\nabla$ denotes the gradient with respect to the reference
# coordinates, $F$ is the deformation gradient, $E$ is the total Green-Lagrange strain,
# $E_{\mathrm{el}}$ is the elastic strain, and $P$ is the plastic strain tensor. The constitutive
# law that relates the 2nd Piola-Kirchhoff stress $S$ to the elastic strain is the
# St. Venant-Kirchhoff material model:
# $$
#   S = 2\mu E_{\mathrm{el}} + \lambda \text{tr}(E_{\mathrm{el}}) I.
# $$
# Yielding is driven by
# $$
#   f(S, B, h) = \sqrt{3 J_2(S - B)} - (S_y + h) \le 0
# $$
# where $\mu$ and $\lambda$ are the Lame constants. The variables $B$ and $h$ denote the
# kinematic and isotropic hardening contributions, $S_y$ is the initial yield stress, and
# $J_2(A) = \tfrac{1}{2} \text{dev}(A) : \text{dev}(A)$ is the second deviatoric
# invariant, where $\text{dev}(A)$ denotes the deviatoric part of $A$ and $A : B$ is the
# Frobenius product.
#
# The hardening follows the standard linear evolution equations
# $$
#   \dot{h} = \dot{\lambda} b_h (q_h - h), \qquad
#   \dot{B} = \dot{\lambda} b_b (q_b \partial_S g - B)
# $$
# where $\dot{\lambda}$ is the plastic multiplier rate and $b_h$, $q_h$, $b_b$, and $q_b$
# are material parameters. For associated flow, the plastic potential equals the yield function
# ($g=f$),
# $$
#   \dot{P} = \dot{\lambda} \frac{\partial g}{\partial S}, \qquad
#   \dot{\lambda} \ge 0, \quad f \le 0, \quad \dot{\lambda} f = 0
# $$
#
# ## Monolithic weak formulation
#
# Instead of predictor-corrector return mapping, the full system is solved directly in weak form.
# Complementarity is encoded by the Macaulay bracket
# $$
#   \Delta\lambda = [f]_+ = \max(f, 0),
# $$
# and the time-discrete weak form requires, for all test functions
# $(\delta u, \delta P, \delta h, \delta B)$,
# \begin{align}
#   \int_\Omega \delta E : S \,\text{d}x &= 0, \\
#   \int_\Omega \delta P : \big[(P - P_0) - \Delta\lambda \, \partial_S g\big]
#     \,\text{d}x &= 0, \\
#   \int_\Omega \delta h \big[(h - h_0) - \Delta\lambda b_h (q_h - h)\big]
#     \,\text{d}x &= 0, \\
#   \int_\Omega \delta B
#   : \big[(B - B_0) - \Delta\lambda b_b (q_b \partial_S g - B)\big] \,\text{d}x &= 0
# \end{align}
# Here $\Omega$ is the reference configuration, $\delta E$ is the variation of the
# Green-Lagrange strain induced by $\delta u$, and $\partial_S g$ is shorthand for
# $\partial g / \partial S$. Subscript $0$ denotes the converged state from the previous load
# step.
#
# ```{note}
# The monolithic formulation with explicit formula for the plastic multiplier
# increment $\Delta \lambda$ is possible since we use the simple $J_2$ plasticity model,
# where the return mapping has a closed-form solution. More complex models
# with, e.g., non-associative flow or non-smooth yield surfaces require a nonlinear
# solution strategy with a Newton iteration at the Gauss point level to solve
# for $\Delta \lambda$ and the internal variables, see our [Rankine plasticity demo](rankine.ipynb).
# ```
#
# ## Parameters and mesh
#
# We model a standard dog-bone tensile specimen (ISO 6892) with gauge length $l_0 = 100$ mm
# and diameter $d_0 = 20$ mm. Physical groups identify the gauge volume and the two grip
# surfaces where Dirichlet boundary conditions are applied.
#
# Material parameters correspond to a generic structural metal: $\mu = 100$ GPa,
# $\lambda = 10$ GPa, initial yield stress $S_y = 0.3$ GPa. The hardening moduli set the
# post-yield shape:
# - Isotropic: $b_h = 20$, $q_h = 0.1$ GPa — flow stress saturates at $S_y + q_h = 0.4$ GPa.
# - Kinematic: $b_b = 250$, $q_b = 0.1$ GPa — rapid back-stress saturation drives the
#   Bauschinger effect.
# Related cyclic mild-steel and structural-steel modelling contexts are discussed in
# {cite:t}`Kowalsky2012` and {cite:t}`Heinrich2022`.
#
# A maximum axial strain $\varepsilon_{\max} = 1\%$ is applied in $K = 10$ increments per
# phase over $Z = 2$ tension-compression cycles (load, unload, reverse, recover). Later on,
# the scalar load factor $\mu \in [-1, 1]$ scales the reference boundary displacement field.

# %% tags=["hide-input", "hide-output"]
from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
import mesh_iso6892_gmshapi as mg
import numpy as np
import pyvista

import dolfiny

# Basic settings
name = "J2_monolithic"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
l0, d0 = 0.10, 0.02  # [m]
nr = 5
# Geometry and physics ansatz order
o, p = 1, 1

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mg.mesh_iso6892_gmshapi(name, l0, d0, nr, order=o)

# Create the mesh of the specimen with given dimensions and save as msh, then read into gmsh model
# mg.mesh_iso6892_gmshapi(name, l0, d0, nr, order=o, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
domain_gauge = mesh_data.physical_groups["domain_gauge"].tag
surface_1 = mesh_data.physical_groups["surface_grip_left"].tag
surface_2 = mesh_data.physical_groups["surface_grip_right"].tag

if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_mesh(
        grid, show_edges=True, color="white", line_width=dolfiny.pyvista.pixels // 1000
    )
    plotter.show_axes()
    plotter.camera.elevation = 15
    plotter.screenshot("J2_monolithic_mesh.png")
    plotter.close()
    plotter.deep_clean()

# Solid: material parameters
mu = dolfinx.fem.Constant(mesh, scalar(100.0))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
la = dolfinx.fem.Constant(mesh, scalar(10.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
Sy = dolfinx.fem.Constant(mesh, scalar(0.300))  # initial yield stress [GPa]
bh = dolfinx.fem.Constant(mesh, scalar(20.00))  # isotropic hardening: saturation rate  [-]
qh = dolfinx.fem.Constant(mesh, scalar(0.100))  # isotropic hardening: saturation value [GPa]
bb = dolfinx.fem.Constant(mesh, scalar(250.0))  # kinematic hardening: saturation rate  [-]
qb = dolfinx.fem.Constant(mesh, scalar(0.100))  # kinematic hardening: saturation value [GPa]

# Solid: load parameters
μ = dolfinx.fem.Constant(mesh, scalar(1.0))  # load factor


def u_bar(x):
    eps_max = 0.01
    return μ.value * np.array([l0 * eps_max * np.sign(x[0]), 0.0 * x[1], 0.0 * x[2]])


# %% [markdown]
# ```{figure} J2_monolithic_mesh.png
# :alt: ISO 6892 dog-bone tensile specimen mesh used for the J2 plasticity simulation.
# :align: center
# :label: fig-j2-mesh
#
# ISO 6892 dog-bone tensile specimen mesh used for the J2 plasticity simulation.
# ```


# %% [markdown]
# ## Finite element discretisation and weak form
#
# Displacement $u$ uses continuous $P_1$ Lagrange elements. Internal variables $P$, $h$, and $B$
# use quadrature elements, i.e. Gauss-point DOFs without inter-element continuity. This is the
# natural representation for history-dependent constitutive variables. The associated test
# functions are denoted by $(\delta u, \delta P, \delta h, \delta B)$ in the mixed weak form.
#
# With `quad_degree = p = 1`, one Gauss point per linear tetrahedron gives piecewise-constant
# internal fields, consistent with the $P_1$ displacement approximation. The measure `dx`
# therefore represents cell integration with that quadrature rule.
#
# The $B$ test space uses `Tf.clone()` so `dolfiny` assembles separate Jacobian block rows for
# the test directions corresponding to $P$ and $B$.

# %% tags=["hide-input"]
quad_degree = p
dx = ufl.Measure(
    "dx",
    domain=mesh,
    subdomain_data=mesh_data.cell_tags,
    metadata={"quadrature_degree": quad_degree},
)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
He = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=quad_degree)
Te = basix.ufl.blocked_element(He, shape=(mesh.geometry.dim, mesh.geometry.dim), symmetry=True)

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Tf = dolfinx.fem.functionspace(mesh, Te)
Hf = dolfinx.fem.functionspace(mesh, He)
Sf = dolfinx.fem.functionspace(mesh, ("DG", 0))

# Define functions
u = dolfinx.fem.Function(Uf, name="u")  # displacement
P = dolfinx.fem.Function(Tf, name="P")  # plastic strain
h = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening
B = dolfinx.fem.Function(Tf, name="B")  # kinematic hardening

u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
P0 = dolfinx.fem.Function(Tf, name="P0")
h0 = dolfinx.fem.Function(Hf, name="h0")
B0 = dolfinx.fem.Function(Tf, name="B0")

S0 = dolfinx.fem.Function(Tf, name="S0")  # stress, previous converged solution (load step)

u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, defines state at boundary
eps_p = dolfinx.fem.Function(Sf, name="eps_p")  # projected plastic strain magnitude
# for output
Po = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="P")
Bo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="B")
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0, (3, 3))), name="S")
ho = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DP", 0)), name="h")

# clone to be distinct from Tf
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Tf, Hf, Tf.clone()))
δu, δP, δh, δB = δm

# Define state and variation of state as (ordered) list of functions
m = [u, P, h, B]


def rJ2(A):
    """Square root of J2 invariant of tensor A: sqrt(1/2 * dev(A):dev(A))"""
    J2 = 1 / 2 * ufl.inner(ufl.dev(A), ufl.dev(A))
    rJ2 = ufl.sqrt(J2)
    return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


# Configuration gradient
I = ufl.Identity(3)  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
E = 1 / 2 * (F.T * F - I)  # E = E(F), total Green-Lagrange strain
E_el = E - P  # E_el = E - P, elastic strain = total strain - plastic strain

# Stress
S = 2 * mu * E_el + la * ufl.tr(E_el) * I  # S = S(E_el), PK2, St.Venant-Kirchhoff

# Wrap variable around expression (for diff)
S, B, h = ufl.variable(S), ufl.variable(B), ufl.variable(h)

# Yield function
f = ufl.sqrt(3) * rJ2(S - B) - (Sy + h)  # von Mises criterion (J2), with hardening

# Plastic potential
g = f

# Derivative of plastic potential wrt stress
dgdS = ufl.diff(g, S)

# Total differential of yield function, used for checks only
df = (
    +ufl.inner(ufl.diff(f, S), S - S0)
    + ufl.inner(ufl.diff(f, h), h - h0)
    + ufl.inner(ufl.diff(f, B), B - B0)
)

# Unwrap expression from variable
S, B, h = S.expression(), B.expression(), h.expression()

# Variation of Green-Lagrange strain
δE = ufl.derivative(E, m, δm)

# Plastic multiplier (J2 plasticity: closed-form solution for return-map)
dλ = ufl.max_value(f, 0)  # ppos = Macaulay bracket

# Weak form (as one-form)
form = (
    ufl.inner(δE, S) * dx
    + ufl.inner(δP, (P - P0) - dλ * dgdS) * dx
    + ufl.inner(δh, (h - h0) - dλ * bh * (qh * 1.00 - h)) * dx
    + ufl.inner(δB, (B - B0) - dλ * bb * (qb * dgdS - B)) * dx
)

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# %% [markdown]
# ## Solver setup and execution
#
# PETSc SNES with Newton line-search handles the non-smooth system.
# The Jacobian is **non-symmetric**: the $\partial\Delta\lambda/\partial S$ coupling term
# from the Macaulay bracket breaks self-adjointness, ruling out symmetric preconditioners.
# We use a sparse direct $LU$ factorisation via MUMPS.
#
# Boundary conditions are re-applied each step by scaling the reference field `u_bar` by
# $\mu$. Here `u_bar` prescribes an axial displacement profile on the two grip surfaces, while all
# other boundaries remain traction-free.

# %% tags=["hide-input", "hide-output"]
# Create output xdmf file - open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh data
ofile.write_mesh_data(mesh_data)

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "bt"
opts["snes_linesearch_monitor"] = ""
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-08
opts["snes_stol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"  # NOTE: this monolithic formulation is not symmetric
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_1_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, mesh_data.facet_tags, surface_1)
surface_2_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, mesh_data.facet_tags, surface_2)

# Book-keeping of results
results: dict[str, list[float]] = {"E": [], "S": [], "P": [], "μ": []}

# Set up load steps
K = 10  # number of steps per load phase
Z = 2  # number of cycles
load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
cycle = np.concatenate((load, unload, -load, -unload))
cycles = np.concatenate([cycle] * Z)

# Process load steps
for step, factor in enumerate(cycles):
    # Set current load factor
    μ.value = factor

    dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    # Update values for given boundary displacement
    u_.interpolate(u_bar)

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.dirichletbc(u_, surface_1_dofs_Uf),  # disp left
        dolfinx.fem.dirichletbc(u_, surface_2_dofs_Uf),  # disp right
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Post-process data
    dxg = dx(domain_gauge)
    V = dolfiny.expression.assemble(1.0, dxg)
    n = ufl.as_vector([1, 0, 0])
    results["E"].append(dolfiny.expression.assemble(ufl.dot(E * n, n), dxg) / V)
    results["S"].append(dolfiny.expression.assemble(ufl.dot(S * n, n), dxg) / V)
    results["P"].append(dolfiny.expression.assemble(ufl.dot(P * n, n), dxg) / V)
    results["μ"].append(factor)

    # Basic consistency checks
    assert dolfiny.expression.assemble(dλ * df, dxg) / V < 1.0e-03, "|| dλ*df || != 0.0"
    assert dolfiny.expression.assemble(dλ * f, dxg) / V < 1.0e-05, "|| dλ*f || != 0.0"

    # Project plastic strain magnitude to DG0 for output and visualisation
    dolfiny.projection.project(ufl.sqrt(ufl.inner(P, P)), eps_p)

    # Write output
    ofile.write_function(u, step)

    # Interpolate and write output
    dolfiny.interpolation.interpolate(P, Po)
    dolfiny.interpolation.interpolate(B, Bo)
    dolfiny.interpolation.interpolate(S, So)
    dolfiny.interpolation.interpolate(h, ho)
    ofile.write_function(Po, step)
    ofile.write_function(Bo, step)
    ofile.write_function(So, step)
    ofile.write_function(ho, step)
    ofile.write_function(eps_p, step)

    # Store stress state
    dolfiny.interpolation.interpolate(S, S0)

    # Store primal states
    for source, target in zip([u, P, h, B], [u0, P0, h0, B0]):
        target.x.array[:] = source.x.array

ofile.close()

if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space.mesh))
    grid.point_data["u"] = u.x.array.reshape(-1, mesh.geometry.dim)
    grid_warped = grid.warp_by_vector("u", factor=1)

    grid_warped.cell_data["eps_p"] = eps_p.x.array

    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_mesh(
        grid_warped,
        scalars="eps_p",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Plastic strain"},
        n_colors=20,
        line_width=dolfiny.pyvista.pixels // 1000,
    )
    plotter.show_axes()
    plotter.camera.elevation = 15
    plotter.screenshot("J2_monolithic_deformed.png")
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ```{figure} J2_monolithic_deformed.png
# :alt: Deformed dog-bone specimen coloured by accumulated plastic strain magnitude.
# :align: center
# :label: fig-j2-plastic-strain
#
# Deformed dog-bone specimen coloured by accumulated plastic strain magnitude at the
# end of cyclic loading.
# ```
#
# ## Post-processing and visualization
#
# The stress-strain curve uses gauge-domain volume averages
# $$
#   \bar{S} = \frac{1}{V_g} \int_{\Omega_g} n^T S n \,\text{d}V, \qquad
#   \bar{E} = \frac{1}{V_g} \int_{\Omega_g} n^T E n \,\text{d}V
# $$
# where $\Omega_g$ is the gauge subdomain, $V_g = \int_{\Omega_g} 1 \,\text{d}V$ is its
# volume, and $n = e_x$ is the unit vector in the loading direction. The resulting hysteresis
# loops show elastic response, hardening, and Bauschinger-type reversal due to back-stress
# evolution.
# %% tags=["hide-input", "hide-output"]
plt.figure(dpi=300)
plt.title("Rate-independent plasticity: $J_2$, monolithic formulation, 3D", fontsize=12)
plt.xlabel(r"volume-averaged strain $\frac{1}{V}\int n^T E n \, \mathrm{d}V$ [-]", fontsize=12)
plt.ylabel(r"volume-averaged stress $\frac{1}{V}\int n^T S n \, \mathrm{d}V$ [GPa]", fontsize=12)
plt.grid(linewidth=0.25)

E = np.array(results["E"])
S = np.array(results["S"])

plt.plot(
    E,
    S,
    linestyle="-",
    linewidth=1.0,
    marker=".",
    markersize=4.0,
    label=r"$S-E$ curve",
)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("J2_monolithic_stress_strain.png", dpi=300)
plt.close()

# %% [markdown]
# ```{figure} J2_monolithic_stress_strain.png
# :alt: Stress-strain hysteresis with elastic loading, hardening, and Bauschinger effect.
# :align: center
# :label: fig-stress-strain
#
# Stress-strain hysteresis diagram showing elastic loading, hardening,
# Bauschinger effect, and cyclic stabilisation.
# ```
