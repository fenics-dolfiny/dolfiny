# %% [markdown]
# # Nonlinear Naghdi shell with arc-length continuation
#
# This demo traces the load-displacement response of a thin elastic cylindrical roof panel
# subjected to a downward point load at its crown — a classical benchmark for geometric
# nonlinearity in shells {cite:p}`Sze2004`. The panel undergoes snap-through (and snap-back),
# so arc-length continuation is used. The mechanical model uses five-parameter Naghdi
# kinematics with partial selective reduced integration. The spatial formulation
# employs the tangential differential calculus (TDC) framework
# {cite:p}`Hansbo2015,Schollhammer2019,Neunteufel2021` (§7.2.1 of the latter).
#
# Boundary conditions: the straight side edges are *hinged*
# (zero displacement, free rotation). The curved axial-end edges are *free*.
#
# In particular, this demo emphasizes:
# - Naghdi shell kinematics with three Euler-angle director rotations,
# - through-thickness decomposition of Green-Lagrange strains and PK2 stresses into membrane,
#   bending and shear contributions,
# - Crisfield {cite:p}`Crisfield1981` arc-length continuation past a snap-through limit point
#   including snap-back,
# - partial selective reduced integration to suppress membrane and shear locking,
# - a Lagrange-multiplier stabilisation of the unphysical drilling rotation,
# - validation against the digitised reference curve of {cite:p}`Sze2004`.
#
# ---

# %% tags=["hide-input"]

import warnings

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl

import matplotlib.pyplot as plt
import mesh_opencylinder_gmshapi as mg
import numpy as np
import pyvista

import dolfiny

warnings.filterwarnings("error")

name = "tdc_shell_naghdi_cylindrical_roof"
comm = MPI.COMM_WORLD

# %% [markdown]
# ## Geometry and reference configuration
#
# The benchmark geometry (Fig. 9a of {cite:p}`Sze2004`) is a thin cylindrical roof panel with
# radius $R = 2540$, axial half-length $L = 254$, half-opening angle $\theta = 0.1$, and
# thickness $h = 6.35$. Material parameters are $E = 3102.75$ and $\nu = 0.3$. The reference
# load is $P = 1$, with $\lambda$ the applied force.
#
# The full panel spans $\theta \in [-0.1, 0.1]$ circumferentially and $y \in [0, 2L]$ axially,
# with the crown ridge along $\theta = 0$. The point load acts at the midpoint of that ridge,
# `topmid`. Straight side edges (`sides` at $\theta = \pm 0.1$) are hinged. The arc-shaped
# axial ends (`front`, `rear`) are free.
#
# ```{figure} tdc_shell_naghdi_cylindrical_roof_geometry.png
# :alt: Geometry of the cylindrical roof panel (Fig. 9a of Sze 2004).
# :align: center
# :label: fig-roof-geometry
#
# Geometry of the cylindrical roof panel, reproduced from {cite:p}`Sze2004`.
# ```
#
# ```{note}
# The mesh geometric order must be $q \geq 2$. The reference normal $n_0$ is interpolated from
# the mesh geometry, so a linear mesh would flatten every curved element to a flat facet and
# lose the curvature information required by the shell kinematics.
# ```

# %% tags=["hide-input", "hide-output"]
R = 2540  # cylinder radius
L = 254  # axial half-length
h = 6.35  # shell thickness
θ = 0.1  # half-opening angle
N = 11  # nodes per edge
p = 2  # polynomial order, physics
q = 2  # polynomial order, geometry
do_quads = False

gmsh_model, tdim = mg.mesh_opencylinder_gmshapi(
    name, Ly=L, R=R, θ=θ, nL=N, nR=N, order=q, do_quads=do_quads
)

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, 0)
mesh = mesh_data.mesh
mts = mesh_data.cell_tags
gdim = mesh.geometry.dim

# %% [markdown]
# ## Naghdi shell kinematics via TDC
#
# A material point in the shell is parametrised by a midsurface coordinate $x_0 \in \Omega$ and a
# through-thickness coordinate $\xi \in [-h/2, h/2]$. Its placement in the reference and deformed
# configurations is
#
# $$ b_0(x_0, \xi) = x_0 + \xi\, d_0(x_0), \qquad b(x_0, \xi) = x_0 + u(x_0) + \xi\, d(x_0), $$
#
# where $u$ is the midsurface displacement, $d_0 = n_0$ is the reference unit normal (the
# *director*) and $d$ is the deformed director. In the Naghdi model $d$ need not stay normal to the
# deformed midsurface: it is parametrised by three Euler-angle rotations stored in a vector field
# $r$,
#
# $$ d = R_x(r_0)\, R_y(r_1)\, R_z(r_2)\, d_0. $$
#
# The third rotation component rotates the director about itself and carries no physical meaning.
# We suppress it by introducing a scalar Lagrange multiplier $c$ that enforces $r \cdot n_0 = 0$
# weakly. The primary unknowns are therefore $u, r \in [H^1(\Omega)]^3$ and $c \in L^2(\Omega)$,
# all discretised here with continuous Lagrange elements of degree $p = 2$.

# %% tags=["hide-input"]
metadata = {"quadrature_degree": p * p}
metadata_prsi = {"quadrature_degree": p * (p - 1)}
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags, metadata=metadata)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags, metadata=metadata)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=mesh_data.facet_tags, metadata=metadata)
dP = ufl.Measure("dP", domain=mesh, subdomain_data=mesh_data.ridge_tags, metadata=metadata)

Ue = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=p, shape=(gdim,))
Re = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=p, shape=(gdim,))
Ce = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=p, shape=())

Uf = dolfinx.fem.functionspace(mesh, Ue)
Rf = dolfinx.fem.functionspace(mesh, Re)
Cf = dolfinx.fem.functionspace(mesh, Ce)

u = dolfinx.fem.Function(Uf, name="u")
r = dolfinx.fem.Function(Rf, name="r")
c = dolfinx.fem.Function(Cf, name="c")
u_ = dolfinx.fem.Function(Uf, name="u_")
r_ = dolfinx.fem.Function(Rf, name="r_")

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf, Rf, Cf))
δu, δr, δc = δm

m = [u, r, c]

# Saint Venant-Kirchhoff parameters (plane stress)
t0 = dolfinx.fem.Constant(mesh, h)
E_val, nu = 3102.75, 0.3
matλ = dolfinx.fem.Constant(mesh, E_val * nu / (1 - nu**2))
matμ = dolfinx.fem.Constant(mesh, E_val / (2 * (1 + nu)))

# Hinged BCs on the straight side edges (θ=±θ): zero displacement, rotations free.
# The curved axial-end edges (front/rear) are unconstrained.
sides_dofs_Uf = dolfiny.mesh.locate_dofs_topological(
    Uf, mesh_data.facet_tags, mesh_data.physical_groups["sides"].tag
)

bcs = [dolfinx.fem.dirichletbc(u_, sides_dofs_Uf)]

# %% [markdown]
# ## Strain and stress decomposition
#
# The configuration gradient $J = \nabla b$ and its reference counterpart $J_0 = \nabla b_0$
# combine into the Green-Lagrange strain
#
# $$ E = \tfrac{1}{2}\,(J^\top J - J_0^\top J_0). $$
#
# Because $J$ depends affinely on the through-thickness coordinate $\xi$, the strain admits a clean
# separation into membrane, bending and shear parts via the tangent plane projector
# $P = I - n_0 \otimes n_0$:
#
# $$
#  E_m = P\, E\big|_{\xi=0}\, P, \quad
#  E_b = P\, \partial_\xi E\big|_{\xi=0}\, P, \quad
#  E_s = E\big|_{\xi=0} - E_m.
# $$
#
# The same projection applied to the second Piola-Kirchhoff stress $S = 2\mu E + \lambda\,
# \text{tr}(E)\, I$ yields the corresponding stress measures $S_m$, $S_b$, $S_s$. Integration
# through the thickness gives the customary stress resultants
#
# $$ N = h\, S_m, \qquad Q = h\, S_s, \qquad M = \tfrac{h^3}{12}\, S_b. $$

# %% tags=["hide-input"]
I = ufl.Identity(mesh.geometry.dim)  # noqa: E741

We = basix.ufl.element("DG", mesh.basix_cell(), degree=q, shape=(gdim,))
W = dolfinx.fem.functionspace(mesh, We)
n0 = dolfinx.fem.Function(W, name="n0")
dolfiny.interpolation.interpolate(ufl.CellNormal(mesh), n0)

Sb_norm = dolfinx.fem.Function(Cf, name="Sb_norm")

P = I - ufl.outer(n0, n0)
x0 = ufl.SpatialCoordinate(mesh)

Ξe = basix.ufl.element("DG", mesh.basix_cell(), degree=q, shape=())
Ξ = dolfinx.fem.functionspace(mesh, Ξe)
ξ = dolfinx.fem.Function(Ξ, name="ξ")

# Reference and deformed placements
d0 = n0
b0 = x0 + ξ * d0

R0 = ufl.as_matrix(
    [[1, 0, 0], [0, ufl.cos(r[0]), -ufl.sin(r[0])], [0, ufl.sin(r[0]), ufl.cos(r[0])]]
)
R1 = ufl.as_matrix(
    [[ufl.cos(r[1]), 0, ufl.sin(r[1])], [0, 1, 0], [-ufl.sin(r[1]), 0, ufl.cos(r[1])]]
)
R2 = ufl.as_matrix(
    [[ufl.cos(r[2]), -ufl.sin(r[2]), 0], [ufl.sin(r[2]), ufl.cos(r[2]), 0], [0, 0, 1]]
)

d = (R0 * R1 * R2) * d0
b = x0 + u + ξ * d

# Configuration gradients (the explicit subtraction of d0⊗d0 makes J non-degenerate
# normal-to-surface so that derivatives in ξ are well defined).
J0 = ufl.grad(b0) - ufl.outer(d0, d0)
J0 = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(J0)
J0 = ufl.algorithms.apply_derivatives.apply_derivatives(J0)
J0 = ufl.replace(J0, {ufl.grad(ξ): d0})

J = ufl.grad(b) - ufl.outer(d0, d0)
J = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(J)
J = ufl.algorithms.apply_derivatives.apply_derivatives(J)
J = ufl.replace(J, {ufl.grad(ξ): d0})

# Green-Lagrange strain and its membrane/bending/shear projections
E = (J.T * J - J0.T * J0) / 2

Em = P * ufl.replace(E, {ξ: 0.0}) * P

Eb = ufl.diff(E, ξ)
Eb = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(Eb)
Eb = ufl.algorithms.apply_derivatives.apply_derivatives(Eb)
Eb = P * ufl.replace(Eb, {ξ: 0.0}) * P

Es = ufl.replace(E, {ξ: 0.0}) - P * ufl.replace(E, {ξ: 0.0}) * P

δEm = ufl.derivative(Em, m, δm)
δEs = ufl.derivative(Es, m, δm)
δEb = ufl.derivative(Eb, m, δm)

# Saint Venant-Kirchhoff PK2 stress with the same membrane/bending/shear split
S = 2 * matμ * E + matλ * ufl.tr(E) * I

Sm = P * ufl.replace(S, {ξ: 0.0}) * P

Sb = ufl.diff(S, ξ)
Sb = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(Sb)
Sb = ufl.algorithms.apply_derivatives.apply_derivatives(Sb)
Sb = P * ufl.replace(Sb, {ξ: 0.0}) * P

Ss = ufl.replace(S, {ξ: 0.0}) - P * ufl.replace(S, {ξ: 0.0}) * P

# Through-thickness integration → stress resultants
Sm *= t0
Ss *= t0
Sb *= t0**3 / 12

# %% [markdown]
# ## Variational form and locking treatment
#
# Equilibrium is expressed as the principle of virtual work
#
# $$
#  \int_\Omega \big( \delta E_m : N + \delta E_s : Q + \delta E_b : M \big) \,\text{d}x
#  \;=\; \lambda\, p_z\, \delta u_z \big|_{\text{topmid}},
# $$
#
# where the right-hand side is the virtual work of a unit downward concentrated load
# $-e_z$ applied at the co-dimension-2 point `topmid` (crown centre), scaled by $\lambda$.
# Since the reference load magnitude is 1, $\lambda$ is the applied force.
# Low-order shell elements are prone to membrane and transverse-shear locking. We address this with
# *partial selective reduced integration* {cite:p}`Arnold1997`, blending fully and
# reduced-integrated contributions via the element-wise weight $\alpha = h^2 / |J|$, where $|J|$
# is the cell Jacobian determinant. The drilling rotation $r \cdot n_0$ is also constrained weakly
# by the Lagrange multiplier $c$, giving the augmented form
#
# $$ \delta c \,(r \cdot n_0)\, \mu + c \,(\delta r \cdot n_0)\, \mu. $$

# %% tags=["hide-input"]
P_max = 1.0  # benchmark reference load
λ = dolfinx.fem.Constant(mesh, 1.0)

# Element-wise weight for partial selective reduced integration (Arnold/Brezzi 1997)
Ae = basix.ufl.element("DG", mesh.basix_cell(), degree=0)
A = dolfinx.fem.functionspace(mesh, Ae)
α = dolfinx.fem.Function(A)
dolfiny.interpolation.interpolate(t0**2 / ufl.JacobianDeterminant(mesh), α)

f = (
    -ufl.inner(δEm, Sm) * α * dx
    - ufl.inner(δEm, Sm) * (1 - α) * dx(metadata=metadata_prsi)
    - ufl.inner(δEs, Ss) * α * dx
    - ufl.inner(δEs, Ss) * (1 - α) * dx(metadata=metadata_prsi)
    - ufl.inner(δEb, Sb) * dx
    + δc * ufl.inner(r, n0) * matμ * dx
    + c * ufl.inner(δr, n0) * matμ * dx
    + δc * dolfinx.fem.Constant(mesh, 0.0) * c * dx
    + λ
    * ufl.inner(δu, ufl.as_vector((0.0, 0.0, -P_max)))
    * dP(mesh_data.physical_groups["topmid"].tag)
)

F = ufl.extract_blocks(f)

# %% [markdown]
# ## Reference configuration
#
# Before launching the continuation we visualise the un-deformed midsurface as a sanity check on
# the mesh and the inferred normal field $n_0$. Bending stress resultants are zero in this state.


# %% tags=["hide-input"]
def plot_roof_pyvista(u, s, png, comm=MPI.COMM_WORLD):
    if comm.size > 1:
        return

    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))

    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.theme.font.fmt = "%1.3f"
    plotter.add_axes()
    plotter.enable_parallel_projection()

    grid.point_data["u"] = u.x.array.reshape(-1, 3)
    grid.point_data["stress"] = s.x.array

    grid_warped = grid.warp_by_vector("u", factor=1.0)

    levels = 5 if not grid.get_cell(0).is_linear else 0

    surf = plotter.add_mesh(
        grid_warped.extract_surface(nonlinear_subdivision=levels, algorithm="dataset_surface"),
        scalars="stress",
        scalar_bar_args={"title": "Bending stress resultant [-]"},
        n_colors=10,
        color="white",
    )
    surf.mapper.scalar_range = [0.0, 300]

    plotter.add_mesh(
        grid_warped.separate_cells()
        .extract_surface(nonlinear_subdivision=levels, algorithm="dataset_surface")
        .extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=dolfiny.pyvista.pixels // 1000,
        render_lines_as_tubes=True,
    )

    plotter.view_xz()
    plotter.camera.azimuth += 45
    plotter.camera.elevation += 30
    # Frame on the reference geometry so the camera does not zoom with the deformation.
    plotter.camera.focal_point = grid.center
    plotter.camera.parallel_scale = 0.8 * max(
        grid.bounds[1] - grid.bounds[0],
        grid.bounds[3] - grid.bounds[2],
        grid.bounds[5] - grid.bounds[4],
    )
    plotter.show_axes()
    plotter.screenshot(png)
    plotter.close()
    plotter.deep_clean()


plot_roof_pyvista(u, Sb_norm, f"{name}_initial.png")

# %% [markdown]
# ```{figure} tdc_shell_naghdi_cylindrical_roof_initial.png
# :alt: Reference configuration of the cylindrical roof panel.
# :align: center
# :label: fig-roof-reference
#
# Reference configuration of the cylindrical roof panel.
# ```

# %% [markdown]
# ## Arc-length continuation
#
# Around the snap-through limit point the load-displacement tangent becomes vertical: at fixed
# $\lambda$ no nearby equilibrium exists, and Newton iteration on the displacement alone diverges.
# The Crisfield method {cite:p}`Crisfield1981` addresses this by promoting $\lambda$ to an unknown
# and appending a spherical arc-length constraint of prescribed step $\Delta s$,
#
# $$ \langle \Delta m,\, \Delta m \rangle + \psi^2\, \Delta\lambda^2 = \Delta s^2, $$
#
# where $\langle \cdot,\, \cdot \rangle$ is an inner product on the discrete state $m = (u, r, c)$
# and $\psi$ is a scaling factor. The augmented system is solved at every step by `dolfiny`'s
# `Crisfield` class, which wraps the underlying SNES nonlinear solver. We use a sparse direct LU
# (MUMPS) for the linearised systems. Step size control is adaptive: if a step fails — either
# because SNES does not converge or the arc-length quadratic has no real root — the step is
# retried with $\Delta s$ halved and a zero displacement predictor; on success the step size
# doubles back towards the prescribed maximum $\Delta s_{\max}$.

# %% tags=["hide-input", "hide-output"]
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    if q <= 2:
        ofile.write_mesh_meshtags(mesh)

opts = PETSc.Options("roof")  # type: ignore[attr-defined]
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-14
opts["snes_rtol"] = 1.0e-08
# opts["snes_stol"] = 1.0e-08
opts["snes_max_it"] = 40
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

u_step: list[np.ndarray] = []
λ_step: list[float] = []


def monitor(context=None):
    if comm.size > 1:
        return

    idx = dolfiny.mesh.locate_dofs_topological(
        Uf, mesh_data.ridge_tags, mesh_data.physical_groups["topmid"].tag
    )
    idx = dolfiny.function.unroll_dofs(idx, Uf.dofmap.bs)

    u_step.append(u.x.array[idx])
    λ_step.append(context.λ.value.item())


def block_inner(a1, a2):
    b1, b2 = [], []
    for mi in m:
        b1.append(dolfinx.fem.Function(mi.function_space, name=mi.name))
        b2.append(dolfinx.fem.Function(mi.function_space, name=mi.name))

    dolfinx.fem.petsc.assign(a1, b1)
    dolfinx.fem.petsc.assign(a2, b2)
    inner = 0.0
    for b1i, b2i in zip(b1, b2):
        inner += dolfiny.expression.assemble(ufl.inner(b1i, b2i), ufl.dx(mesh))
    return inner


def plot_load_displacement(u_step, λ_step, png):
    if comm.size > 1:
        return

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=400)
    ax1.set_title(
        f"Cylindrical roof $h={h}$, elems=[{(N - 1) * 2}, {(N - 1) * 2}]",
        fontsize=12,
    )
    ax1.set_xlabel("crown displacement $-u_z$ [-]", fontsize=12)
    ax1.set_ylabel("load $P$ [-]", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    sol_tdc = np.column_stack([λ_step, u_step])
    ax1.plot(
        -sol_tdc[:, 3],
        sol_tdc[:, 0],
        ls="-",
        lw=1.0,
        color="k",
        label="TDC (Naghdi, this work)",
    )

    # Overlay Sze (2004) Table 9d reference data (S4R, 24×24 mesh, P_max=3000).
    sze = np.loadtxt(f"{name}_sze2004.csv", delimiter=",", skiprows=1)
    ax1.plot(
        sze[:, 1],
        sze[:, 0],
        marker="o",
        ls="",
        color="tab:blue",
        label="Sze (2004), S4R 24×24",
    )

    ax1.legend()
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


problem = dolfiny.snesproblem.SNESProblem(F, m, bcs, prefix="roof")

continuation = dolfiny.continuation.Crisfield(problem, λ, inner=block_inner)
continuation.initialise(ds=0.1, λ=0.0, psi=1.0)
monitor(continuation)

for k in range(64):
    dolfiny.utils.pprint(f"\n*** Continuation step {k:d}")
    continuation.solve_step(ds=200)

    monitor(continuation)

    with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "a") as ofile:
        if q <= 2:
            ofile.write_function(u, k)
            dolfiny.projection.project(ufl.inner(Sb, Sb) ** 0.5, Sb_norm)
            ofile.write_function(Sb_norm, k)


# %% [markdown]
# ## Deformed configuration and load-response curve
#
# After the run the roof has snapped through and, for this geometry, snapped back. The bending
# stress resultant peaks along the curvature ridge at the crown. The load-displacement curve
# shows the characteristic limit-point softening, the load reversal during snap-back, and the
# subsequent post-buckling rise.

# %% tags=["hide-input"]
plot_roof_pyvista(u, Sb_norm, f"{name}_deformed.png")

# %% [markdown]
# ```{figure} tdc_shell_naghdi_cylindrical_roof_deformed.png
# :alt: Final deformed cylindrical roof coloured by bending stress resultant magnitude.
# :align: center
# :label: fig-roof-deformed
#
# Final deformed configuration coloured by bending stress resultant magnitude.
# ```

# %% tags=["hide-input"]
if comm.size == 1:
    sol_tdc = np.column_stack([λ_step, u_step])
    np.savetxt(f"{name}_tdc_solution.csv", sol_tdc, delimiter=",", header="λ,u", comments="")
    plot_load_displacement(u_step, λ_step, f"{name}.png")

# %% [markdown]
# ```{figure} tdc_shell_naghdi_cylindrical_roof.png
# :alt: Load factor versus crown displacement.
# :align: center
# :label: fig-roof-load-disp
#
# Load factor $\lambda$ versus downward crown displacement $-u_z$ traced by Crisfield arc-length
# continuation, showing the snap-through limit point, load reversal during snap-back, and
# post-buckling rise.
# ```
