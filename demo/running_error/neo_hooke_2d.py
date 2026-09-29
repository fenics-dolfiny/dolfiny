# %% [markdown]
# ---
# authors:
#   - mh
#   - ptk
#   - mc
#   - az
# ---

# %% [markdown]
# # Running error analysis for the Neo-Hooke strain energy density
#
# This demo solves a St. Venant-Kirchhoff problem for a 2D cantilever beam and computes the
# Neo-Hooke strain energy density. It demonstrates how to use running error analysis to
# obtain upper bounds or estimates on the running error accumulated during assembly.
#
# In particular, this demo emphasizes:
# - custom assemblers that track rounding errors,
# - worst and exact error modes,
# - catastrophic cancellation in the standard Neo-Hooke energy density at small strains,
# - reformulation of the Neo-Hooke energy in terms of 3rd order expansion.
#
# ## Background: Running error analysis
#
# Running error analysis is an a posteriori method to estimate the numerical error in
# floating-point computations, see Chapter 3.3 of {cite:t}`Higham2002`. Every floating-point
# operation $\hat f = \text{fl}(\hat x \text{ op } \hat y)$ commits a fresh rounding error
# bounded by $\epsilon_\text{mach} |\hat f|$, with $\epsilon_\text{mach}$ the machine epsilon,
# and on top of that propagates the errors $e_x = \hat x - x$, $e_y = \hat y - y$ that its
# inputs already carry, with $|e_x| \leq \bar e_x$, $|e_y| \leq \bar e_y$. To first order the
# two contributions combine into
#
# $$
# |\hat f - f| \leq \epsilon_\text{mach} |\hat f|
#                + \left|\frac{\partial f}{\partial x}\right| \bar e_x
#                + \left|\frac{\partial f}{\partial y}\right| \bar e_y + \text{h.o.t.},
# $$
#
# where $f = x \text{ op } y$ is the exact result for exact inputs, and the derivatives are
# evaluated at $(\hat x, \hat y)$. For addition both derivatives are $1$, giving
# $\epsilon_\text{mach} |\hat f| + \bar e_x + \bar e_y$. For multiplication it gives
# $\epsilon_\text{mach} |\hat f| + |\hat y| \bar e_x + |\hat x| \bar e_y$.
#
# The bound can therefore be evaluated *on the fly*, by pairing every value with its error
# and updating both at each operation. This is what the `running_error_t<T, Mode>` type of
# the header-only C++23 library [`rea`](https://github.com/fenics-dolfiny/rea) implements
# (`running_error.h`, which `dolfiny` fetches and installs alongside its extension module).
# It costs one extra scalar per value and requires no change to the algorithm.
#
# ## Running error modes: "worst" vs "exact"
#
# The `ErrorMode` template parameter selects between two tracking strategies. In both, the
# value and its error share the working precision `T`, so a scalar is a homogeneous pair
# `{T val; T err;}` of size `2 * sizeof(T)`.
#
# **`ErrorMode::WORST` (default), "worst" mode.** `err` is a nonnegative bound, updated by the
# formula above. Addition, for instance, reads
#
# ```cpp
# re_t operator+(const re_t& other) const {
#   const T new_val = val + other.val;
#   return re_t{new_val, err + other.err + re_eps<T> * re_abs(new_val)};
# }
# ```
#
# Absolute values are taken throughout, so the bound can never decrease: it is an approximate,
# first-order worst-case bound, and pessimistic. The local term uses the machine epsilon of `T`
# rather than the unit roundoff $\epsilon_\text{mach}/2$ of round-to-nearest, which keeps it
# conservative.
#
# **`ErrorMode::EXACT`, "exact" mode.** `err` is instead a *signed* estimate: the derivatives
# are taken with sign, and the local rounding is measured rather than bounded — exactly, by
# error-free transformations, for `+ - *`, by an FMA-based estimate for `/`, and by
# re-evaluating in a wider type (`float32` for
# `float16`, `float64` for `float32`, `long double` for `float64`) for `sqrt`, `log`, `pow`
# and the trigonometric functions. Errors of opposite sign then cancel, which exposes true
# cancellation instead of hiding it under a worst-case envelope.
#
# ## Python interface
#
# `dolfiny.fem.form` JIT-compiles a UFL form with `running_error_t` as its scalar type,
# using the templated FFCx kernels of `ffcx-backends` and `cppjit`.
# `dolfiny.fem.assemble_vector` then runs the DOLFINx assembly with that type, so every
# operation in the element kernel tracks its own error.
#
# nanobind and DLPack cannot describe the packed `(value, error)` struct, so the buffer is
# exposed without a copy as an opaque carrier of matching width (`complex64` for the
# `float32` used here) and reinterpreted by `dolfiny.fem.split`. Coefficients and constants
# enter with zero error, i.e. treated as exact, unless an error is passed explicitly as
# `assemble_vector(L, errors={u: u_err})`. Assembly returns the
# value and the accumulated error as two real arrays of dtype `T` — an approximate upper bound
# in "worst" mode, a signed estimate in "exact" mode.
#
# We start by generating a rectangular mesh for a cantilever beam.

# %% tags=["hide-input"]
from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl

import gmsh
import numpy as np
import pyvista as pv

import dolfiny

comm = MPI.COMM_WORLD

w, h = 0.5, 0.1  # 50cm long, 10cm high beam
res = 15

if comm.rank == 0:
    gmsh.initialize()
    gmsh.model.add("cantilever")
    gmsh.model.occ.addRectangle(0, 0, 0, w, h)
    gmsh.model.occ.synchronize()

    # Add physical groups for domain and boundaries
    gmsh.model.addPhysicalGroup(2, [1], name="domain")

    # Find left and right boundaries
    lines = gmsh.model.getEntities(1)
    left_line = None
    right_line = None
    for dim, tag in lines:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[0], 0.0):
            left_line = tag
        elif np.isclose(com[0], w):
            right_line = tag

    gmsh.model.addPhysicalGroup(1, [left_line], name="left")
    gmsh.model.addPhysicalGroup(1, [right_line], name="right")

    gmsh.option.setNumber("Mesh.MeshSizeMax", h / res)
    gmsh.model.mesh.generate(2)

mesh_data = dolfinx.io.gmsh.model_to_mesh(
    gmsh.model if comm.rank == 0 else None, comm, rank=0, gdim=2
)
mesh = mesh_data.mesh
fdim = mesh.topology.dim - 1
facet_tags = mesh_data.facet_tags
assert facet_tags is not None
tag_left = mesh_data.physical_groups["left"].tag
tag_right = mesh_data.physical_groups["right"].tag

if comm.rank == 0:
    gmsh.finalize()
    dolfiny.utils.pprint("Number of cells:", mesh.topology.index_map(mesh.topology.dim).size_local)

# %% [markdown]
# ## Pre-processing: St. Venant-Kirchhoff solution
#
# The displacement field $\boldsymbol u$ is computed with a St. Venant-Kirchhoff material law
# with steel-like properties, $\mu \approx 77$ GPa and first Lamé parameter
# $\lambda \approx 115$ GPa (Young's modulus $200$ GPa, Poisson's ratio $\nu = 0.3$). The beam
# is clamped on the left boundary and subjected to a downward traction $t_y = 1$ MPa on the
# right boundary.

# %% tags=["hide-input", "hide-output"]
Ve = basix.ufl.element("P", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dolfinx.fem.functionspace(mesh, Ve)

u0_svk = dolfinx.fem.Function(V, name="displacement_svk")

# Steel properties
E = 200e9  # 200 GPa
nu = 0.3  # Poisson ratio
mu = E / (2.0 * (1.0 + nu))  # Shear modulus

# Plane strain
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lame first param


def GL(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T + ufl.grad(u).T * ufl.grad(u))


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Traction load on the right boundary
f = np.array([0.0, -1e6], dtype=np.float64)  # 1 MPa
T = dolfinx.fem.Constant(mesh, f)

energy_load = ufl.inner(T, u0_svk) * ds(tag_right)
energy_svk = (
    lmbda / 2 * ufl.tr(GL(u0_svk)) ** 2 * ufl.dx
    + mu * ufl.inner(GL(u0_svk), GL(u0_svk)) * ufl.dx
    - energy_load
)

F_svk = ufl.derivative(energy_svk, u0_svk)

left_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facet_tags.find(tag_left))
bc = dolfinx.fem.dirichletbc(np.zeros(mesh.geometry.dim, dtype=np.float64), left_dofs, V)

opts = PETSc.Options("svk")  # type: ignore[attr-defined]
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

problem = dolfiny.snesproblem.SNESProblem([F_svk], [u0_svk], bcs=[bc], prefix="svk")

_ = problem.solve()


# %% [markdown]
# ## Neo-Hooke strain energy density with running error bounds
#
# We assemble the strain energy density in two algebraically equivalent forms and compare
# their running error behaviour.
#
# **Naive Neo-Hooke (unstable).** We use the variant $W_a$ of {cite:t}`Pence2014`, Eq. 2.11,
# which under a 2D plane-strain kinematic simplification reads
#
# $$
# W = \frac{\mu}{2}(I_1 - 2 - 2 \log J) + \frac{\lambda}{2}(J - 1)^2,
# $$
#
# with $I_1 = \operatorname{tr}(\boldsymbol C)$, $J = \det(\boldsymbol F)$,
# $\boldsymbol C = \boldsymbol F^\mathsf{T} \boldsymbol F$ and
# $\boldsymbol F = \boldsymbol I + \nabla \boldsymbol u$.
# At small strains $I_1 \approx 2$ and $J \approx 1$, so both $I_1 - 2 - 2 \log J$ and
# $(J - 1)^2$ suffer catastrophic cancellation, see {cite:t}`Shakeri2024`.
#
# **Third-order expansion in the Green-Lagrange strain (stable).** Following
# {cite:t}`Habera2026DimensionalAnalysis`, the cancellation is avoided by rewriting the
# energy in the Green-Lagrange strain
# $\boldsymbol E = \tfrac{1}{2}(\boldsymbol C - \boldsymbol I)$, which eliminates the identity
# term of $\boldsymbol F$. Splitting $\boldsymbol E$ into a linear and a nonlinear part,
#
# $$
# \boldsymbol E = \boldsymbol E_1 + \boldsymbol E_2,
# \qquad
# \boldsymbol E_1 = \tfrac{1}{2}\bigl(\nabla \boldsymbol u + \nabla \boldsymbol u^\mathsf{T}\bigr),
# \qquad
# \boldsymbol E_2 = \tfrac{1}{2}\,\nabla \boldsymbol u^\mathsf{T} \nabla \boldsymbol u,
# $$
#
# the two contributions scale as $\boldsymbol E_1 = \mathcal{O}(\|\nabla \boldsymbol u\|)$ and
# $\boldsymbol E_2 = \mathcal{O}(\|\nabla \boldsymbol u\|^2)$. Expanding to third order in
# $\|\nabla \boldsymbol u\|$ gives
# $W_\text{stable} = W^{(2)} + W^{(3)} + \mathcal{O}(\|\nabla \boldsymbol u\|^4)$ with
#
# $$
# \begin{aligned}
# W^{(2)} &= \underbrace{\mu\,\operatorname{tr}(\boldsymbol E_1^2)}_{W^{(2)}_\mu}
#   + \underbrace{\tfrac{\lambda}{2}\,\operatorname{tr}(\boldsymbol E_1)^2}_{W^{(2)}_\lambda}, \\
# W^{(3)} &= \underbrace{\mu\bigl(2\,\boldsymbol E_1 : \boldsymbol E_2
#   - \tfrac{4}{3}\operatorname{tr}(\boldsymbol E_1^3)\bigr)}_{W^{(3)}_\mu}
#   + \underbrace{\lambda\bigl(\operatorname{tr}(\boldsymbol E_1)\operatorname{tr}(\boldsymbol E_2)
#   + \tfrac{1}{2}\operatorname{tr}(\boldsymbol E_1)^3
#   - \operatorname{tr}(\boldsymbol E_1)\operatorname{tr}
#     (\boldsymbol E_1^2)\bigr)}_{W^{(3)}_\lambda}.
# \end{aligned}
# $$
#
# These are the four terms ``shear_2``, ``bulk_2``, ``shear_3``, ``bulk_3`` below. Every
# intermediate now carries the same $\|\nabla \boldsymbol u\|$-scaling as the result itself, so
# no cancellation occurs at small strains.
#
# Both energy densities enter a linear form, which is assembled cell-wise,
#
# $$
# L(v; \boldsymbol u) = \int_\Omega W(\boldsymbol u) |\mathcal K|^{-1} v \, \mathrm dx,
# \qquad
# L_\text{stable}(v; \boldsymbol u)
#   = \int_\Omega W_\text{stable}(\boldsymbol u) |\mathcal K|^{-1} v \, \mathrm dx,
# $$
#
# for test functions $v \in W_h$ of cell-wise constants, so that the assembled vector
# $b_i = L(\varphi_i; \boldsymbol u)$ is a cell-averaged strain energy density. Both forms are
# assembled in single precision (`dtype=np.float32`) and compared side-by-side in the
# visualisation below.

# %% [markdown]
# ### Assembling strain energy with running error bounds
#
# The helper `extract_energy_fields` compiles a form with `dolfiny.fem.form`, assembles it
# with `dolfiny.fem.assemble_vector`, and returns the energy density together with its
# absolute and relative error as cell-wise (DG-0) fields, in either error mode. It also
# times the assembly against the plain DOLFINx one, to show the cost of the error tracking.
#
# The input errors of the displacement degrees-of-freedom are initialized randomly at the
# scale of machine precision, using a fixed seed. The relative rounding error estimate is
#
# $$
# \eta_i = \frac{|e_i|}{|b_i^\text{ref}|},
# $$
#
# where $e_i$ is the error estimate of the assembled $b_i$, and $b_i^\text{ref}$ is the double
# precision evaluation of the stable expression. For the "worst" mode the estimate is a
# nonnegative approximate bound, $|\text{fl}(b_i) - b_i| \leq e_i + \text{h.o.t.}$, while for
# the "exact" mode it has a sign, $\text{fl}(b_i) - b_i \approx e_i$.

# %% tags=["hide-input"]
# DG-0 space for cell-wise strain energy density
S = dolfinx.fem.functionspace(mesh, ("DG", 0))
δs = ufl.TestFunction(S)


def extract_energy_fields(form, name, suffix="", mode="worst"):
    """Compile form, assemble vector, and extract values and error bounds."""
    import time

    # Input errors of the displacement dofs: random relative error at the scale of machine
    # precision, with a fixed seed. The "worst" mode takes its magnitude.
    rng = np.random.default_rng(0)
    u_err = 2 * np.finfo(np.float32).eps * rng.uniform(-1, 1, u0_svk.x.array.size)
    u_err *= u0_svk.x.array
    if mode == "worst":
        u_err = np.abs(u_err)

    compiled_form = dolfiny.fem.form(form, mode=mode, dtype=np.float32)
    t0 = time.time()
    val, err = dolfiny.fem.assemble_vector(compiled_form, errors={u0_svk: u_err})
    time_dolfiny = time.time() - t0

    compiled_form_dolfinx = dolfinx.fem.form(form)
    t0 = time.time()
    _ = dolfinx.fem.petsc.assemble_vector(compiled_form_dolfinx)
    time_dolfinx = time.time() - t0

    slowdown = time_dolfiny / time_dolfinx if time_dolfinx > 0 else float("inf")
    if comm.rank == 0:
        dolfiny.utils.pprint(
            f"{name:41s} ({mode:5s} mode): {time_dolfiny:8.2g}s (slowdown: {slowdown:6.2f}x)"
        )

    energy = dolfinx.fem.Function(S, name=name)
    energy.x.array[:] = val

    abs_err = dolfinx.fem.Function(S, name=f"absolute_error{suffix}_{mode}")
    abs_err.x.array[:] = np.abs(err)

    # Relative error eta_i = |e_i| / |b_i^ref|, see the reference below
    rel_err = dolfinx.fem.Function(S, name=f"relative_error{suffix}_{mode}")
    nonzero = np.abs(reference) > 0
    rel_err.x.array[nonzero] = abs_err.x.array[nonzero] / np.abs(reference[nonzero])

    return energy, abs_err, rel_err


# --- 1) Neo-Hooke strain energy density ---
dim = mesh.geometry.dim


def neo_hooke(u):
    """Naive Neo-Hooke strain energy density W, unstable at small strains."""
    F = ufl.Identity(dim) + ufl.grad(u)
    C = F.T * F
    I1, J = ufl.tr(C), ufl.det(F)

    return mu / 2 * (I1 - 2 - 2 * ufl.ln(J)) + lmbda / 2 * (J - 1) ** 2


def neo_hooke_expansion(u):
    """Neo-Hooke energy W_stable, expanded to 3rd order in the Green-Lagrange strain."""
    H = ufl.grad(u)
    E1 = ufl.sym(H)
    E2 = 0.5 * H.T * H

    # Second-order terms
    shear_2 = mu * ufl.tr(E1 * E1)  # W^(2)_mu
    bulk_2 = lmbda / 2 * ufl.tr(E1) ** 2  # W^(2)_lambda

    # Third-order terms
    shear_3 = mu * (2 * ufl.inner(E1, E2) - 4 / 3 * ufl.tr(E1 * E1 * E1))  # W^(3)_mu
    bulk_3 = lmbda * (
        ufl.tr(E1) * ufl.tr(E2) + 1 / 2 * ufl.tr(E1) ** 3 - ufl.tr(E1) * ufl.tr(E1 * E1)
    )  # W^(3)_lambda

    return shear_2 + bulk_2 + shear_3 + bulk_3


def density_form(u, energy_density):
    """Cell-wise averaged strain energy density as a DG-0 form."""
    return (energy_density(u) / ufl.CellVolume(mesh)) * δs * ufl.dx


# Reference b^ref for both forms: the stable expression assembled in double precision. The
# unstable expression loses its correct digits at small strains even in double precision.
reference = dolfinx.fem.assemble_vector(
    dolfinx.fem.form(density_form(u0_svk, neo_hooke_expansion))
).array


energy_form = density_form(u0_svk, neo_hooke)

energy_fn, abs_error_fn, rel_error_fn = extract_energy_fields(energy_form, "Strain Energy Density")
energy_fn_exact, abs_error_fn_exact, rel_error_fn_exact = extract_energy_fields(
    energy_form, "Strain Energy Density", mode="exact"
)

energy_approx_form = density_form(u0_svk, neo_hooke_expansion)

energy_approx_fn, abs_error_approx_fn, rel_error_approx_fn = extract_energy_fields(
    energy_approx_form, "Neo-Hooke Expansion Strain Energy Density", suffix="_approx"
)
energy_approx_fn_exact, abs_error_approx_fn_exact, rel_error_approx_fn_exact = (
    extract_energy_fields(
        energy_approx_form,
        "Neo-Hooke Expansion Strain Energy Density",
        suffix="_approx",
        mode="exact",
    )
)

# Write results
with dolfinx.io.XDMFFile(comm, "error.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u0_svk)
    xdmf.write_function(energy_fn)
    xdmf.write_function(rel_error_fn)
    xdmf.write_function(abs_error_fn)
    xdmf.write_function(rel_error_fn_exact)


# %% [markdown]
# ## Visualisation
#
# All field plots below show the St. Venant-Kirchhoff solution at the nominal load, whose
# deformation scale $\Pi = \|\boldsymbol u\|_\infty / w \sim C \|\nabla \boldsymbol u\|$,
# reported first, is what drives the cancellation in the unstable energy.


# %% tags=["hide-input"]
u_max = comm.allreduce(
    np.max(np.linalg.norm(u0_svk.x.array.reshape(-1, dim), axis=1), initial=0.0), op=MPI.MAX
)
if comm.rank == 0:
    dolfiny.utils.pprint("St. Venant-Kirchhoff solution, deformation scale:")
    dolfiny.utils.pprint(f"  max |u| = {u_max:.3e} m, w = {w:.3g} m, Pi = {u_max / w:.3e}")


def frame_beam(plotter):
    """Frame the beam tightly, leaving headroom for the horizontal colorbar."""
    plotter.view_xy()
    plotter.camera.zoom(2.9)
    # Shift the view upwards so the beam sits below the colorbar.
    fx, fy, fz = plotter.camera.focal_point
    px, py, pz = plotter.camera.position
    plotter.camera.focal_point = (fx, fy + 0.1 * h, fz)
    plotter.camera.position = (px, py + 0.1 * h, pz)


# Shared plotting configuration, applied once to the theme used by all figures below.
n_colors = 10
show_edges = False
window_size = (dolfiny.pyvista.pixels, dolfiny.pyvista.pixels // 3)
dolfiny.pyvista.theme.colorbar_horizontal.position_y = 0.83
dolfiny.pyvista.theme.colorbar_horizontal.height = 0.1
dolfiny.pyvista.theme.cmap = "coolwarm"


def save_field_plot(grid, scalars, filename, clim):
    """Render a single log-scaled scalar field on the beam and save as a PNG."""
    plotter = pv.Plotter(
        off_screen=True,
        theme=dolfiny.pyvista.theme,
        window_size=window_size,
        border=False,
    )
    plotter.add_mesh(
        grid.copy(deep=False),
        scalars=scalars,
        cmap="coolwarm",
        n_colors=10,
        log_scale=True,
        clim=clim,
        show_edges=False,
        lighting=False,
        scalar_bar_args={"title": ""},
    )
    frame_beam(plotter)
    plotter.screenshot(filename)
    plotter.close()
    plotter.deep_clean()


if comm.size == 1:
    # Create pyvista grid
    grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))

    # Add data to grid
    grid.cell_data["Neo-Hooke Unstable [Pa]"] = energy_fn.x.array
    grid.cell_data["Neo-Hooke 3rd Order Expansion [Pa]"] = energy_approx_fn.x.array

    save_field_plot(grid, "Neo-Hooke Unstable [Pa]", "energy_unstable.png", clim=(1, 1e4))
    save_field_plot(
        grid, "Neo-Hooke 3rd Order Expansion [Pa]", "energy_expansion.png", clim=(1, 1e4)
    )

# %% [markdown]
# ```{figure}
# :label: fig-energies
# :align: center
#
# ![Assembled unstable strain energy density W.](energy_unstable.png)
# ![Assembled stable strain energy density W_stable.](energy_expansion.png)
#
# Assembled strain energy densities (in Pa): unstable $W$ (top) and stable $W_\text{stable}$
# (bottom).
# ```

# %% tags=["hide-input"]
if comm.size == 1:
    # Create pyvista grid
    grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))

    # Add data to grid
    grid.cell_data["Rel. error bound (W, worst) [-]"] = np.abs(rel_error_fn.x.array)
    grid.cell_data["Rel. error estimate (W, exact) [-]"] = np.abs(rel_error_fn_exact.x.array)

    save_field_plot(
        grid, "Rel. error bound (W, worst) [-]", "error_unstable_worst.png", clim=(1e-1, 1e7)
    )
    save_field_plot(
        grid, "Rel. error estimate (W, exact) [-]", "error_unstable_exact.png", clim=(1e-1, 1e7)
    )

# %% [markdown]
# ```{figure}
# :label: fig-errors-unstable
# :align: center
#
# ![Rel. error bound for W and "worst" mode.](error_unstable_worst.png)
# ![Rel. error estimate for W and "exact" mode.](error_unstable_exact.png)
#
# Relative rounding error $\eta_i$ for the unstable expression $W$: "worst" mode bound (top)
# and "exact" mode estimate (bottom) show the pessimism of the worst-case bound.
# ```

# %% tags=["hide-input"]
if comm.size == 1:
    # Create pyvista grid
    grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))

    # Add data to grid
    grid.cell_data["Rel. error bound (W_stable, worst) [-]"] = np.abs(rel_error_approx_fn.x.array)
    grid.cell_data["Rel. error estimate (W_stable, exact) [-]"] = np.abs(
        rel_error_approx_fn_exact.x.array
    )

    save_field_plot(
        grid,
        "Rel. error bound (W_stable, worst) [-]",
        "error_expansion_worst.png",
        clim=(1e-8, 1e-2),
    )
    save_field_plot(
        grid,
        "Rel. error estimate (W_stable, exact) [-]",
        "error_expansion_exact.png",
        clim=(1e-8, 1e-2),
    )

# %% [markdown]
# ```{figure}
# :label: fig-errors-expansion
# :align: center
#
# ![Rel. error bound for W_stable and "worst" mode.](error_expansion_worst.png)
# ![Rel. error estimate for W_stable and "exact" mode.](error_expansion_exact.png)
#
# Relative rounding error $\eta_i$ for the stable expression $W_\text{stable}$: "worst" mode
# bound (top) and "exact" mode estimate (bottom) demonstrate significantly lower error
# accumulation due to improved numerical stability.
# ```
