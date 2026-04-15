# %% [markdown]
# # Rate-independent Rankine plasticity
#
# This demo solves a quasi-static small-strain boundary value problem in 2D using a
# Rankine (maximum principal stress) yield criterion with isotropic hardening.
#
# Compared with $J_2$ (von Mises) plasticity, which depends on deviatoric stress and is
# symmetric in tension/compression, the Rankine criterion is tension-driven: yielding starts when
# the largest principal stress reaches the tensile strength.
# The main numerical feature is **nonlinear static condensation** with `dolfiny.localsolver`:
# the constitutive unknowns are solved cell-by-cell inside each Newton step, and only the
# displacement field appears in the global linear solve, see {cite:t}`Habera2022LocalSolver`
# for details.
#
# In particular, this demo emphasizes:
# - cell-wise constitutive solves combined with nonlinear static condensation,
# - regularised complementarity and eigenvalue-based local plastic updates, and
# - localisation in a perforated specimen under tension-driven Rankine plasticity.
#
# ---
#
# ## Model
#
# The linearised strain is
# $$
#   \varepsilon = \text{sym}(\nabla u) = \varepsilon_{\mathrm{el}} + P
# $$
# Here $\varepsilon$ is the total small strain, $\varepsilon_{\mathrm{el}}$ is the elastic strain,
# $P$ is the accumulated plastic strain, and $\nabla$ denotes the spatial gradient. Hooke's law
# gives
# $$
#   \sigma = 2\mu \varepsilon_{\mathrm{el}} +
#   \lambda \text{tr}(\varepsilon_{\mathrm{el}}) I
# $$
# where $\sigma$ is the Cauchy stress, $\mu$ and $\lambda$ are the Lame constants, and $I$
# is the identity tensor. The Rankine yield function with isotropic hardening is
# $$
#   f(\sigma, \lambda_p) = \sigma_{\max}(\sigma) - S_y - H \lambda_p \le 0
# $$
# where $\sigma_{\max}(\sigma)$ is the largest principal stress, $S_y$ is the tensile yield
# stress, $H$ is the hardening modulus, and $\lambda_p$ is the accumulated plastic multiplier.
# For associated flow,
# $$
#   \dot{P} = \dot{\lambda}_p \frac{\partial f}{\partial \sigma}, \qquad
#   \dot{\lambda}_p \ge 0, \quad f \le 0, \quad \dot{\lambda}_p f = 0.
# $$
#
# In this implementation, $\partial f / \partial \sigma$ is obtained by automatic differentiation
# (`ufl.diff`) of a `ufl.variable`-tagged stress tensor.
#
# The time-discrete weak form uses increments $(\Delta P, \Delta\lambda_p)$ and requires, for all
# $(\delta u, \delta \Delta P, \delta \Delta\lambda_p)$:
# \begin{align}
#   \int_\Omega \sigma : \text{sym}(\nabla \delta u) \,\text{d}x &= 0, \\
#   \int_\Omega \delta \Delta P
#   : \big[\Delta P - \Delta\lambda_p \frac{\partial f}{\partial \sigma}\big] \,
#   \text{d}x &= 0, \\ \int_\Omega \delta \Delta\lambda_p
#   \phi_{\mathrm{FB},\eta}(\Delta\lambda_p, -f(\sigma, \lambda_{p,0} + \Delta\lambda_p))
#   \,\text{d}x &= 0
# \end{align}
# Here $\Omega$ is the spatial domain, $A : B$ denotes the Frobenius product of tensors, and
# $\lambda_{p,0}$ is the converged value from the previous load step.
#
# ## Non-linear condensation
#
# The local unknowns $(\Delta P, \Delta\lambda_p)$ are stored in quadrature spaces, so their
# degrees of freedom belong to one cell only. There is therefore no coupling between neighbouring
# cells in the constitutive problem.
#
# `dolfiny.localsolver` uses this structure in two stages inside every outer Newton iteration:
# first it solves, on each cell, the nonlinear local system
# $(F_{\Delta P}, F_{\Delta\lambda_p}) = 0$ for the current displacement iterate $u^{(k)}$;
# then it assembles a condensed global residual and Jacobian for the displacement field alone.
# The plastic variables are therefore eliminated locally, but in a nonlinear way because their
# current values depend on the current displacement iterate.
# In optimal control language, this is analogous to a reduced-functional approach:
# the local constitutive solve plays the role of a control-to-state map that expresses the local
# variables as functions of the current global displacement iterate.
#
# Writing the Jacobian in block form with displacement block $u$ and local constitutive block
# $l$, `localsolver` assembles the condensed tangent
# $$
#   \tilde{K}_{uu} = K_{uu} - K_{ul} K_{ll}^{-1} K_{lu}
# $$
# after the local Newton solve has produced a cellwise consistent constitutive state. Here
# $K_{uu}$ is the displacement-displacement block, $K_{ll}$ is the local-local block, and
# $K_{ul}$ and $K_{lu}$ are the coupling blocks.
#
# This is different from Schur condensation. In a Schur-condensed formulation one first builds the
# global residual and tangent for all fields $(u, \Delta P, \Delta\lambda_p)$ around the current
# iterate, and only then algebraically condenses that linearised system to a smaller system for
# $u$. In that setting the local residuals are not enforced during the outer iterations; they
# vanish only at the final converged equilibrium. Here, by contrast, the local residuals are
# solved to consistency at every Newton iterate before the condensed global system is assembled.

# %% [markdown]
# :::{warning}
# This demo is run with **positive isotropic hardening** ($H > 0$).
# It can also be run with **negative hardening** ($H < 0$), i.e. softening, but then the
# pathological mesh dependence of local plasticity becomes central: refining the mesh yields a
# narrower localisation band and a lower dissipated energy, so the solution does not converge to a
# physically meaningful limit.
# In addition, the simple displacement-control loop used here is usually not sufficient for robust
# post-peak tracing in the softening regime; arc-length methods such as Crisfield or Riks are
# typically needed.
# Regularisation is required to obtain mesh-objective results. Common approaches include gradient
# plasticity, crack band-width scaling, non-local integral models, and phase-field fracture;
# see {cite:t}`Shen2025` for an overview and comparative study.
# :::
#
# ## Parameters and mesh
#
# The geometry is a unit square $[0,1]^2$ perforated by a staggered row of small circular holes
# around mid-height ($y \approx 0.5$). These holes create a sequence of thin ligaments and local
# stress raisers, so the tensile failure path is no longer straight or unique.
#
# The loading is uniaxial vertical tension: the bottom edge ($y=0$) is constrained in $y$,
# a single point is pinned in $x$ to remove the horizontal rigid-body mode (while allowing
# Poisson contraction), and the top edge ($y=1$) is displaced vertically. This produces a
# predominantly tensile $\sigma_{yy}$ field, with localisation competing between the perforations,
# which is the regime for the Rankine criterion.
#
# Material parameters correspond to a generic brittle solid: $\mu = 100$ GPa, $\lambda = 10$
# GPa, tensile yield stress $S_y = 0.3$ GPa, and hardening modulus $H = 0.1$ GPa. The notation
# $\sigma_{yy}$ below refers to the normal stress in the loading direction.

# %% tags=["hide-input", "hide-output"]
from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl

import matplotlib.pyplot as plt
import numpy as np
import pyvista
from mesh_perforated import mesh_perforated

import dolfiny

comm = MPI.COMM_WORLD
name = "rankine"
gmsh_model, tdim = mesh_perforated(name, clscale=0.02)

# Get mesh and meshtags
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0, gdim=2)
mesh = mesh_data.mesh

# Write mesh and meshtags to file
with dolfiny.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_data(mesh_data)

# Boundary facets
top_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 1.0))
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 0.0))

facet_dim = mesh.topology.dim - 1
TOP, BOTTOM = 1, 2
marked_facets = np.hstack([top_facets, bottom_facets]).astype(np.int32)
marked_values = np.hstack(
    [np.full_like(top_facets, TOP), np.full_like(bottom_facets, BOTTOM)]
).astype(np.int32)
facet_order = np.argsort(marked_facets)
facet_tags = dolfinx.mesh.meshtags(
    mesh, facet_dim, marked_facets[facet_order], marked_values[facet_order]
)

# Material parameters
mu = 100  # shear modulus [GPa]
la = 10  # first Lamé parameter [GPa]
Sy = 0.3  # tensile yield stress [GPa]
H = 0.1  # isotropic hardening modulus [GPa]

if comm.size == 1:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_mesh(
        grid, show_edges=True, color="white", line_width=dolfiny.pyvista.pixels // 1000
    )
    plotter.show_axes()
    plotter.view_xy()
    plotter.screenshot(f"{name}_mesh.png")
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ```{figure} rankine_mesh.png
# :alt: Perforated specimen mesh showing the staggered circular holes that act as stress raisers.
# :align: center
# :label: fig-rankine-mesh
#
# Perforated specimen mesh showing the staggered circular holes that act as stress raisers.
# ```

# %% [markdown]
# ## Finite element discretisation and weak form
#
# Displacement $u$ uses continuous $P_1$ Lagrange elements. The local constitutive unknowns
# $(\Delta P, \Delta\lambda_p)$ use quadrature elements (degree 1), i.e. Gauss-point DOFs with no
# inter-element continuity. This is what makes nonlinear static condensation possible here:
# the constitutive update is purely cell-local, while displacement remains the only globally
# coupled field.
#
# The accumulated state $(P_0, \lambda_{p,0})$ is stored in the same quadrature spaces and updated
# after each converged load step, while $(\Delta P, \Delta\lambda_p)$ denotes the current local
# increment during one load step.
#
# The yield function $f$ uses a regularised closed form for $\sigma_{\max}$ in 2D:
# if $p = \text{tr}(\sigma)/2$ and
# $q = \sqrt{((\sigma_{11}-\sigma_{22})/2)^2 + \sigma_{12}^2}$, then
# $\sigma_{\max} = p + \sqrt{q^2 + \varepsilon^2}$. The $\varepsilon$-regularisation removes the
# eigenvalue-degeneracy singularity and keeps local Newton derivatives bounded.
#
# The NCP equation is written with the regularised Fischer-Burmeister function
# $$
#   \phi_{\mathrm{FB},\eta}(a, b) = \sqrt{a^2 + b^2 + \eta^2} - a - b,
# $$
# so the local complementarity condition becomes
# $$
#   \phi_{\mathrm{FB},\eta}(\Delta\lambda_p, -f(\sigma, \lambda_{p,0} + \Delta\lambda_p)) = 0.
# $$
# This replaces the exact min-type NCP by a smooth approximation with small parameter $\eta$.
# The local Newton solve is still safeguarded because two features remain delicate numerically:
# the regularised eigenvalue expression in $\sigma_{\max}$ and the regularised NCP function.
# We therefore use a simple backtracking line search with an Armijo condition on the merit
# function
# $$
#   \Psi(z) = \frac{1}{2} \|R(z)\|^2,
# $$
# where $R$ collects the local residuals for $(\Delta P, \Delta\lambda_p)$, $z$ is the stacked
# local unknown vector, and $\alpha$ is the backtracking step length. Trial updates
# $z_{\mathrm{trial}} = z - \alpha \Delta z$ are accepted only if $\Psi$ decreases sufficiently.

# %% tags=["hide-input"]
quad_degree = 1
Ue = basix.ufl.element("P", mesh.basix_cell(), 1, shape=(2,))
Qe = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=quad_degree)
Pe = basix.ufl.blocked_element(Qe, shape=(2, 2), symmetry=True)
Ze = basix.ufl.mixed_element([Pe, Qe])  # local state = (dP, dl)

Uf = dolfinx.fem.functionspace(mesh, Ue)
Zf = dolfinx.fem.functionspace(mesh, Ze)

u = dolfinx.fem.Function(Uf, name="u")
z = dolfinx.fem.Function(Zf, name="z")  # current local increment = (dP, dl)
z0 = dolfinx.fem.Function(Zf, name="z0")  # accumulated history   = (P0, l0)

dP, dl = ufl.split(z)
P0, l0 = ufl.split(z0)

δu = ufl.TestFunction(Uf)
δz = ufl.TestFunction(Zf)
δdP, δdl = ufl.split(δz)

# for output (P1 interpolation of displacement for XDMF compatibility)
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (2,))), name="u")


def f(sigma, lam, eps_eig=1e-3 * Sy):
    """Rankine yield function with regularised max eigenvalue (2D).

    The regularisation replaces q = sqrt(s^2 + t^2) with sqrt(s^2 + t^2 + eps^2),
    bounding the second derivative of f w.r.t. sigma at O(1/eps). This keeps
    the local Newton Jacobian well-conditioned near degenerate stress states
    (sigma_1 ≈ sigma_2) while shifting the yield surface by at most eps at
    the vertex and is negligible for this perforated-specimen demonstration.
    """
    p = ufl.tr(sigma) / 2
    q = ufl.sqrt(((sigma[0, 0] - sigma[1, 1]) / 2) ** 2 + sigma[0, 1] ** 2 + eps_eig**2)
    return p + q - Sy - H * lam


# Strain measures
E = ufl.sym(ufl.grad(u))  # linearised total strain
E_el = E - (P0 + dP)  # elastic strain: E_el = E - P

sigma_expr = 2 * mu * E_el + la * ufl.tr(E_el) * ufl.Identity(2)
sigma = ufl.variable(sigma_expr)  # tag for automatic differentiation

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

F0 = ufl.inner(sigma, ufl.sym(ufl.grad(δu))) * dx


def phi_fb(a, b, eta=1e-10 * Sy):
    return ufl.sqrt(a * a + b * b + eta * eta) - a - b


g = -f(sigma, l0 + dl)

F1 = (ufl.inner(dP - dl * ufl.diff(f(sigma, l0 + dl), sigma), δdP) + phi_fb(dl, g) * δdl) * dx

# Boundary conditions
# Bottom edge: u_y = 0 (allow horizontal sliding for Poisson contraction)
Uf_y, _ = Uf.sub(1).collapse()
u_bottom_y = dolfinx.fem.Function(Uf_y, name="u_bottom_y")
bottom_dofs_y = dolfinx.fem.locate_dofs_topological((Uf.sub(1), Uf_y), 1, bottom_facets)
bc_bottom_y = dolfinx.fem.dirichletbc(u_bottom_y, bottom_dofs_y, Uf.sub(1))

# Pin u_x = 0 at bottom-left corner to remove horizontal rigid body mode
Uf_x, _ = Uf.sub(0).collapse()
u_pin_x = dolfinx.fem.Function(Uf_x, name="u_pin_x")
pin_vertices = dolfinx.mesh.locate_entities_boundary(
    mesh, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
)
pin_dofs_x = dolfinx.fem.locate_dofs_topological((Uf.sub(0), Uf_x), 0, pin_vertices)
bc_pin_x = dolfinx.fem.dirichletbc(u_pin_x, pin_dofs_x, Uf.sub(0))

# Top edge: prescribed u_y displacement
u_top_y = dolfinx.fem.Function(Uf_y, name="u_top_y")
top_dofs_y = np.asarray(
    dolfinx.fem.locate_dofs_topological((Uf.sub(1), Uf_y), 1, top_facets), dtype=np.int32
)
owned_size_u = Uf.dofmap.index_map.size_local * Uf.dofmap.index_map_bs
top_dofs_y_owned = top_dofs_y[top_dofs_y < owned_size_u]
bc_top_y = dolfinx.fem.dirichletbc(u_top_y, top_dofs_y, Uf.sub(1))

bcs = [bc_bottom_y, bc_pin_x, bc_top_y]

# %% [markdown]
# ## Local solver kernels
#
# `dolfiny.localsolver` is configured with three C++ kernels, each acting on one cell at a time:
#
# - `solve_z`: performs the constitutive update. For the current displacement iterate on that cell,
#   it runs a local Newton iteration with backtracking line search for
#   $(\Delta P, \Delta\lambda_p)$, i.e. it solves
#   $(F_{\Delta P}, F_{\Delta\lambda_p}) = 0$.
# - `sc_J`: forms the condensed element tangent
#   $\tilde{K}_{uu} = K_{uu} - K_{ul} K_{ll}^{-1} K_{lu}$ once the local state is consistent.
# - `sc_F_cell`: returns the condensed element residual for the displacement block.
#
# The helper `local_update` below transfers the current iterate into the local work vector,
# assembles the local forms, and scatters the updated cellwise solution back into the quadrature
# functions before the condensed global assembly proceeds. In other words, the condensed global
# solve sees only the displacement residual and tangent after the local variables have been
# refreshed on every cell.
#
# We use `dolfiny.localsolver.view()` to obtain diagnostic information about the data layout
# expected for the local kernels.

# %% tags=["hide-input"]
sc_J = dolfiny.localsolver.UserKernel(
    name="sc_J",
    code=r"""
    template <typename T>
    void sc_J(T& A)
    {
        A = J00.array - J01.array * J11.array.partialPivLu().solve(J10.array);
    }
    """,
    required_J=[(0, 0), (0, 1), (1, 0), (1, 1)],
)

sc_F_cell = dolfiny.localsolver.UserKernel(
    name="sc_F_cell",
    code=r"""
    template <typename T>
    void sc_F_cell(T& A)
    {
        A = F0.array;
    }
    """,
    required_J=[],
)


solve_body = r"""
    // Mixed local field z = [dP11, dP22, dP12, dl]
    auto zloc = Eigen::Map<Eigen::Matrix<double, 4, 1>>(&F1.w[6]);

    Eigen::Matrix<double, 4, 1> loc = zloc;
    Eigen::Matrix<double, 4, 1> dloc;
    Eigen::Matrix<double, 4, 1> R;
    Eigen::Matrix<double, 4, 4> Jll;

    double err0 = 0.0;
    double alpha_prev = 1.0;
    const int N = 100;

    for (int i = 0; i < N; ++i)
    {
        F1.array.setZero();
        F1.kernel(F1.array.data(), F1.w.data(), F1.c.data(),
                  F1.coords.data(), F1.entity_local_index.data(),
                  F1.permutation.data(), nullptr);

        R = F1.array;
        const double err = R.norm();
        if (i == 0)
            err0 = err;

        if ((err < 1e-8 * err0) || (err < 1e-12))
            break;

        if (i == (N - 1))
            throw std::runtime_error("Local Newton failed to converge.");

        J11.array.setZero();
        J11.kernel(J11.array.data(), J11.w.data(), J11.c.data(),
                   J11.coords.data(), J11.entity_local_index.data(),
                   J11.permutation.data(), nullptr);

        Jll = J11.array;
        dloc = Jll.partialPivLu().solve(R);

        // Backtracking line search with Armijo on Psi = 0.5 * ||R||^2
        {
            const double c1 = 1e-4;
            const double rho = 0.5;
            const double alpha_min = 1e-12;

            Eigen::Matrix<double, 4, 1> loc_old = loc;
            const double psi0 = 0.5 * R.squaredNorm();

            bool accepted = false;
            double alpha = std::min(1.0, alpha_prev / rho);

            for (; alpha >= alpha_min; alpha *= rho)
            {
                Eigen::Matrix<double, 4, 1> loc_trial = loc_old - alpha * dloc;

                F1.w(Eigen::seq(6, 9)) = loc_trial;
                J11.w(Eigen::seq(6, 9)) = loc_trial;

                F1.array.setZero();
                F1.kernel(F1.array.data(), F1.w.data(), F1.c.data(),
                        F1.coords.data(), F1.entity_local_index.data(),
                        F1.permutation.data(), nullptr);

                const double psi_trial = 0.5 * F1.array.squaredNorm();

                if (psi_trial <= psi0 - c1 * alpha * R.squaredNorm())
                {
                    loc = loc_trial;
                    accepted = true;
                    alpha_prev = alpha;
                    break;
                }
            }

            if (!accepted)
            {
                F1.w(Eigen::seq(6, 9)) = loc_old;
                J11.w(Eigen::seq(6, 9)) = loc_old;
                throw std::runtime_error("Local line search failed.");
            }

            F1.w(Eigen::seq(6, 9)) = loc;
            J11.w(Eigen::seq(6, 9)) = loc;
        }
    }
"""

solve_z = dolfiny.localsolver.UserKernel(
    name="solve_z",
    code=f"""
    template <typename T>
    void solve_z(T& A)
    {{
        {solve_body}
        A = loc;
    }}
    """,
    required_J=[(1, 1)],
)


def local_update(problem):
    with problem.xloc.localForm() as x_local:
        x_local.set(0.0)

    dolfinx.fem.petsc.assign(
        problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id]
    )

    for idx in problem.localsolver.local_spaces_id:
        problem.u[idx].x.scatter_forward()

    # Assemble into local vector and scatter to functions
    dolfinx.fem.petsc.assemble_vector(problem.xloc, problem.local_form)
    problem.xloc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.assign(
        problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id]
    )

    for idx in problem.localsolver.local_spaces_id:
        problem.u[idx].x.scatter_forward()


all_cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local)

ls = dolfiny.localsolver.LocalSolver(
    [Uf, Zf],
    local_spaces_id=[1],
    F_integrals=[{dolfinx.fem.IntegralType.cell: [(0, sc_F_cell, all_cells)]}],
    J_integrals=[[{dolfinx.fem.IntegralType.cell: [(0, sc_J, all_cells)]}]],
    local_integrals=[
        {dolfinx.fem.IntegralType.cell: [(0, solve_z, all_cells)]},
    ],
    local_update=local_update,
)

opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "bt"
opts["snes_linesearch_order"] = 1
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-8
opts["snes_max_it"] = 100
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_cntl_1"] = 0.0
opts["snes_linesearch_monitor"] = ""

problem = dolfiny.snesproblem.SNESProblem([F0, F1], [u, z], bcs=bcs, prefix=name, localsolver=ls)

ls.view()

# %% [markdown]
# ## Loading
#
# A loading-unloading cycle is applied in vertical tension: the top edge ($y=1$) is displaced
# vertically to a peak of $\bar{u}_y = 0.01$ m in $K = 60$ equal increments, then returned to
# zero in $K / 2 = 30$ increments. Here $\bar{u}_y$ denotes the prescribed top-edge displacement.
# The bottom edge ($y=0$) is held fixed in $y$; horizontal sliding is permitted (Poisson
# contraction) except at the pinned corner.

# %% tags=["hide-input", "hide-output"]
K = 60  # load steps
du1 = 0.01  # peak vertical displacement [m]
load = np.linspace(0.0, 1.0, num=K, endpoint=False)
unload = np.linspace(1.0, 0.0, num=K // 2)
cycle = np.concatenate([load, unload])

# Force-displacement results from measurable boundary quantities
results: dict[str, list[float]] = {"displacement": [], "force": []}
F0_form = dolfinx.fem.form(F0)
reaction_vec = dolfinx.fem.petsc.create_vector(Uf)

# L2-project ||P0|| onto DG0 (cell average) — quadrature fields cannot be interpolated directly
Sf = dolfinx.fem.functionspace(mesh, ("DG", 0))
eps_p = dolfinx.fem.Function(Sf, name="eps_p")
f_sig = dolfinx.fem.Function(Sf, name="f")

for step, factor in enumerate(cycle):
    dolfiny.utils.pprint(f"\n+++ Processing step {step:3d}, load factor = {factor:5.4f}")

    # Update prescribed y-displacement on top edge
    u_top_y.interpolate(lambda x: du1 * factor * np.ones_like(x[1]))
    dolfiny.utils.pprint(f"Applied displacement: {1.0e3 * du1 * factor:.3f} mm")

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    z0.x.array[:] += z.x.array[:]
    z0.x.scatter_forward()

    z.x.array[:] = 0.0
    z.x.scatter_forward()

    # Collect imposed top-edge displacement (in mm for plotting) and the summed
    # vertical reaction on the top-edge DOFs.
    with reaction_vec.localForm() as reaction_local:
        reaction_local.set(0.0)
    dolfinx.fem.petsc.assemble_vector(reaction_vec, F0_form)
    reaction_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    reaction_y = comm.allreduce(reaction_vec.array[top_dofs_y_owned].sum(), op=MPI.SUM)
    results["displacement"].append(1.0e3 * du1 * factor)
    results["force"].append(reaction_y)

    # Interpolate and write output
    dolfiny.interpolation.interpolate(u, uo)
    dolfiny.projection.project(ufl.sqrt(ufl.inner(P0, P0)), eps_p)
    dolfiny.projection.project(f(sigma, l0), f_sig)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "a") as ofile:
        ofile.write_function(uo, step)
        ofile.write_function(eps_p, step)
        ofile.write_function(f_sig, step)

if comm.size == 1:
    # Build pyvista grid and attach displacement (padded to 3 components for warping)
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(uo.function_space.mesh))
    u_arr = np.zeros((uo.x.array.reshape(-1, 2).shape[0], 3))
    u_arr[:, :2] = uo.x.array.reshape(-1, 2)
    grid.point_data["u"] = u_arr
    grid_warped = grid.warp_by_vector("u", factor=1)

    # Attach cell data from DG0 field
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
    plotter.view_xy()
    plotter.screenshot(f"{name}_deformed.png")
    plotter.close()
    plotter.deep_clean()

# %% [markdown]
# ```{figure} rankine_deformed.png
# :alt: Deformed specimen with accumulated plastic strain localisation bands.
# :align: center
# :label: fig-rankine-plastic-strain
#
# Deformed specimen coloured by accumulated plastic strain magnitude, showing
# localisation bands through the ligaments between perforations.
# ```

# %% [markdown]
# ## Force-displacement curve
#
# Force-displacement response based on measurable boundary quantities: imposed top-edge
# displacement $\bar{u}_y$ in mm and the summed vertical reaction force $F_y$ on the prescribed
# top-edge DOFs. In this 2D setting $F_y$ is a force per unit thickness.

# %% tags=["hide-input", "hide-output"]
plt.figure(dpi=300)
plt.title("Rankine plasticity: perforated specimen", fontsize=12)
plt.xlabel(r"top-edge displacement $\bar{u}_y$ [mm]", fontsize=12)
plt.ylabel(r"vertical reaction force $F_y$ [GN/m]", fontsize=12)
plt.grid(linewidth=0.25)
plt.plot(
    np.array(results["displacement"]),
    np.array(results["force"]),
    linestyle="-",
    linewidth=1.0,
    markersize=4.0,
    marker=".",
    label=r"$F_y$-$\bar{u}_y$ curve",
)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{name}.png", dpi=300)
plt.close()

# %% [markdown]
# ```{figure} rankine.png
# :alt: Force-displacement curve for Rankine plasticity on a perforated specimen.
# :align: center
# :name: fig-force-displacement
# Force-displacement curve for Rankine plasticity on a perforated specimen,
# based on imposed top-edge displacement and the summed top-edge reaction force.
# ```
