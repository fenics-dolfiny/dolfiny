from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx.fem.function import FunctionSpace

import numpy as np
import pytest

import dolfiny
import dolfiny.inequality
import dolfiny.taoproblem


def _L2_norm(u: dolfinx.fem.Function) -> float:
    form = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)
    norm: float = np.sqrt(MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(form)))
    return norm


@pytest.mark.parametrize(
    "n,order,atol,element",
    [
        (4, 0, 1e-5, ("P", 1)),
        # (4, 0, 1e-5, ("P", 2)), TODO: hits max it
        (10, 1, 1e-7, ("P", 1)),
        # (10, 1, 1e-7, ("P", 2)), TODO hits max it
        (10, 2, 1e-13, ("P", 1)),
        (10, 2, 1e-13, ("P", 2)),
        (10, 2, 1e-13, ("P", 3)),
        (10, 2, 1e-13, ("P", 4)),
        # (10, 2, 1e-13, ("P", 5)), # TODO: diverges
    ],
)
def test_poisson_discrete(n, order, atol, element):
    """Solves poissons problem stated in discretized form."""
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    W = dolfinx.fem.functionspace(mesh, element)

    f = -1.0

    top = W.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, 0.0), boundary_dofs, W)

    u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = -ufl.inner(f, v) * ufl.dx

    a = dolfinx.fem.form(a)
    L = dolfinx.fem.form(L)

    A = dolfinx.fem.petsc.assemble_matrix(a, [bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])

    tmp = b.copy()
    tmp.zeroEntries()
    tmp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(tao: PETSc.TAO, x: PETSc.Vec) -> float:  # type: ignore
        A.mult(x, tmp)
        return 0.5 * tmp.dot(x) - b.dot(x)  # type: ignore

    def J(tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Vec) -> None:  # type: ignore
        J.zeroEntries()
        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        A.mult(x, J)
        J.axpy(-1, b)

        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    def H(tao: PETSc.TAO, x: PETSc.Vec, H: PETSc.Mat, P: PETSc.Mat) -> None:  # type: ignore
        pass

    opts = PETSc.Options("poisson_discrete")
    match order:
        case 0:
            opts["tao_type"] = "nm"
        case 1:
            opts["tao_type"] = "bqnls"
        case 2:
            opts["tao_type"] = "nls"

    opts["tao_gatol"] = 1e-12
    opts["tao_grtol"] = 0
    opts["tao_gttol"] = 0
    opts["tao_nls_ksp_type"] = "preonly"
    opts["tao_nls_pc_type"] = "cholesky"
    opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"

    u = dolfinx.fem.Function(W)
    opt_problem = dolfiny.taoproblem.TAOProblem(
        F, [u], bcs=[bc], J=(J, u.x.petsc_vec.copy()), H=(H, A), prefix="poisson_discrete"
    )
    opt_problem.solve()

    u_direct = dolfinx.fem.Function(W)
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        u=u_direct,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem.solve()

    # TODO:
    # assert opt_problem.tao.getConvergedReason() == PETSc.TAO.ConvergedReason.CONVERGED_GATOL
    # assert opt_problem.tao.getConvergedReason() > 0

    assert _L2_norm(u - u_direct) == pytest.approx(0, abs=atol)
    if order == 2:
        assert opt_problem.tao.getIterationNumber() == 1


# TODO: verify patch date aligns with https://gitlab.com/petsc/petsc/-/merge_requests/8386
@pytest.mark.parametrize("autodiff", [True, False])
@pytest.mark.parametrize(
    "order",
    [
        pytest.param(
            1,
            marks=pytest.mark.skipif(
                (PETSc.Sys().getVersion()[1] < 23) or (PETSc.Sys().getVersion()[2] < 3),  # type: ignore
                reason="Missing PETSc exports",
            ),
        ),
        2,
    ],
)
def test_poisson(autodiff: bool, order: int):
    """Tests poissons problem as variational problem, with different optimisation setups."""
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    W = dolfinx.fem.functionspace(mesh, ("P", 1))

    u = dolfinx.fem.Function(W)
    f = -1.0

    F = 1 / 2 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx - ufl.inner(f, u) * ufl.dx
    J = None if autodiff else [ufl.derivative(F, u)]
    H = None if autodiff else [[ufl.derivative(J[0], u)]]  # type: ignore

    top = W.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, np.array(0.0)), boundary_dofs, W)

    lb = PETSc.NINFINITY  # type: ignore
    ub = PETSc.INFINITY  # type: ignore
    if order == 1:
        lb = 0.0
        ub = 1.0

    opts = PETSc.Options("poisson")  # type: ignore
    opts["info"] = ""
    opts["tao_type"] = "bqnls" if order == 1 else "nls"
    opts["tao_recycle"] = ""
    opts["tao_gatol"] = 1e-10
    opts["tao_monitor"] = ""
    opts["tao_max_it"] = 1000
    opts["tao_ls_monitor"] = ""
    opts["tao_monitor"] = ""

    opt_problem = dolfiny.taoproblem.TAOProblem(
        F, [u], bcs=[bc], lb=lb, ub=ub, J=J, H=H, prefix="poisson"
    )

    if order == 1:
        mass = dolfinx.fem.form(ufl.TrialFunction(W) * ufl.TestFunction(W) * ufl.dx)
        M = dolfinx.fem.petsc.assemble_matrix(mass, bcs=[bc], diag=1)
        M.assemble()

        opt_problem.tao.getLMVMMat().setLMVMJ0(M)
        opt_problem.tao.setGradientNorm(M)

    opt_problem.solve()
    sol_optimization = u.copy()

    u.x.array[:] = 0
    a = ufl.derivative(ufl.derivative(F, u), u)
    L = -ufl.derivative(F, u)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form, _, _ = problem.solve()

    assert np.allclose(_L2_norm(sol_optimization - sol_weak_form), 0)
    assert opt_problem.tao.getConvergedReason() > 0
    if order == 2:
        # TODO: currently takes 2?
        assert opt_problem.tao.getIterationNumber() == 2


def test_poisson_mixed():
    """Solve mixed poisson formulation as blocked optimisation problem."""
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    k = 1
    V_σ = dolfinx.fem.functionspace(mesh, ("RT", k))
    V_u = dolfinx.fem.functionspace(mesh, ("DP", k - 1))

    σ, u = dolfinx.fem.Function(V_σ), dolfinx.fem.Function(V_u)

    x = ufl.SpatialCoordinate(mesh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)

    fdim = mesh.topology.dim - 1
    facets_top = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))
    facets_bottom = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    dofs_top = dolfinx.fem.locate_dofs_topological(V_σ, fdim, facets_top)
    dofs_bottom = dolfinx.fem.locate_dofs_topological(V_σ, fdim, facets_bottom)
    cells_top_ = dolfinx.mesh.compute_incident_entities(mesh.topology, facets_top, fdim, fdim + 1)
    cells_bottom = dolfinx.mesh.compute_incident_entities(
        mesh.topology, facets_bottom, fdim, fdim + 1
    )

    g = dolfinx.fem.Function(V_σ)
    g.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))), cells0=cells_top_)
    g.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))), cells0=cells_bottom
    )
    bcs = [dolfinx.fem.dirichletbc(g, dofs_top), dolfinx.fem.dirichletbc(g, dofs_bottom)]

    n = ufl.FacetNormal(mesh)
    F = -(
        1 / 2 * σ**2 * ufl.dx + ufl.div(σ) * u * ufl.dx + f * u * ufl.dx
        # - ufl.inner(σ, n) * f * ufl.dx # TODO
    )

    opts = PETSc.Options("poisson_mixed")
    opts["tao_type"] = "nls"
    opts["tao_ls_type"] = "unit"
    opts["tao_nls_ksp_type"] = "preonly"
    opts["tao_nls_pc_type"] = "lu"
    opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"
    opts["tao_monitor"] = ""

    opt_problem = dolfiny.taoproblem.TAOProblem(F, [σ, u], bcs=bcs, prefix="poisson_mixed")
    σ_opt, u_opt = opt_problem.solve()

    opts = PETSc.Options("poisson_mixed_direct")
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_max_it"] = 1
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    σ.x.array[:] = 0
    u.x.array[:] = 0

    δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V_σ, V_u))
    J = ufl.derivative(F, [σ, u], δm)
    J = ufl.extract_blocks(J)

    problem = dolfiny.snesproblem.SNESProblem(J, [σ, u], bcs=bcs, prefix="poisson_mixed_direct")
    σ_direct, u_direct = problem.solve()

    assert np.allclose(_L2_norm(σ_opt - σ_direct), 0)
    assert np.allclose(_L2_norm(u_opt - u_direct), 0)
    assert opt_problem.tao.getIterationNumber() == 1
    assert opt_problem.tao.getConvergedReason() > 0


@pytest.mark.skipif(
    (PETSc.Sys().getVersion()[1] < 23) or (PETSc.Sys().getVersion()[2] < 1),  # type: ignore
    reason="Missing PETSc exports",
)
@pytest.mark.parametrize(
    "eq_constrained",
    [True, False],
)
@pytest.mark.parametrize("autodiff", [True, False])
def test_poisson_constrained(V1: FunctionSpace, eq_constrained: bool, autodiff: bool):
    """Test Poisson problem with scalar eq-/inequality constraint."""
    W = V1

    # TODO: return to smaller mesh at some point
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    W = dolfinx.fem.functionspace(mesh, ("P", 1))

    u = dolfinx.fem.Function(W)
    f = -1.0

    F = 1 / 2 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx + ufl.inner(f, u) * ufl.dx
    J = None if autodiff else [ufl.derivative(F, u)]
    H = None if autodiff else [[ufl.derivative(J[0], u)]]  # type: ignore

    top = W.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, np.array(0.0)), boundary_dofs, W)

    a = ufl.derivative(ufl.derivative(F, u), u)
    L = -ufl.derivative(F, u)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form, _, _ = problem.solve()
    assert isinstance(sol_weak_form, dolfinx.fem.Function)
    L2_norm_unconstrained = _L2_norm(sol_weak_form)

    C = L2_norm_unconstrained * 0.5
    g = [u**2 * ufl.dx == C**2] if eq_constrained else None
    h = [(u**2) * ufl.dx >= C**2] if not eq_constrained else None

    v = ufl.TestFunction(W)
    Jg = None if not eq_constrained or autodiff else [[2 * u * v * ufl.dx]]
    Jh = None if eq_constrained or autodiff else [[-2 * u * v * ufl.dx]]

    opts = PETSc.Options("poisson_constrained")  # type: ignore

    opts["tao_type"] = "almm"
    opts["tao_gatol"] = 1e-6
    opts["tao_catol"] = 1e-3
    opts["tao_crtol"] = 1e-3
    opts["tao_almm_subsolver_tao_type"] = "bqnls"
    opts["tao_almm_subsolver_tao_ls_type"] = "unit"
    opts["tao_almm_subsolver_ksp_type"] = "preonly"
    opts["tao_almm_subsolver_pc_type"] = "lu"
    opts["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"

    opt_problem = dolfiny.taoproblem.TAOProblem(
        F, [u], bcs=[bc], J=J, H=H, g=g, Jg=Jg, h=h, Jh=Jh, prefix="poisson_constrained"
    )
    (sol_optimization,) = opt_problem.solve()

    assert _L2_norm(sol_optimization) == pytest.approx(L2_norm_unconstrained * 0.5, 1e-3)
    assert opt_problem.tao.getConvergedReason() > 0
    # TODO: subsolver convergence


def test_optimal_control_reduced():
    """Optimal control problem, test custom callback (reduced functional)."""
    # based on https://www.dolfin-adjoint.org/en/stable/documentation/poisson-mother/poisson-mother.html

    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    V_state = dolfinx.fem.functionspace(mesh, ("P", 1))
    V_control = dolfinx.fem.functionspace(mesh, ("DG", 0))

    top = V_state.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V_state, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(V_state.mesh, 0.0), boundary_dofs, V_state)

    u = dolfinx.fem.Function(V_state, name="state")
    d = dolfinx.fem.Function(V_state, name="state_desired")
    d.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    f = dolfinx.fem.Function(V_control)
    f.interpolate(lambda x: x[0] + x[1])

    alpha = 1e-4
    F = 0.5 * ufl.inner(u - d, u - d) * ufl.dx + alpha / 2 * f**2 * ufl.dx
    F = dolfinx.fem.form(F)
    v = ufl.TestFunction(V_state)

    a = ufl.inner(ufl.grad(ufl.TrialFunction(V_state)), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    L_adj = (u - d) * v * ufl.dx

    state_problem = dolfinx.fem.petsc.LinearProblem(a, L, [bc], u)
    p = dolfinx.fem.Function(V_state, name="p")
    adjoint_problem = dolfinx.fem.petsc.LinearProblem(a, L_adj, [bc], p)

    @dolfiny.taoproblem.sync_functions(f)
    def F_reduced(tao, x):
        state_problem.solve()

        local_J = dolfinx.fem.assemble_scalar(F)
        return MPI.COMM_WORLD.allreduce(local_J, op=MPI.SUM)

    jacobian = dolfinx.fem.Function(V_control)

    @dolfiny.taoproblem.sync_functions(f)
    def J_reduced(tao, x, J):
        state_problem.solve()

        adjoint_problem.solve()

        # Riesz scaling of gradient given by p + alpha * f
        dolfiny.projection.project(p + alpha * f, jacobian)

        dolfinx.fem.petsc.assign(jacobian, J)

    opts = PETSc.Options("optimal_control_reduced")
    opts["tao_type"] = "lmvm"  # pdipm
    opts["tao_gatol"] = 0
    opts["tao_gttol"] = 1e-6

    opt_problem = dolfiny.taoproblem.TAOProblem(
        F_reduced, [f], J=(J_reduced, f.x.petsc_vec.copy()), prefix="optimal_control_reduced"
    )
    opt_problem.solve()

    x = ufl.SpatialCoordinate(mesh)
    f_ana = 1 / (1 + 4 * alpha * np.pi**4) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    u_ana = 1 / (2 * np.pi**2) * f_ana

    assert np.allclose(0, _L2_norm(f - f_ana), atol=5e-2)
    assert np.allclose(0, _L2_norm(u - u_ana), atol=1e-4)
    assert opt_problem.tao.getConvergedReason() > 0


# @pytest.mark.parametrize("almm_type", [PETSc.TAO.ALMMType.CLASSIC, PETSc.TAO.ALMMType.PHR])
@pytest.mark.skipif(
    (PETSc.Sys().getVersion()[1] < 23) or (PETSc.Sys().getVersion()[2] < 2),  # type: ignore
    reason="Missing PETSc exports",
)
def test_optimal_control_full_space():  # almm_type
    """Optimal control problem as full space approach, singular jacobian."""

    # based on https://www.dolfin-adjoint.org/en/stable/documentation/poisson-mother/poisson-mother.html

    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    V_state = dolfinx.fem.functionspace(mesh, ("P", 1))
    V_control = dolfinx.fem.functionspace(mesh, ("DG", 0))

    top = V_state.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V_state, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(V_state.mesh, 0.0), boundary_dofs, V_state)

    u = dolfinx.fem.Function(V_state, name="state")
    u.x.petsc_vec.set(1.0)
    d = dolfinx.fem.Function(V_state, name="state_desired")
    d.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    f = dolfinx.fem.Function(V_control)
    f.interpolate(lambda x: x[0] + x[1])

    alpha = 1e-4
    F = 0.5 * ufl.inner(u - d, u - d) * ufl.dx + alpha / 2 * f**2 * ufl.dx
    v = ufl.TestFunction(V_state)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    g = [a - L == 0]
    Jg = [
        [
            ufl.derivative(g[0].lhs, u),
            ufl.derivative(g[0].lhs, f),
        ]
    ]

    opts = PETSc.Options("optimal_control_full_space")
    opts["tao_type"] = "almm"  # pdipm
    # opts["tao_almm_type"] = "classic" if almm_type == PETSc.TAO.ALMMType.CLASSIC else "phr"
    opts["tao_gatol"] = 1e-5

    opt_problem = dolfiny.taoproblem.TAOProblem(
        F, [u, f], bcs=[bc], g=g, Jg=Jg, prefix="optimal_control_full_space"
    )
    # mass = dolfinx.fem.form([[ufl.TrialFunction(V_state) * ufl.TestFunction(V_state) * ufl.dx,
    # None], [None, ufl.TrialFunction(V_control) * ufl.TestFunction(V_control) * ufl.dx]])
    # M = dolfinx.fem.petsc.assemble_matrix(mass, bcs=[bc], diag=1)
    # M.assemble()
    # opt_problem.tao.getALMMSubsolver().getLMVMMat().setLMVMJ0(M)
    # opt_problem.tao.getALMMSubsolver().setGradientNorm(M)
    opt_problem.solve()

    x = ufl.SpatialCoordinate(mesh)
    f_ana = 1 / (1 + 4 * alpha * np.pi**4) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    u_ana = 1 / (2 * np.pi**2) * f_ana

    assert np.allclose(0, _L2_norm(f - f_ana), atol=5e-2)
    assert np.allclose(0, _L2_norm(u - u_ana), atol=5e-4)
    assert opt_problem.tao.getConvergedReason() > 0


# @pytest.mark.parametrize("almm_type", [PETSc.TAO.ALMMType.PHR]) # TODO: PETSc.TAO.ALMMType.CLASSIC
@pytest.mark.skipif(
    (PETSc.Sys().getVersion()[1] < 23) or (PETSc.Sys().getVersion()[2] < 2),  # type: ignore
    reason="Missing PETSc exports",
)
def test_poisson_pde_as_constraint():  # almm_type
    """Poisson problem as constraint. Test variational constraints."""
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 1))

    top = V.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(V.mesh, 0.0), boundary_dofs, V)

    u = dolfinx.fem.Function(V, name="u")

    f = -1

    a = ufl.inner(ufl.grad(u), ufl.grad(ufl.TestFunction(V))) * ufl.dx
    L = f * ufl.TestFunction(V) * ufl.dx

    g = [a - L == 0]
    Jg = [[ufl.derivative(a - L, u)]]

    # TODO:
    # F = ufl.ZeroBaseForm((u,))
    F = 0.5 * dolfinx.fem.Constant(mesh, 0.0) * (u) ** 2 * ufl.dx
    opts = PETSc.Options("poisson_pde_constraint")
    opts["tao_type"] = "almm"  # pdipm
    # opts["tao_almm_type"] = "classic" if almm_type == PETSc.TAO.ALMMType.CLASSIC else "phr"
    opts["tao_gatol"] = 1e-6
    opts["tao_grtol"] = 1e-6
    opts["tao_catol"] = 1e-6
    opts["tao_crtol"] = 1e-6
    opts["tao_almm_subsolver_ksp_type"] = "preonly"
    opts["tao_almm_subsolver_pc_type"] = "lu"
    opts["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"

    opt_problem = dolfiny.taoproblem.TAOProblem(
        F, [u], bcs=[bc], g=g, Jg=Jg, prefix="poisson_pde_constraint"
    )
    opt_problem.solve()

    linear_problem = dolfinx.fem.petsc.LinearProblem(
        ufl.inner(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx, L, [bc]
    )
    u_lp, _, _ = linear_problem.solve()

    assert np.allclose(_L2_norm(u_lp - u), 0, atol=1e-4)
    assert opt_problem.tao.getConvergedReason() > 0
    # TODO: subsolver convergence?
