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
import dolfiny.taoblockproblem

# TODO: options prefix should be unique -> cross influence


def L2_norm(u: dolfinx.fem.Function) -> float:
    form = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)
    return np.sqrt(MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(form)))


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
        (10, 2, 1e-13, ("P", 5)),
    ],
)
def test_poisson_discrete(n, order, atol, element):
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
    L = -ufl.inner(f, u) * ufl.dx  # TODO: v!!!

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
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(tao: PETSc.TAO, x: PETSc.Vec) -> float:
        A.mult(x, tmp)
        return 0.5 * tmp.dot(x) - b.dot(x)

    def J(tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Vec) -> None:
        J.zeroEntries()
        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        A.mult(x, J)
        J.axpy(-1, b)

        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def H(tao: PETSc.TAO, x: PETSc.Vec, H: PETSc.Mat, P: PETSc.Mat) -> None:
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
    # opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    u = dolfinx.fem.Function(W)

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F, [u], bcs=[bc], J=(J, u.x.petsc_vec.copy()), H=(H, A), prefix="poisson_discrete"
    )
    (sol_optimization,) = opt_problem.solve()

    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        u=dolfinx.fem.Function(W),
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    sol_weak_form = problem.solve()

    # TODO:
    # assert opt_problem.tao.getConvergedReason() == PETSc.TAO.ConvergedReason.CONVERGED_GATOL
    # assert opt_problem.tao.getConvergedReason() > 0

    assert L2_norm(sol_optimization - sol_weak_form) == pytest.approx(0, abs=atol)
    if order == 2:
        assert opt_problem.tao.getIterationNumber() == 1

    # with dolfinx.io.VTXWriter(W.mesh.comm, "opt.bp", sol_optimization, "bp4") as file:
    #     file.write(0.0)
    # with dolfinx.io.VTXWriter(W.mesh.comm, "direct.bp", sol_weak_form, "bp4") as file:
    #     file.write(0.0)


# TODO: add test case without dbc
@pytest.mark.parametrize("autodiff", [True, False])
def test_poisson(autodiff: bool):
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    W = dolfinx.fem.functionspace(mesh, ("P", 1))

    u = dolfinx.fem.Function(W)
    f = -1.0

    F = 1 / 2 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx + ufl.inner(f, u) * ufl.dx
    J = None if autodiff else [ufl.derivative(F, u, ufl.TestFunction(W))]
    H = None if autodiff else [[ufl.derivative(J[0], u, ufl.TrialFunction(W))]]

    top = W.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, 0.0), boundary_dofs, W)

    opts = PETSc.Options("poisson")
    opts["tao_type"] = "nls"
    opts["tao_gatol"] = 1e-12
    opts["tao_grtol"] = 0
    opts["tao_gttol"] = 0
    opts["tao_nls_ksp_type"] = "preonly"
    opts["tao_nls_pc_type"] = "cholesky"
    opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"
    # opts["tao_monitor"] = ""form_constraints
    # opts["tao_ls_monitor"] = ""

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F, [u], bcs=[bc], J=J, H=H, prefix="poisson"
    )
    (sol_optimization,) = opt_problem.solve()

    # TODO: if derivative stay consistent: two derivatives no replace.
    weak_form = ufl.replace(ufl.derivative(F, u, ufl.TestFunction(W)), {u: ufl.TrialFunction(W)})
    a, L = ufl.lhs(weak_form), ufl.rhs(weak_form)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form = problem.solve()

    assert np.allclose(L2_norm(sol_optimization - sol_weak_form), 0)
    assert opt_problem.tao.getIterationNumber() == 1
    assert opt_problem.tao.getConvergedReason() > 0

    # with dolfinx.io.VTXWriter(W.mesh.comm, "opt.bp", u, "bp4") as file:
    #     file.write(0.0)
    # with dolfinx.io.VTXWriter(W.mesh.comm, "direct.bp", sol_weak_form, "bp4") as file:
    #     file.write(0.0)


def test_poisson_mixed():
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
        1 / 2 * (σ**2 * ufl.dx + ufl.div(σ) * u * ufl.dx + ufl.div(σ) * u * ufl.dx) + f * u * ufl.dx
        # - ufl.inner(σ, n) * f * ufl.dx # TODO
    )

    opts = PETSc.Options("opt")
    opts["tao_type"] = "nls"
    opts["tao_ls_type"] = "unit"
    # opts["tao_max_it"] = 1
    opts["tao_nls_ksp_type"] = "preonly"
    opts["tao_nls_pc_type"] = "lu"
    opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"
    opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(F, [σ, u], bcs=bcs, prefix="opt")
    σ_opt, u_opt = opt_problem.solve()

    opts = PETSc.Options("direct")  # type: ignore[attr-defined]
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_max_it"] = 1
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    σ.x.petsc_vec.zeroEntries()
    σ.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u.x.petsc_vec.zeroEntries()
    u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    δu, δσ = ufl.TestFunction(V_u), ufl.TestFunction(V_σ)
    J = dolfiny.expression.derivative(F, [σ, u], [δσ, δu])
    J = dolfiny.function.extract_blocks(J, [δσ, δu])

    # TODO: this does not work - why?
    # δδu, δδσ = ufl.TrialFunction(V_u), ufl.TrialFunction(V_σ)
    # H = dolfiny.expression.derivative(J, [σ, u], [δδσ, δδu])
    # H = dolfiny.function.extract_blocks(H, [δσ, δu], [δδσ, δδu])

    # TODO: remove if above works
    H = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            uj = [σ, u][j]
            duj = ufl.TrialFunction(uj.function_space)
            H[i][j] = ufl.derivative(J[i], uj, duj)

            # If the form happens to be empty replace with None
            if H[i][j].empty():
                H[i][j] = None

    problem = dolfiny.snesblockproblem.SNESBlockProblem(
        J, [σ, u], J_form=H, bcs=bcs, prefix="direct"
    )
    σ_direct, u_direct = problem.solve()

    assert np.allclose(L2_norm(σ_opt - σ_direct), 0)
    assert np.allclose(L2_norm(u_opt - u_direct), 0)
    assert opt_problem.tao.getIterationNumber() == 1
    assert opt_problem.tao.getConvergedReason() > 0

    local = dolfinx.fem.Function(V_u, name="process")
    local.x.array[:] = MPI.COMM_WORLD.rank

    # with dolfinx.io.VTXWriter(V_u.mesh.comm, "direct_u.bp", u, "bp4") as file:
    #     file.write(0.0)
    # with dolfinx.io.VTXWriter(V_u.mesh.comm, "opt_u.bp", u_opt, "bp4") as file:
    #     file.write(0.0)


def test_poisson_blocked():
    n = 32
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    V1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    V2 = dolfinx.fem.functionspace(mesh, ("P", 1))

    u1 = dolfinx.fem.Function(V1)
    u2 = dolfinx.fem.Function(V2)
    f = -1.0

    F = (
        0.5
        * (
            ufl.inner(ufl.grad(u1), ufl.grad(u1)) * ufl.dx
            + ufl.inner(ufl.grad(u2), ufl.grad(u2)) * ufl.dx
        )
        + ufl.inner(f, u1) * ufl.dx
        + ufl.inner(f, u2) * ufl.dx
    )

    top = V1.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)

    def get_bc(V):
        boundary_dofs = dolfinx.fem.locate_dofs_topological(V, top.dim - 1, boundary_facets)
        return dolfinx.fem.dirichletbc(dolfinx.fem.Constant(V.mesh, 0.0), boundary_dofs, V)

    bcs = [get_bc(V1), get_bc(V2)]

    opts = PETSc.Options("poisson")
    opts["tao_type"] = "nls"
    opts["tao_gatol"] = 1e-12
    opts["tao_grtol"] = 0
    opts["tao_gttol"] = 0
    opts["tao_nls_ksp_type"] = "preonly"
    opts["tao_nls_pc_type"] = "cholesky"
    opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"
    # opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    J = [ufl.derivative(F, u1, ufl.TestFunction(V1)), ufl.derivative(F, u2, ufl.TestFunction(V2))]

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F, [u1, u2], bcs=bcs, J=J, prefix="poisson"
    )
    sol_u1, sol_u2 = opt_problem.solve()

    weak_form = ufl.replace(J[0], {u1: ufl.TrialFunction(V1)})
    a, L = ufl.lhs(weak_form), ufl.rhs(weak_form)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bcs[0]], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form = problem.solve()

    assert np.allclose(L2_norm(sol_u1 - sol_weak_form), 0)
    assert np.allclose(L2_norm(sol_u2 - sol_weak_form), 0)
    assert opt_problem.tao.getIterationNumber() == 1
    assert opt_problem.tao.getConvergedReason() > 0


@pytest.mark.skipif(
    (PETSc.Sys().getVersion()[1] < 23) or (PETSc.Sys().getVersion()[2] < 1),
    reason="Missing PETSc exports",
)
@pytest.mark.parametrize(
    "eq_constrained",
    [True, False],
)
def test_poisson_constrained(V1: FunctionSpace, eq_constrained: bool):
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
    J = ufl.derivative(F, u, ufl.TestFunction(W))
    H = ufl.derivative(J, u, ufl.TrialFunction(W))

    top = W.mesh.topology
    top.create_connectivity(top.dim - 1, top.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, 0.0), boundary_dofs, W)

    weak_form = ufl.replace(J, {u: ufl.TrialFunction(W)})
    a, L = ufl.lhs(weak_form), ufl.rhs(weak_form)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form = problem.solve()
    L2_norm_unconstrained = L2_norm(sol_weak_form)

    C = L2_norm_unconstrained * 0.5
    g = [u**2 * ufl.dx == C**2] if eq_constrained else None
    h = [(u**2) * ufl.dx >= C**2] if not eq_constrained else None

    v = ufl.TestFunction(W)
    Jg = [[2 * u * v * ufl.dx]] if eq_constrained else None
    Jh = [[-2 * u * v * ufl.dx]] if not eq_constrained else None

    opts = PETSc.Options("poisson_constrained")

    opts["tao_type"] = "almm"
    opts["tao_gatol"] = 1e-6
    opts["tao_catol"] = 1e-3
    opts["tao_crtol"] = 1e-3
    # opts["tao_almm_type"] = "classic"
    # opts["tao_almm_mu_init"] = 1.5
    # opts["tao_almm_mu_factor"] = 1.2
    # opts["tao_almm_mu_factor"] = 1e5
    # opts["tao_max_it"] = 50
    # opts["tao_almm_subsolver_tao_test_gradient"] = ""
    opts["tao_almm_subsolver_tao_type"] = "bqnls"
    opts["tao_almm_subsolver_tao_ls_type"] = "unit"
    opts["tao_almm_subsolver_ksp_type"] = "preonly"
    opts["tao_almm_subsolver_pc_type"] = "lu"
    opts["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"
    # opts["tao_almm_subsolver_tao_monitor"] = ""
    # opts["tao_almm_subsolver_tao_ls_monitor"] = ""
    # opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F, [u], bcs=[bc], J=[J], H=[[H]], g=g, Jg=Jg, h=h, Jh=Jh, prefix="poisson_constrained"
    )
    (sol_optimization,) = opt_problem.solve()

    assert L2_norm(sol_optimization) == pytest.approx(L2_norm_unconstrained * 0.5, 1e-3)
    assert opt_problem.tao.getConvergedReason() > 0

    # with dolfinx.io.VTXWriter(W.mesh.comm, "opt.bp", u, "bp4") as file:
    #     file.write(0.0)
    # with dolfinx.io.VTXWriter(W.mesh.comm, "direct.bp", sol_weak_form, "bp4") as file:
    #     file.write(0.0)


def test_optimal_control_reduced():
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
    # alpha = 1e-3
    F = 0.5 * ufl.inner(u - d, u - d) * ufl.dx + alpha / 2 * f**2 * ufl.dx
    F = dolfinx.fem.form(F)
    v = ufl.TestFunction(V_state)
    # F_u = ufl.inner(u-d, v) * ufl.dx
    # F_f = f*ufl.TestFunction(V_control) * ufl.dx
    # J = ufl.derivative(F, u, ufl.TestFunction(W))
    # H = ufl.derivative(J, u, ufl.TrialFunction(W))

    a = ufl.inner(ufl.grad(ufl.TrialFunction(V_state)), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    L_adj = (u - d) * v * ufl.dx

    state_problem = dolfinx.fem.petsc.LinearProblem(a, L, [bc], u)
    p = dolfinx.fem.Function(V_state, name="p")
    adjoint_problem = dolfinx.fem.petsc.LinearProblem(a, L_adj, [bc], p)

    @dolfiny.taoblockproblem.link_state(f)
    def F_reduced(tao, x):
        state_problem.solve()

        local_J = dolfinx.fem.assemble_scalar(F)
        return MPI.COMM_WORLD.allreduce(local_J, op=MPI.SUM)

    # JF = dolfinx.fem.form(ufl.derivative(ufl.action(a-L, p), f) + ufl.derivative(F, f))
    JF = p * ufl.TestFunction(V_control) * ufl.dx + alpha * f * ufl.TestFunction(V_control) * ufl.dx
    JF = dolfinx.fem.form(JF)

    @dolfiny.taoblockproblem.link_state(f)
    def J_reduced(tao, x, J):
        state_problem.solve()

        adjoint_problem.solve()

        J.zeroEntries()
        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        dolfinx.fem.petsc.assemble_vector(J, JF)
        J.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    opts = PETSc.Options("optimal_control_reduced")
    opts["tao_type"] = "lmvm"  # pdipm
    opts["tao_gatol"] = 1e-9
    # opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F_reduced, [f], J=(J_reduced, f.x.petsc_vec.copy()), prefix="optimal_control_reduced"
    )
    opt_problem.solve([f])

    x = ufl.SpatialCoordinate(mesh)
    f_ana = 1 / (1 + 4 * alpha * np.pi**4) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    u_ana = 1 / (2 * np.pi**2) * f_ana

    assert np.allclose(0, L2_norm(f - f_ana), atol=5e-2)
    assert np.allclose(0, L2_norm(u - u_ana), atol=1e-4)
    assert opt_problem.tao.getConvergedReason() > 0
    # with dolfinx.io.VTXWriter(V_state.mesh.comm, "opt.bp", u, "bp4") as file:
    #     file.write(0.0)

    # with dolfinx.io.VTXWriter(V_control.mesh.comm, "opt_f.bp", f, "bp4") as file:
    #     file.write(0.0)


# TODO: autodiff


# TODO: very unstable at the moment + huge number of iterations necessary -> fine tune
def test_poisson_pde_as_constraint():
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
    # u.x.petsc_vec.set(1.0)
    # bc.set(u.x.array)
    f = -1

    a = ufl.inner(ufl.grad(u), ufl.grad(ufl.TestFunction(V))) * ufl.dx
    # a = u*ufl.TestFunction(V) * ufl.dx
    L = f * ufl.TestFunction(V) * ufl.dx

    g = [a - L == 0]
    Jg = [[ufl.derivative(a - L, u, ufl.TrialFunction(V))]]
    # Jg = [[ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx]]
    # F = .5 * u**2 * ufl.dx
    # F = .5 * (u - f) * ufl.dx
    # F = dolfinx.fem.Constant(mesh, 1.0) * ufl.dx
    F = 0.5 * dolfinx.fem.Constant(mesh, 0.0) * (u) ** 2 * ufl.dx
    opts = PETSc.Options("poisson_pde_constraint")
    opts["tao_type"] = "almm"  # pdipm
    # opts["tao_max_it"] = 2
    opts["tao_gatol"] = 1e-4
    opts["tao_grtol"] = 1e-4
    opts["tao_catol"] = 1e-4
    opts["tao_crtol"] = 1e-4
    # opts["tao_almm_subsolver_tao_test_gradient"] = ""
    # opts["tao_almm_subsolver_tao_type"] = "blmvm"
    opts["tao_almm_subsolver_tao_gatol"] = 1e-3
    opts["tao_almm_subsolver_tao_grtol"] = 1e-3
    # opts["tao_almm_subsolve_tao_blmvm_mat_lmvm_J0_ksp_type"] = "preonly"
    # opts["tao_almm_subsolve_tao_blmvm_mat_lmvm_J0_pc_type"] = "mumps"
    # opts["tao_almm_subsolver_view"] = ""
    # opts["tao_almm_subsolver_tao_ls_type"] = "unit"
    opts["tao_almm_mu_init"] = 2  # penalty
    opts["tao_almm_mu_factor"] = 2
    opts["tao_almm_subsolver_ksp_type"] = "preonly"
    opts["tao_almm_subsolver_pc_type"] = "lu"
    opts["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"
    # opts["tao_test_gradient"] = ""
    # opts["tao_almm_subsolver_tao_test_gradient"] = ""
    # opts["tao_monitor"] = ""
    # opts["tao_ls_monitor"] = ""

    opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
        F, [u], bcs=[bc], g=g, Jg=Jg, prefix="poisson_pde_constraint"
    )

    # See https://petsc.org/main/src/tao/tutorials/ex3.c.html
    # m = ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx
    # m = dolfinx.fem.form(m)
    # M = dolfinx.fem.petsc.assemble_matrix(m)
    # M.assemble()
    # problem.tao.getALMMSubsolver().setLMVMH0(M)
    # problem.tao.getALMMSubsolver().setGradientNorm(M)
    # problem.tao.setGradientNorm(M)

    (u_opt,) = opt_problem.solve([u])

    linear_problem = dolfinx.fem.petsc.LinearProblem(
        ufl.inner(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx, L, [bc]
    )
    u_lp = linear_problem.solve()

    assert np.allclose(L2_norm(u_lp - u_opt), 0, atol=1e-3)
    assert opt_problem.tao.getConvergedReason() > 0
    # with dolfinx.io.VTXWriter(V.mesh.comm, "opt.bp", u, "bp4") as file:
    #     file.write(0.0)

    # with dolfinx.io.VTXWriter(V_control.mesh.comm, "opt_f.bp", f, "bp4") as file:
    #     file.write(0.0)


# TODO: never moving away from initial condition - why?
# def test_poisson_pde_as_constraint_discrete():
#     n = 32
#     mesh = dolfinx.mesh.create_unit_square(
#         MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
#     )

#     V = dolfinx.fem.functionspace(mesh, ("P", 1))

#     top = V.mesh.topology
#     top.create_connectivity(top.dim - 1, top.dim)
#     boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
#     boundary_dofs = dolfinx.fem.locate_dofs_topological(V, top.dim - 1, boundary_facets)
#     bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(V.mesh, 0.0), boundary_dofs, V)

#     f = -1

#     a = ufl.inner(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx
#     a = dolfinx.fem.form(a)
#     L = f * ufl.TestFunction(V) * ufl.dx
#     L = dolfinx.fem.form(L)

#     A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc], diag=1.0)
#     A.assemble()

#     m = ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx
#     m = dolfinx.fem.form(m)
#     M = dolfinx.fem.petsc.assemble_matrix(m)
#     M.assemble()
#     # problem.tao.getALMMSubsolver().setLMVMH0(M)
#     # problem.tao.getALMMSubsolver().setGradientNorm(M)
#     # problem.tao.setGradientNorm(M)

#     R = PETSc.KSP().create(MPI.COMM_WORLD)
#     R.setOperators(M)
#     R.setType('preonly')
#     R.getPC().setType('lu')
#     R.setUp()

#     b = dolfinx.fem.petsc.assemble_vector(L)
#     dolfinx.fem.petsc.apply_lifting(b, [a], [bc])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     dolfinx.fem.petsc.set_bc(b, [bc])

#     def F(tao, x) -> float:
#         return x.norm()**2

#     def J(tao, x, J_vec) -> None:
#         J_vec.zeroEntries()
#         J_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#         x.copy(J_vec)
#         J_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#         # print(f"J-norm: {J_vec.norm()}")

#     def g(tao, x, c) -> None:
#         # tmp = Ax - b
#         tmp = x.copy()
#         tmp.assemble()
#         A.mult(x, tmp)
#         tmp.aypx(-1, b)

#         # c = x^T (Ax - b)
#         c[0] = x.dot(tmp)
#         c.assemble()

#     ksp = PETSc.KSP().create(MPI.COMM_WORLD)
#     ksp.setType("preonly")
#     ksp.getPC().setType("lu")
#     # See https://petsc.org/main/src/tao/tutorials/ex3.c.html
#     m = ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx
#     m = dolfinx.fem.form(m)
#     M = dolfinx.fem.petsc.assemble_matrix(m)
#     M.assemble()
#     # problem.tao.getALMMSubsolver().setLMVMH0(M)
#     # problem.tao.getALMMSubsolver().setGradientNorm(M)
#     # problem.tao.setGradientNorm(M)
#     ksp.setOperators(M, None)
#     ksp.setUp()
#     u = dolfinx.fem.Function(V, name="u")

#     Jg_vec = u.x.petsc_vec.copy()
#     Jg_vec.assemble()
#     Jg_mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)  # type: ignore
#     # TODO: should all k rows belong to fist process? implications?
#     Jg_mat.setSizes(
#         [[u.x.petsc_vec.getLocalSize(), u.x.petsc_vec.getSize()], [PETSc.DECIDE, 1]]
#   # type: ignore
#     )  # [[nrl, nrg], [ncl, ncg]]
#     Jg_mat.setType("dense")
#     Jg_mat.setUp()
#     Jg_mat_T = PETSc.Mat()  # type: ignore
#     Jg_mat_T.createTranspose(Jg_mat)
#     Jg_mat = Jg_mat_T  # TODO: document
#     def Jg(tao, x, J, P) -> None:
#         Jg_vec.zeroEntries()
#         Jg_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#         A.mult(x, Jg_vec)
#         Jg_vec.axpy(-1, b)
#         Jg_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#         ksp.solve(Jg_vec, Jg_vec)

#         offset = Jg_vec.getOwnershipRange()[0]
#         JT = J.getTransposeMat()
#         JT.zeroEntries()
#         for i in range(Jg_vec.getLocalSize()):
#             JT.setValue(offset + i, 0, Jg_vec.getArray()[i])
#         JT.assemble()
#         JT.createTranspose(J)
#         J.assemble()

#     # g = [a - L == 0]
#     # Jg = [[ufl.derivative(a - L, u, ufl.TrialFunction(V))]]
#     # Jg = [[ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx]]
#     # F = .5 * u**2 * ufl.dx
#     # F = .5 * (u - f) * ufl.dx
#     # F = dolfinx.fem.Constant(mesh, 1.0) * ufl.dx
#     F = 0.5 * dolfinx.fem.Constant(mesh, 0.0) * (u) ** 2 * ufl.dx
#     opts = PETSc.Options("poisson")
#     opts["tao_type"] = "almm"  # pdipm
#     opts["tao_almm_type"] = "classic"
#     opts["tao_max_it"] = 20
#     # opts["tao_gatol"] = 1e-7
#     # opts["tao_almm_subsolver_tao_test_gradient"] = ""
#     # opts["tao_almm_subsolver_tao_type"] = "bqnls"
#     # opts["tao_almm_subsolver_tao_ls_type"] = "armijo"
#     # opts["tao_almm_subsolver_ksp_type"] = "preonly"
#     # opts["tao_almm_subsolver_pc_type"] = "lu"
#     # opts["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"
#     # opts["tao_catol"] = 1e-7
#     # opts["tao_test_gradient"] = ""
#     opts["tao_monitor"] = ""
#     opts["tao_ls_monitor"] = ""


#     # F = u**2 * ufl.dx
#     g_vec = PETSc.Vec().createMPI(  # type: ignore
#         [1 if MPI.COMM_WORLD.rank == 0 else 0, 1], comm=MPI.COMM_WORLD
#     )
#     g_vec.setUp()
#     problem = dolfiny.taoblockproblem.TAOBlockProblem(
#         # J=(J, u.x.petsc_vec.copy()),
#         F, [u],  bcs=[bc], g=(g, g_vec), Jg=(Jg, Jg_mat), prefix="poisson"
#     )
#     problem.tao.setGradientNorm(M)

#     # u.x.petsc_vec.set(1.0)
#     with u.x.petsc_vec.localForm() as local_u:
#         local_u.set(1.0)

#     (u_opt,) = problem.solve([u])

#     linear_problem = dolfinx.fem.petsc.LinearProblem(
#         ufl.inner(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx, L, [bc]
#     )
#     u_lp = linear_problem.solve()

#     # assert np.allclose(L2_norm(u_lp - u_opt), 0, atol=5e-4)

#     with dolfinx.io.VTXWriter(V.mesh.comm, "opt.bp", u_opt, "bp4") as file:
#         file.write(0.0)

#     # with dolfinx.io.VTXWriter(V_control.mesh.comm, "opt_f.bp", f, "bp4") as file:
#     #     file.write(0.0)

# TODO: blocksizes seem to be none consistent, objective produces vectors with block sizes and
# constraint not.
# def test_poisson_pde_constraint_discrete():
#     n = 32
#     mesh = dolfinx.mesh.create_unit_square(
#         MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
#     )
#     W = dolfinx.fem.functionspace(mesh, ("CG", 1))

#     f = -1.0

#     top = W.mesh.topology
#     top.create_connectivity(top.dim - 1, top.dim)
#     boundary_facets = dolfinx.mesh.exterior_facet_indices(top)
#     boundary_dofs = dolfinx.fem.locate_dofs_topological(W, top.dim - 1, boundary_facets)
#     bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(W.mesh, 0.0), boundary_dofs, W)

#     u, v = dolfinx.fem.Function(W), ufl.TestFunction(W)
#     a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
#     L = -ufl.inner(f, v) * ufl.dx

#     g = [a - L == 0]
#     Jg = [[ufl.derivative(g[0].lhs, u, ufl.TrialFunction(W))]]

#     # a = dolfinx.fem.form(a)
#     # L = dolfinx.fem.form(L)

#     # A = dolfinx.fem.petsc.assemble_matrix(a, [bc])
#     # A.assemble()
#     # b = dolfinx.fem.petsc.assemble_vector(L)
#     # dolfinx.fem.petsc.apply_lifting(b, [a], [bc])
#     # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     # dolfinx.fem.petsc.set_bc(b, [bc])

#     # tmp = b.copy()
#     # tmp.zeroEntries()
#     # b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#     # def zero_F(tao: PETSc.TAO: )

#     def F(tao: PETSc.TAO, x: PETSc.Vec) -> float:
#         # A.mult(x, tmp)
#         # return 0.5 * tmp.dot(x) - b.dot(x)
#         return 0

#     def J(tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Vec) -> None:
#         pass
#         # J.zeroEntries()
#         # J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#         # A.mult(x, J)
#         # J.axpy(-1, b)

#         # J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#     def H(tao: PETSc.TAO, x: PETSc.Vec, H: PETSc.Mat, P: PETSc.Mat) -> None:
#         pass

#     opts = PETSc.Options("poisson_discrete")
#     opts["tao_type"] = "almm"
#     opts["tao_gatol"] = 1e-3
#     opts["tao_grtol"] = 1e-3
#     opts["tao_gttol"] = 1e-3
#     # opts["tao_nls_ksp_type"] = "preonly"
#     # opts["tao_nls_pc_type"] = "cholesky"
#     # opts["tao_nls_pc_factor_mat_solver_type"] = "mumps"
#     opts["tao_monitor"] = ""
#     # opts["tao_ls_monitor"] = ""

#     u = dolfinx.fem.Function(W)

#     opt_problem = dolfiny.taoblockproblem.TAOBlockProblem(
#         F, [u], bcs=[bc], J=(J, u.x.petsc_vec.copy()), g=g, Jg=Jg, prefix="poisson_discrete"
#     )
#     (sol_optimization,) = opt_problem.solve([u])

#     problem = dolfinx.fem.petsc.LinearProblem(
#         a,
#         L,
#         u=dolfinx.fem.Function(W),
#         bcs=[bc],
#         petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
#     )
#     sol_weak_form = problem.solve([u])

#     # TODO:
#     # assert opt_problem.tao.getConvergedReason() == PETSc.TAO.ConvergedReason.CONVERGED_GATOL

#     assert L2_norm(sol_optimization - sol_weak_form) == pytest.approx(0, abs=atol)

# test_poisson_constrained(None, True)
