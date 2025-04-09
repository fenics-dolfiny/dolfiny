from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl

import numpy as np

import dolfiny
import dolfiny.taoblockproblem


def L2_norm(u, comm) -> float:
    form = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)
    local_squared = dolfinx.fem.assemble_scalar(form)
    global_squared = comm.allreduce(local_squared, op=MPI.SUM)
    return np.sqrt(global_squared)


def test_poisson(V1):
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

    problem = dolfiny.taoblockproblem.TAOBlockProblem(
        [F], [u], bcs=[bc], J=[J], H=[H], prefix="poisson"
    )
    (sol_optimization,) = problem.solve()

    weak_form = ufl.replace(J, {u: ufl.TrialFunction(W)})
    a, L = ufl.lhs(weak_form), ufl.rhs(weak_form)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    sol_weak_form = problem.solve()

    norm = L2_norm(sol_optimization - sol_weak_form, MPI.COMM_WORLD)

    assert np.allclose(norm, 0)

    # with dolfinx.io.VTXWriter(W.mesh.comm, "opt.bp", u, "bp4") as file:
    #     file.write(0.0)
    # with dolfinx.io.VTXWriter(W.mesh.comm, "direct.bp", sol_weak_form, "bp4") as file:
    #     file.write(0.0)
