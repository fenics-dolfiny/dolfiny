#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl

import mesh_notched
import numpy as np

import dolfiny

name = "notched"
gmsh_model, tdim = mesh_notched.mesh_notched(name, clscale=0.2)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)

# Write mesh and meshtags to file
with dolfiny.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh, mts)

top_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, 1, lambda x: np.logical_and(np.isclose(x[0], 0.0), np.greater_equal(x[1], 0.5))
)
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 0.0))

quad_degree = 8
Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(2,))
Qe = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=quad_degree)
Pe = basix.ufl.blocked_element(Qe, shape=(2, 2), symmetry=True)

Uf = dolfinx.fem.functionspace(mesh, Ue)
Pf = dolfinx.fem.functionspace(mesh, Pe)
Lf = dolfinx.fem.functionspace(mesh, Qe)

u = dolfinx.fem.Function(Uf, name="u")  # displacement
dP = dolfinx.fem.Function(Pf, name="dP")  # increment of plastic strain, symmetric
dl = dolfinx.fem.Function(Lf, name="dl")  # increment of plastic multiplier

P0 = dolfinx.fem.Function(Pf, name="P0")  # total plastic strain, symmetric
l0 = dolfinx.fem.Function(Lf, name="l0")  # total plastic multiplier

δu = ufl.TestFunction(Uf)
δdP = ufl.TestFunction(Pf)
δdl = ufl.TestFunction(Lf)

# for output
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (2,))), name="u")


def f(sigma):
    (_, s_max), _ = dolfiny.invariants.eigenstate2(sigma)
    return s_max - Sy


# Strain measures
E = ufl.sym(ufl.grad(u))  # linearised total strain
E_el = E - (P0 + dP)  # E_el = E - P, elastic strain

mu = 100
la = 10
Sy = 0.3
sigma = 2 * mu * E_el + la * ufl.tr(E_el) * ufl.Identity(2)

sigma = ufl.variable(sigma)

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})
F0 = ufl.inner(sigma, ufl.sym(ufl.grad(δu))) * dx  # Global momentum equilibrium
F1 = ufl.inner(dP - dl * ufl.diff(f(sigma), sigma), δdP) * dx  # Plastic flow rule
F2 = ufl.inner(ufl.min_value(dl, -f(sigma)), δdl) * dx

# Alternative KKT complementarity functions are:
# (Note: changes in these lead to changes in coefficient dofs ordering, needs
# adjusting the kernels)
#
# F2 = ufl.inner(cn * dl - ufl.Max(0.0, cn*dl + f(sigma)), δdl) * dx
# F2 = ufl.inner(ufl.min_value(dl, -f(sigma)), δdl) * dx
#
# with
# young = 4 * mu * (la + mu) / (la + 2 * mu)
# cn = 1 / young


u_top = dolfinx.fem.Function(Uf, name="u_top")
u_bottom = dolfinx.fem.Function(Uf, name="u_bottom")

bcs = [
    dolfinx.fem.dirichletbc(u_top, dolfinx.fem.locate_dofs_topological(Uf, 1, top_facets)),
    dolfinx.fem.dirichletbc(u_bottom, dolfinx.fem.locate_dofs_topological(Uf, 1, bottom_facets)),
]

sc_J = dolfiny.localsolver.UserKernel(
    name="sc_J",
    code=r"""
    template <typename T>
    void sc_J(T& A)
    {
        Eigen::MatrixXd Jllrow0(J11.array.rows(), J11.array.cols() + J12.array.cols());
        Jllrow0 << J11.array, J12.array;

        Eigen::MatrixXd Jllrow1(J21.array.rows(), J21.array.cols() + J22.array.cols());
        Jllrow1 << J21.array, J22.array;

        Eigen::MatrixXd Jll(J11.array.rows() + J21.array.rows(),
                            J11.array.cols() + J22.array.cols());
        Jll << Jllrow0,
               Jllrow1;

        Eigen::MatrixXd Jlg(J10.array.rows() + J20.array.rows(), J10.array.cols());
        Jlg << J10.array,
               J20.array;

        auto J02 = Eigen::MatrixXd::Zero(J00.array.rows(), J12.array.cols());

        Eigen::MatrixXd Jgl(J01.array.rows(), J01.array.cols() + J02.cols());
        Jgl << J01.array, J02;

        A = J00.array - Jgl * Jll.partialPivLu().solve(Jlg);
    }
    """,
    required_J=[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
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
    auto dP = Eigen::Map<Eigen::Matrix<double, 48, 1>>(&F1.w[12]);
    auto dl = Eigen::Map<Eigen::Matrix<double, 16, 1>>(&F1.w[60]);

    Eigen::Matrix<double, 48+16, 1> loc = Eigen::Matrix<double, 48+16, 1>::Zero();
    loc << dP,
           dl;
    Eigen::Matrix<double, 48+16, 1> dloc = Eigen::Matrix<double, 48+16, 1>::Zero();

    Eigen::MatrixXd Jllrow0(J11.array.rows(), J11.array.cols() + J12.array.cols());
    Eigen::MatrixXd Jllrow1(J21.array.rows(), J21.array.cols() + J22.array.cols());
    Eigen::MatrixXd Jll(J11.array.rows() + J21.array.rows(), J11.array.cols() + J22.array.cols());

    Eigen::Matrix<double, 48+16, 1> R = Eigen::Matrix<double, 48+16, 1>::Zero();

    const int N = 20;
    for (int i = 0; i < N; ++i)
    {
        F1.array.setZero();
        F1.kernel(F1.array.data(), F1.w.data(), F1.c.data(),
                    F1.coords.data(), F1.entity_local_index.data(),
                    F1.permutation.data(), nullptr);

        F2.array.setZero();
        F2.kernel(F2.array.data(), F2.w.data(), F2.c.data(),
                    F2.coords.data(), F2.entity_local_index.data(),
                    F2.permutation.data(), nullptr);

        R << F1.array,
             F2.array;
        double err = R.norm();
        double err0 = 0.0;
        if ((err < 1e-9 * err0) || (err < 1e-16))
            break;

        if (i > (N - 5))
            std::cout << "it=" << i << " error = " << err << std::endl;

        if (i == (N - 1))
            throw std::runtime_error("Failed to converge locally.");

        J11.array.setZero();
        J11.kernel(J11.array.data(), J11.w.data(), J11.c.data(),
                   J11.coords.data(), J11.entity_local_index.data(),
                   J11.permutation.data(), nullptr);

        J12.array.setZero();
        J12.kernel(J12.array.data(), J12.w.data(), J12.c.data(),
                   J12.coords.data(), J12.entity_local_index.data(),
                   J12.permutation.data(), nullptr);

        J21.array.setZero();
        J21.kernel(J21.array.data(), J21.w.data(), J21.c.data(),
                   J21.coords.data(), J21.entity_local_index.data(),
                   J21.permutation.data(), nullptr);

        J22.array.setZero();
        J22.kernel(J22.array.data(), J22.w.data(), J22.c.data(),
                   J22.coords.data(), J22.entity_local_index.data(),
                   J22.permutation.data(), nullptr);

        Jllrow0 << J11.array, J12.array;
        Jllrow1 << J21.array, J22.array;
        Jll << Jllrow0,
               Jllrow1;

        dloc = Jll.partialPivLu().solve(R);
        loc -= dloc;

        auto dP = loc(Eigen::seq(0, 47));
        auto dl = loc(Eigen::seq(48, 63));

        F1.w(Eigen::seq(12, 59)) = dP;
        F1.w(Eigen::seq(60, 75)) = dl;

        F2.w(Eigen::seq(12, 59)) = dP;
        F2.w(Eigen::seq(60, 75)) = dl;

        J11.w(Eigen::seq(12, 59)) = dP;
        J11.w(Eigen::seq(60, 75)) = dl;

        J21.w(Eigen::seq(12, 59)) = dP;
        J21.w(Eigen::seq(60, 75)) = dl;

        J22.w(Eigen::seq(12, 59)) = dP;
        J22.w(Eigen::seq(60, 75)) = dl;

        J12.w(Eigen::seq(12, 59)) = dP;
    }
"""

solve_dP = dolfiny.localsolver.UserKernel(
    name="solve_dP",
    code=f"""
    template <typename T>
    void solve_dP(T& A)
    {{
        {solve_body}
        A = dP;
    }}
    """,
    required_J=[(1, 1), (1, 2), (2, 1), (2, 2)],
)

solve_dl = dolfiny.localsolver.UserKernel(
    name="solve_dl",
    code=f"""
    template <typename T>
    void solve_dl(T& A)
    {{
        {solve_body}
        A = dl;
    }}
    """,
    required_J=[(1, 1), (1, 2), (2, 1), (2, 2)],
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


cells = dict([(-1, np.arange(mesh.topology.index_map(mesh.topology.dim).size_local))])

ls = dolfiny.localsolver.LocalSolver(
    [Uf, Pf, Lf],
    local_spaces_id=[1, 2],
    F_integrals=[{dolfinx.fem.IntegralType.cell: [(-1, sc_F_cell, cells[-1])]}],
    J_integrals=[[{dolfinx.fem.IntegralType.cell: [(-1, sc_J, cells[-1])]}]],
    local_integrals=[
        {dolfinx.fem.IntegralType.cell: [(-1, solve_dP, cells[-1])]},
        {dolfinx.fem.IntegralType.cell: [(-1, solve_dl, cells[-1])]},
    ],
    local_update=local_update,
)


opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-8
opts["snes_max_it"] = 20
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

problem = dolfiny.snesproblem.SNESProblem(
    [F0, F1, F2], [u, dP, dl], bcs=bcs, prefix=name, localsolver=ls
)

ls.view()

K = 25
du1 = 0.005
load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
cycle = np.concatenate((load, unload))

for step, factor in enumerate(cycle):
    dolfiny.utils.pprint(f"\n+++ Processing step {step:3d}, load factor = {factor:5.4f}")

    # Update values for given boundary displacement
    u_top.interpolate(lambda x: (np.zeros_like(x[0]), du1 * factor * np.ones_like(x[1])))

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Accumulate plastic states from increments
    for source, target in zip([dP, dl], [P0, l0]):
        with source.x.petsc_vec.localForm() as locs, target.x.petsc_vec.localForm() as loct:
            loct.axpy(1.0, locs)

    # Interpolate and write output
    dolfiny.interpolation.interpolate(u, uo)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "a") as ofile:
        ofile.write_function(uo, step)
