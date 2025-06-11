from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl

import numpy as np

import dolfiny


def test_neohooke():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 7, 7, 7)
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (mesh.geometry.dim,)))
    P = dolfinx.fem.functionspace(mesh, ("P", 1))

    L = dolfinx.fem.functionspace(mesh, ("DP", 0))

    u = dolfinx.fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    p = dolfinx.fem.Function(P, name="p")
    q = ufl.TestFunction(P)

    lmbda0 = dolfinx.fem.Function(L)

    d = mesh.topology.dim
    Id = ufl.Identity(d)
    F = Id + ufl.grad(u)
    C = F.T * F
    J = ufl.det(F)

    E_, nu_ = 10.0, 0.3
    mu, lmbda = E_ / (2 * (1 + nu_)), E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
    psi = (mu / 2) * (ufl.tr(C) - 3) - mu * ufl.ln(J) + lmbda / 2 * ufl.ln(J) ** 2 + (p - 1.0) ** 2
    pi = psi * ufl.dx

    F0 = ufl.derivative(pi, u, v)
    F1 = ufl.derivative(pi, p, q)

    # Number of eigenvalues to find
    nev = 8

    opts = PETSc.Options("neohooke")
    opts["eps_smallest_magnitude"] = True
    opts["eps_nev"] = nev
    opts["eps_ncv"] = 50 * nev
    opts["eps_conv_abs"] = True

    # opts["eps_non_hermitian"] = True
    opts["eps_tol"] = 1.0e-14
    opts["eps_max_it"] = 1000
    opts["eps_error_relative"] = "ascii::ascii_info_detail"
    opts["eps_monitor"] = "ascii"

    slepcp = dolfiny.slepcproblem.SLEPcProblem([F0, F1], [u, p], lmbda0, prefix="neohooke")
    slepcp.solve()

    # mat = dolfiny.la.petsc_to_scipy(slepcp.eps.getOperators()[0])
    # eigvals, eigvecs = linalg.eigsh(mat, which="SM", k=nev)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "eigvec.xdmf", "w") as ofile:
        ofile.write_mesh(mesh)
        for i in range(nev):
            eigval, ur, ui = slepcp.getEigenpair(i)

            # Expect first 6 eignevalues 0, i.e. rigid body modes
            if i < 6:
                assert np.isclose(eigval, 0.0)

            for func in ur:
                name = func.name
                func.name = f"{name}_eigvec_{i}_real"
                ofile.write_function(func)

                func.name = name
