from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny

eps = np.finfo(np.float64).eps
tol = 100 * eps


@pytest.mark.parametrize("mode", ["worst", "exact"])
def test_scalar(mode: str) -> None:
    """Assembling the area of a rectangle mesh; error bound must be < 100 eps (relative)."""

    w, h = 2.7, 1.3
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([w, h])],
        [8, 5],
        dolfinx.mesh.CellType.triangle,
    )
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    f = dolfinx.fem.Function(V)
    f.x.array[:] = 1.0

    compiled_form = dolfiny.fem.form(f * ufl.dx(domain=mesh), mode=mode)
    val = mesh.comm.allreduce(dolfiny.fem.assemble_scalar(compiled_form), op=MPI.SUM)

    assert np.isclose(val.real, w * h), f"Expected area {w * h}, got {val.real}"
    assert abs(val.imag) < tol * abs(val.real), (
        f"Error bound {val.imag} exceeds tol * |value| = {tol * abs(val.real)}"
    )


@pytest.mark.parametrize("mode", ["worst", "exact"])
def test_vector(mode: str) -> None:
    """Assembling a linear form with a constant coefficient;
    error bounds must be < 100 eps (relative)."""

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 4], dolfinx.mesh.CellType.triangle
    )
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (2,)))
    f = dolfinx.fem.Function(V)
    f.x.array[:] = 1.0

    v = ufl.TestFunction(V)
    compiled_form = dolfiny.fem.form(ufl.inner(f, v) * ufl.dx, mode=mode)
    b = dolfiny.fem.assemble_vector(compiled_form)

    assert b.dtype == np.complex128
    assert len(b) == (
        (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs
    )

    nonzero = np.abs(b.real) > 0
    rel_err = np.max(np.abs(b.imag[nonzero]) / np.abs(b.real[nonzero]))
    assert np.all(np.abs(b.imag[nonzero]) < tol * np.abs(b.real[nonzero])), (
        f"Max relative error {rel_err} exceeds tol = {tol}"
    )


@pytest.mark.parametrize("mode", ["worst", "exact"])
def test_matrix(mode: str) -> None:
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 4], dolfinx.mesh.CellType.triangle
    )
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (2,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    compiled_form = dolfiny.fem.form(ufl.inner(u, v) * ufl.dx, mode=mode)

    A = dolfiny.fem.assemble_matrix(compiled_form)
    for i in range(2):
        assert A.index_map(i).size_global == V.dofmap.index_map.size_global
        assert A.index_map(i).size_local == V.dofmap.index_map.size_local
        assert A.bs == [V.dofmap.index_map_bs] * 2

    assert A.data.dtype == np.complex128

    nonzero = np.abs(A.data) > 0
    rel_err = np.max(np.abs(A.data.imag[nonzero]) / np.abs(A.data.real[nonzero]))
    assert np.all(np.abs(A.data.imag[nonzero]) < tol * np.abs(A.data.real[nonzero])), (
        f"Max relative error {rel_err} exceeds tol = {tol}"
    )


def test_scalar_cancellation():
    """Catastrophic cancellation: ((u - C) + C)*dx with u=1/3, C=1e16.
    Subtracting then re-adding C destroys all significant digits of u, so the
    running error bound must exceed tol * (1/3) by many orders of magnitude."""

    C = 1e16
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1 / 3

    compiled_form = dolfiny.fem.form(((u - C) + C) * ufl.dx(domain=mesh), mode="exact")
    val = mesh.comm.allreduce(dolfiny.fem.assemble_scalar(compiled_form), op=MPI.SUM)

    # exact_error follows the (computed - true) convention. Here the computed result is 0
    # (all significant digits of u are destroyed), so the signed error is ~ -1/3.
    assert val.imag < -tol * (1 / 3), (
        f"Error {val.imag} should be below -tol * (1/3) = {-tol * (1 / 3)}"
    )
