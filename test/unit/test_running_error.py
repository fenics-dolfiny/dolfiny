from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_packed_layout(dtype: type[np.floating]) -> None:
    """The packed pair is 2*sizeof(T), and the opaque carrier must match it exactly."""
    itemsize = np.dtype(dtype).itemsize
    packed = dolfiny.fem._packed_dtype(dtype)
    carrier = np.dtype(dolfiny.fem.nptype_to_carrier[dtype])

    assert packed.itemsize == 2 * itemsize
    assert packed.fields is not None
    assert packed.fields["val"][1] == 0
    assert packed.fields["err"][1] == itemsize
    assert carrier.itemsize == packed.itemsize, (
        f"carrier {carrier.name} does not match the {packed.itemsize}-byte packed pair"
    )
    # A carrier array must be viewable as the packed layout without a copy.
    view = np.zeros(4, dtype=carrier).view(packed)
    assert view["val"].dtype == dtype and view["err"].dtype == dtype


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
@pytest.mark.parametrize("mode", ["worst", "signed"])
def test_scalar(mode: str, dtype: np.dtype) -> None:
    """Assembling the area of a rectangle mesh; error bound must be < 100 eps (relative)."""

    eps = np.finfo(dtype).eps
    tol = 100 * eps

    w, h = 2.7, 1.3
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([w, h])],
        [8, 5],
        dolfinx.mesh.CellType.triangle,
        dtype=np.float64,
    )
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    f = dolfinx.fem.Function(V, dtype=dtype)
    f.x.array[:] = 1.0
    c = dolfinx.fem.Constant(mesh, 1.0)

    compiled_form = dolfiny.fem.form(c * f**2 * ufl.dx(domain=mesh), mode=mode, dtype=dtype)
    val, err = dolfiny.fem.assemble_scalar(compiled_form)
    val = mesh.comm.allreduce(val, op=MPI.SUM)
    err = mesh.comm.allreduce(err, op=MPI.SUM)

    assert np.isclose(val, w * h, rtol=tol, atol=0), f"Expected area {w * h}, got {val}"
    assert abs(err) < tol * abs(val), f"Error bound {err} exceeds tol * |value| = {tol * abs(val)}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
@pytest.mark.parametrize("mode", ["worst", "signed"])
def test_vector(mode: str, dtype: np.dtype) -> None:
    """Assembling a linear form with a constant coefficient;
    error bounds must be < 100 eps (relative)."""

    eps = np.finfo(dtype).eps
    tol = 100 * eps

    # There is no fp16 mesh geometry; carry fp16 dofs on an fp64 mesh.
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [1.0, 1.0]],
        [4, 4],
        dolfinx.mesh.CellType.triangle,
        dtype=np.float64,
    )
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (2,)))
    f = dolfinx.fem.Function(V, dtype=dtype)
    f.x.array[:] = 1.0
    c = dolfinx.fem.Constant(mesh, 1.0)

    v = ufl.TestFunction(V)
    compiled_form = dolfiny.fem.form(c * ufl.inner(f, v) * ufl.dx, mode=mode, dtype=dtype)
    val, err = dolfiny.fem.assemble_vector(compiled_form)

    assert val.dtype == dtype
    assert err.dtype == dtype  # error is carried in the value type
    assert len(val) == (
        (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs
    )

    nonzero = np.abs(val) > 0
    rel_err = np.max(np.abs(err[nonzero]) / np.abs(val[nonzero]))
    assert np.all(np.abs(err[nonzero]) < tol * np.abs(val[nonzero])), (
        f"Max relative error {rel_err} exceeds tol = {tol}"
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
@pytest.mark.parametrize("mode", ["worst", "signed"])
def test_matrix(mode: str, dtype: np.dtype) -> None:

    eps = np.finfo(dtype).eps
    tol = 100 * eps

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [1.0, 1.0]],
        [4, 4],
        dolfinx.mesh.CellType.triangle,
        dtype=np.float64,
    )
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (2,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    compiled_form = dolfiny.fem.form(ufl.inner(u, v) * ufl.dx, mode=mode, dtype=dtype)

    A_val, A_err = dolfiny.fem.assemble_matrix(compiled_form)
    for A in (A_val, A_err):
        for i in range(2):
            assert A.index_map(i).size_global == V.dofmap.index_map.size_global
            assert A.index_map(i).size_local == V.dofmap.index_map.size_local
        assert A.bs == [V.dofmap.index_map_bs] * 2
    assert A_val.data.dtype == dtype
    assert A_err.data.dtype == dtype  # error is carried in the value type

    val, err = A_val.data, A_err.data
    nonzero = np.abs(val) > 0
    rel_err = np.max(np.abs(err[nonzero]) / np.abs(val[nonzero]))
    assert np.all(np.abs(err[nonzero]) < tol * np.abs(val[nonzero])), (
        f"Max relative error {rel_err} exceeds tol = {tol}"
    )


def test_scalar_cancellation():
    """Catastrophic cancellation: ((u - C) + C)*dx with u=1/3, C=1e16.
    Subtracting then re-adding C destroys all significant digits of u, so the
    running error bound must exceed tol * (1/3) by many orders of magnitude."""

    eps = np.finfo(np.float64).eps
    tol = 100 * eps

    C = 1e16
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1 / 3

    compiled_form = dolfiny.fem.form(((u - C) + C) * ufl.dx(domain=mesh), mode="signed")
    _val, err = dolfiny.fem.assemble_scalar(compiled_form)
    err = mesh.comm.allreduce(err, op=MPI.SUM)

    # error follows the (computed - true) convention. Here the computed result is 0
    # (all significant digits of u are destroyed), so the signed error is ~ -1/3.
    assert err < -tol * (1 / 3), f"Error {err} should be below -tol * (1/3) = {-tol * (1 / 3)}"


@pytest.mark.parametrize("mode", ["worst", "signed"])
def test_seed_errors(mode: str) -> None:
    """Errors passed at assembly propagate linearly through a linear functional."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    f = dolfinx.fem.Function(V)
    f.x.array[:] = 1.0

    compiled_form = dolfiny.fem.form(f * ufl.dx, mode=mode)
    _, err0 = dolfiny.fem.assemble_scalar(compiled_form)
    _, err1 = dolfiny.fem.assemble_scalar(compiled_form, errors={f: np.full_like(f.x.array, 1e-3)})
    err0, err1 = (mesh.comm.allreduce(e, op=MPI.SUM) for e in (err0, err1))

    assert np.isclose(err1 - err0, 1e-3, rtol=1e-6)


requires_float128 = pytest.mark.skipif(
    not dolfiny.has_float128, reason="mode 'true' needs long double to be binary128"
)


@requires_float128
def test_true_scalar() -> None:
    """The area of the unit square times fl64(1/3) is exact to binary128."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 5)
    c = dolfinx.fem.Constant(mesh, 1 / 3)

    ref = dolfiny.fem.assemble_scalar(dolfiny.fem.form(c * ufl.dx(domain=mesh), mode="true"))
    ref = mesh.comm.allreduce(ref, op=MPI.SUM)

    assert ref.dtype == np.longdouble
    assert abs(ref - np.longdouble(1 / 3)) < 1e-30


@requires_float128
def test_true_cancellation() -> None:
    """Unstable and stable forms agree in binary128, and the "signed" estimate of fp32, seeded
    with the representation error of the inputs, matches the true error."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 5)
    V = dolfinx.fem.functionspace(mesh, ("P", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1 + 1e-4 * (x[0] + x[1] ** 2))
    unstable, stable = (u * u - 1) * ufl.dx, (u - 1) * (u + 1) * ufl.dx

    def assemble(F, mode, **kwargs):
        dtype = None if mode == "true" else np.float32
        value = dolfiny.fem.assemble_scalar(dolfiny.fem.form(F, mode=mode, dtype=dtype), **kwargs)
        if mode == "true":
            return mesh.comm.allreduce(value, op=MPI.SUM)
        return tuple(mesh.comm.allreduce(v, op=MPI.SUM) for v in value)

    ref_unstable, ref_stable = assemble(unstable, "true"), assemble(stable, "true")
    assert abs(ref_unstable - ref_stable) < 1e-25 * abs(ref_stable)

    errors = {u: u.x.array.astype(np.float32).astype(np.float64) - u.x.array}
    val, err = assemble(unstable, "signed", errors=errors)
    assert np.isclose(np.longdouble(val) - ref_stable, err, rtol=0.1)


@requires_float128
def test_true_vector_matrix() -> None:
    """Vector and matrix in binary128 agree with fp64 dolfinx to fp64 accuracy, normwise."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 5)
    V = dolfinx.fem.functionspace(mesh, ("P", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1 + x[0] * x[1])
    v, w = ufl.TestFunction(V), ufl.TrialFunction(V)
    L = ufl.sin(u) * v * ufl.dx
    a = ufl.exp(u) * ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx
    tol = 100 * np.finfo(np.float64).eps

    b = dolfiny.fem.assemble_vector(dolfiny.fem.form(L, mode="true"))
    b64 = dolfinx.fem.assemble_vector(dolfinx.fem.form(L)).array
    assert b.dtype == np.longdouble
    assert np.max(np.abs(b - b64)) < tol * np.max(np.abs(b))

    A = dolfiny.fem.assemble_matrix(dolfiny.fem.form(a, mode="true"))
    data, data64 = A.data.view(np.longdouble), dolfinx.fem.assemble_matrix(dolfinx.fem.form(a)).data
    assert np.max(np.abs(data - data64)) < tol * np.max(np.abs(data))
