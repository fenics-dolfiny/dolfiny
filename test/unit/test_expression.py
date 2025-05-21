import dolfinx
import ufl

import numpy as np

import dolfiny


def test_expression_evaluate(V1, V2, squaremesh_5):
    u1, v1 = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    u2, v2 = ufl.TrialFunction(V2), ufl.TestFunction(V2)

    dx = ufl.dx(squaremesh_5)

    # Test evaluation of expressions

    assert dolfiny.expression.evaluate(1 * u1, u1, u2) == 1 * u2
    assert dolfiny.expression.evaluate(2 * u1 + u2, u1, u2) == 2 * u2 + 1 * u2
    assert dolfiny.expression.evaluate(u1**2 + u2, u1, u2) == u2**2 + 1 * u2

    assert dolfiny.expression.evaluate(u1 + u2, [u1, u2], [v1, v2]) == v1 + v2
    assert dolfiny.expression.evaluate([u1, u2], u1, v1) == [v1, u2]
    assert dolfiny.expression.evaluate([u1, u2], [u1, u2], [v1, v2]) == [v1, v2]

    # Test evaluation of forms

    assert dolfiny.expression.evaluate(1 * u1 * dx, u1, u2) == 1 * u2 * dx
    assert dolfiny.expression.evaluate(2 * u1 * dx + u2 * dx, u1, u2) == 2 * u2 * dx + 1 * u2 * dx
    assert dolfiny.expression.evaluate(u1**2 * dx + u2 * dx, u1, u2) == u2**2 * dx + 1 * u2 * dx

    assert dolfiny.expression.evaluate(u1 * dx + u2 * dx, [u1, u2], [v1, v2]) == v1 * dx + v2 * dx
    assert dolfiny.expression.evaluate([u1 * dx, u2 * dx], u1, v1) == [v1 * dx, u2 * dx]
    assert dolfiny.expression.evaluate([u1 * dx, u2 * dx], [u1, u2], [v1, v2]) == [v1 * dx, v2 * dx]


def test_expression_linearise(V1, V2, squaremesh_5):
    u1, u10 = dolfinx.fem.Function(V1, name="u1"), dolfinx.fem.Function(V1, name="u10")
    u2, u20 = dolfinx.fem.Function(V2, name="u2"), dolfinx.fem.Function(V2, name="u20")

    # Test linearisation of expressions at u0
    assert dolfiny.expression.linearise(1 * u1, u1, u10) == u10 + (u1 + (-1) * u10)
    assert dolfiny.expression.linearise(2 * u1 + u2, u1, u10) == (u2 + 2 * u10) + (
        2 * u1 + (-1) * (2 * u10)
    )

    assert dolfiny.expression.linearise(u1**2 + u2, u1, u10) == (
        u10 * (2 * u1) + (-1) * (u10 * 2 * u10)
    ) + (u10**2 + u2)

    assert dolfiny.expression.linearise(u1**2 + u2**2, [u1, u2], [u10, u20]) == (
        (
            ((u10 * (2 * u1)) + (-1 * (u10 * (2 * u10))))
            + ((u20 * (2 * u2)) + (-1 * (u20 * (2 * u20))))
        )
        + (u10**2 + u20**2)
    )

    assert dolfiny.expression.linearise([u1**2, u2], u1, u10) == [
        (u10 * (2 * u1) + (-1) * (u10 * 2 * u10)) + u10**2,
        u2,
    ]

    assert dolfiny.expression.linearise([u1**2 + u2, u2], [u1, u2], [u10, u20]) == [
        ((u2 + (-1) * u20) + (u10 * (2 * u1) + (-1) * (u10 * (2 * u10)))) + (u20 + u10**2),
        u20 + (u2 + (-1) * u20),
    ]

    dx = ufl.dx(squaremesh_5)

    assert dolfiny.expression.linearise(1 * u1 * dx, u1, u10) == u10 * dx + (
        u1 * dx + (-1) * u10 * dx
    )
    assert dolfiny.expression.linearise([u1**2 * dx, u2 * dx], u1, u10) == [
        u10**2 * dx + u10 * (2 * u1) * dx + (-1) * (u10 * 2 * u10) * dx,
        u2 * dx,
    ]


def test_expression_assemble(V1, vV1, squaremesh_5):
    u1, u2 = dolfinx.fem.Function(V1), dolfinx.fem.Function(vV1)

    dx = ufl.dx(squaremesh_5)

    u1.x.petsc_vec.set(3.0)
    u2.x.petsc_vec.set(2.0)
    u1.x.petsc_vec.ghostUpdate()
    u2.x.petsc_vec.ghostUpdate()

    # check assembled shapes

    assert np.shape(dolfiny.expression.assemble(1.0, dx)) == ()
    assert np.shape(dolfiny.expression.assemble(ufl.grad(u1), dx)) == (2,)
    assert np.shape(dolfiny.expression.assemble(ufl.grad(u2), dx)) == (2, 2)

    # check assembled values

    assert np.isclose(dolfiny.expression.assemble(1.0, dx), 1.0)
    assert np.isclose(dolfiny.expression.assemble(u1, dx), 3.0)
    assert np.isclose(dolfiny.expression.assemble(u2, dx), 2.0).all()
    assert np.isclose(dolfiny.expression.assemble(u1 * u2, dx), 6.0).all()

    assert np.isclose(dolfiny.expression.assemble(ufl.grad(u1), dx), 0.0).all()
    assert np.isclose(dolfiny.expression.assemble(ufl.grad(u2), dx), 0.0).all()


def test_extract_linear_combination(V1, V2, vV1, vV2):
    for U1, U2 in [(V1, V1), (V1, V2), (V2, V2), (vV1, vV1), (vV1, vV2), (vV2, vV2)]:
        u1, u2 = dolfinx.fem.Function(U1), dolfinx.fem.Function(U2)

        for expr in [u1, u2, u1 + 2 * u1, u1 + 2 * u2, u1 / 3 + 3 * u2 / 2]:
            linc = []
            dolfiny.expression.extract_linear_combination(expr, linc)

            assert len(linc) > 0
