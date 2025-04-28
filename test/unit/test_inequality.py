import dolfinx
import ufl
import ufl.constantvalue

import numpy as np
import pytest

from dolfiny.inequality import Inequality

forms = [
    lambda V: dolfinx.fem.Constant(V.mesh, np.float64(1.0)) * ufl.dx,
    lambda V: dolfinx.fem.Function(V) ** 2 * ufl.dx,
    lambda V: ufl.inner(dolfinx.fem.Function(V), ufl.TestFunction(V)) * ufl.dx,
    lambda V: ufl.inner(ufl.TestFunction(V), ufl.TrialFunction(V)) * ufl.dx,
]


@pytest.mark.parametrize("a", forms)
@pytest.mark.parametrize("b", [1.0, 1])
def test_equality(V1, a, b):
    a = a(V1)

    inequality = a == b
    assert isinstance(inequality, ufl.equation.Equation)
    assert inequality.lhs == a
    assert inequality.rhs == b

    inequality = b == a
    assert isinstance(inequality, ufl.equation.Equation)
    assert inequality.lhs == a
    assert inequality.rhs == b


@pytest.mark.parametrize("a", forms)
@pytest.mark.parametrize("b", [1.0, 1])
def test_inequality(V1, a, b):
    a = a(V1)

    inequality = a <= b
    assert isinstance(inequality, Inequality)
    assert inequality.lhs == a
    assert inequality.rhs == b
    assert str(inequality) == f"{a} <= {b}"

    inequality = b <= a
    assert isinstance(inequality, Inequality)
    assert inequality.lhs == -a
    assert inequality.rhs == -b

    inequality = a >= b
    assert isinstance(inequality, Inequality)
    assert inequality.lhs == -a
    assert inequality.rhs == -b

    inequality = b >= a
    assert isinstance(inequality, Inequality)
    assert inequality.lhs == a
    assert inequality.rhs == b

    assert (a <= b) == (b >= a)
    assert (a >= b) == (b <= a)
