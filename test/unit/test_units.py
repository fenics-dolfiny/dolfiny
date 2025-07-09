from mpi4py import MPI

import dolfinx
import ufl

import pytest
import sympy as sy
import sympy.physics.units as syu

from dolfiny.units import (
    Quantity,
    buckingham_pi_analysis,
    factorize,
    transform,
)


@pytest.fixture(scope="module")
def mesh():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def test_quantity_str_and_symbol(mesh):
    """Test that Quantity objects correctly store and display their symbol."""
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    assert str(length) == "L"
    assert length.symbol == L_symbol


def test_factorize_consistent_units(mesh):
    """Test that factorization works correctly for expressions with consistent units."""
    L_symbol = sy.Symbol("L")
    T_symbol = sy.Symbol("T")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T_symbol)
    expr = length * time
    result = factorize(expr, [length, time])
    # The factorized expression should be 1 (dimensionless constants)
    # The factor should represent the unit powers
    assert result.factor[0] == 1  # length factor
    assert result.factor[1] == 1  # time factor


def test_factorize_inconsistent_units_raises(mesh):
    """Test that factorization raises an error when trying to add incompatible units."""
    L_symbol = sy.Symbol("L")
    T_symbol = sy.Symbol("T")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T_symbol)
    expr = length + time
    with pytest.raises(RuntimeError, match="Inconsistent dimensions"):
        factorize(expr, [length, time], mode="check")


def test_transform_gradient(mesh):
    """Test that gradient expressions can be transformed with dimensional scaling."""
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V, name="u")
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    mapping = {u: length * u, mesh.ufl_domain(): length}
    grad_expr = ufl.grad(u)
    transformed = transform(grad_expr, mapping)
    # This should transform the gradient appropriately
    assert transformed is not None


def test_quantity_scale_setter(mesh):
    """Test that Quantity scale can be updated correctly."""
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    original_scale = length.scale
    # Set a new scale
    length.scale = 2.0
    assert length.scale == 2.0
    # The value should be updated accordingly
    assert length.scale != original_scale


def test_buckingham_pi_basic(mesh):
    """Test Buckingham Pi analysis for dimensional reduction of three physical quantities."""
    # Define three quantities: length (L), time (T), and velocity (L/T)
    L = sy.Symbol("L")
    T = sy.Symbol("T")
    V = sy.Symbol("V")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T)
    velocity = Quantity(mesh, 1.0, unit=syu.meter / syu.second, symbol=V)

    # Buckingham Pi theorem: for 3 quantities, 2 fundamental units (L, T), expect 1 Pi group
    dim_matrix, base_dims, pi_groups = buckingham_pi_analysis([length, time, velocity])
    assert isinstance(pi_groups, list)
    assert len(pi_groups) == 1
    # The pi group should be a sympy matrix representing the dimensionless combination
    assert pi_groups[0] is not None


def test_form_transformation(mesh):
    """Test dimensional transformation and factorization of UFL bilinear forms."""
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    u_ref = Quantity(mesh, 1.0, unit=syu.kelvin, symbol=L_symbol)

    # Create a simple form
    form = length**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    mapping = {u: u_ref * u, v: u_ref * v, mesh.ufl_domain(): length}

    # Transform the form
    transformed_form = transform(form, mapping)
    assert transformed_form is not None
    assert isinstance(transformed_form, ufl.Form)

    # Test factorization of the transformed form
    # The transformed form should be factorizable with the length quantity
    factorized = factorize(transformed_form, [length, u_ref])
    assert factorized is not None
    assert factorized.factor[0] == 2  # length factor should be 2 (L^2)
    assert factorized.factor[1] == 2
