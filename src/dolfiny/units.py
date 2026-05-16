import fractions
import logging
import math
from typing import Any, NamedTuple, overload

import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar
from ufl.algorithms.map_integrands import map_integrands
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.form import Form

import numpy as np
import sympy as sy
from sympy.physics.units import UnitSystem
from sympy.physics.units.dimensions import Dimension

from dolfiny.utils import print_table

logger = logging.getLogger(__name__)


class FactorizedExpr(NamedTuple):
    expr: Expr | Form
    factor: np.ndarray | None


class Quantity(dolfinx.fem.Constant):
    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        scale: float | int,
        unit: sy.Expr,
        symbol: str | sy.Symbol,
        unit_system: UnitSystem = sy.physics.units.si.SI,
        **kwargs,
    ):
        if not isinstance(scale, int | float):
            raise TypeError(f"Scale must be a numeric type, got {type(scale).__name__}.")

        if np.asarray(scale).shape != ():
            raise ValueError("Quantity supports only scalar values.")

        if unit is None:
            unit = sy.sympify(1)

        if not isinstance(unit, sy.Expr):
            raise TypeError(f"Unit must be a sympy expression, got {type(unit).__name__}.")

        self._scale = scale
        self._unit = unit

        if not isinstance(symbol, str | sy.Symbol):
            raise TypeError(
                f"Symbol must be a string or sympy.Symbol, got {type(symbol).__name__}."
            )

        self._symbol = (
            sy.Symbol(symbol, positive=True, real=True) if isinstance(symbol, str) else symbol
        )

        self._unit_system = unit_system

        self._factor, self._dimension, self._dimensional_dependencies = get_factor(
            self._scale, self._unit, self._unit_system
        )
        super().__init__(mesh, scalar(self._factor), **kwargs)

    @property
    def dimension(self):
        return self._dimension

    @property
    def dimensional_dependencies(self):
        return self._dimensional_dependencies

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        self._factor, self._dimension, self._dimensional_dependencies = get_factor(
            self._scale, self._unit, self._unit_system
        )
        np.copyto(self._cpp_object.value, np.asarray(scalar(self._factor)))

    @property
    def expr(self):
        return self._scale * self._unit

    @property
    def unit(self):
        return self._unit

    @property
    def unit_system(self):
        return self._unit_system

    @property
    def symbol(self):
        return self._symbol

    def __str__(self):
        return str(self._symbol)

    def __repr__(self):
        return (
            f"Quantity(scale={self._scale}, unit={self._unit}, symbol={self._symbol}, "
            f"unit_system={self._unit_system})"
        )


class UnitTransformer(MultiFunction):
    def __init__(self, mapping: dict):
        self.mapping = mapping
        meshes = [mesh for mesh in mapping.keys() if isinstance(mesh, ufl.Mesh)]
        if len(meshes) != 1:
            raise ValueError("Mapping must contain exactly one Mesh.")
        self._mesh_scale = self.mapping[meshes[0]]

        if not isinstance(self._mesh_scale, Quantity):
            raise TypeError("Mesh scale must be a Quantity.")

        super().__init__()

    def ufl_type(self, o, *args):
        return self.mapping.get(o, self.reuse_if_untouched(o, *args))

    def grad(self, o, a):
        return o._ufl_expr_reconstruct_(a) / self._mesh_scale

    def spatial_coordinate(self, o, *ops):
        return o * self._mesh_scale

    def cell_volume(self, o, *ops):
        tdim = o.ufl_domain().topological_dimension()
        return o * (self._mesh_scale**tdim)

    def facet_area(self, o, *ops):
        tdim = o.ufl_domain().topological_dimension()
        return o * (self._mesh_scale ** (tdim - 1))

    div = grad
    curl = grad
    circumradius = spatial_coordinate
    min_edge_length = spatial_coordinate
    max_edge_length = spatial_coordinate


class QuantityFactorizer(MultiFunction):
    factors: dict[Expr, np.ndarray]

    def __init__(self, quantities: list["Quantity"], mode="factorize"):
        self._quantities = quantities

        if mode not in ("factorize", "check"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'factorize' or 'check'.")

        self._mode = mode
        if any(q.ufl_shape != () for q in quantities):
            raise NotImplementedError("Factorization of non-scalar quantities is not implemented.")
        self.factors = {}
        super().__init__()

    def product(self, o, *ops):
        a, b = o.ufl_operands
        self.factors[o] = self.factors[a] + self.factors[b]
        return self.reuse_if_untouched(o, *ops)

    def division(self, o, *ops):
        a, b = o.ufl_operands
        self.factors[o] = self.factors[a] - self.factors[b]
        return self.reuse_if_untouched(o, *ops)

    def power(self, o, *ops):
        a, n = o.ufl_operands
        self.factors[o] = float(n) * self.factors[a]
        return self.reuse_if_untouched(o, *ops)

    def sqrt(self, o, *ops):
        a = o.ufl_operands[0]
        self.factors[o] = self.factors[a] / 2
        return self.reuse_if_untouched(o, *ops)

    def independent(self, o, *ops):
        self.factors.setdefault(o, np.zeros(len(self._quantities)))
        return self.reuse_if_untouched(o, *ops)

    def linear(self, o, *ops):
        # Linear nodes which have all operands with the same factor (e.g. sum, grad, etc.)
        # can be assigned that factor.
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        if len(factors) > 0:
            self._check_operands(o, factors[0])
            self.factors[o] = factors[0]

        return self.reuse_if_untouched(o, *ops)

    def inhomogeneous(self, o, *ops):
        # Inhomogeneous nodes (e.g. sin(x), exp(x), etc.) must have dimensionless operands,
        # and in factorize mode can only have a trivial factor of 1.
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        if len(factors) > 0:
            self._check_operands(o, np.zeros_like(factors[0]))
            self.factors[o] = factors[0]
        return self.reuse_if_untouched(o, *ops)

    def multi_index(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def label(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def constant(self, o, *ops):
        if o in self._quantities:
            idx = self._quantities.index(o)
            self.factors[o] = np.zeros(len(self._quantities))
            self.factors[o][idx] = 1
            return 1
        else:
            self.factors.setdefault(o, np.zeros(len(self._quantities)))
            return self.reuse_if_untouched(o, *ops)

    def expr_list(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def _check_operands(self, o, reference_factor):
        r"""Check that all operands of the expression are consistent with a reference factor.

        Verifies that all operands of a given expression have compatible units/dimensions
        and, in factorize mode, identical factors.

        """
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        f0_expr = expand(reference_factor, [q.dimension for q in self._quantities]).simplify()
        f0_symbol = expand(reference_factor, [q.symbol for q in self._quantities])

        for fb in factors:
            fb_symbol = expand(fb, [q.symbol for q in self._quantities])

            dimsys = self._quantities[0].unit_system.get_dimension_system()
            fb_expr = expand(fb, [q.dimension for q in self._quantities]).simplify()
            if not dimsys.equivalent_dims(f0_expr, fb_expr):
                raise RuntimeError(
                    f"Inconsistent dimensions in operands of {o.__class__.__name__}.\n"
                    f"dim({f0_symbol}) != dim({fb_symbol}), (i.e. {f0_expr} != {fb_expr}).\n"
                )

            if self._mode == "factorize" and not np.allclose(factors[0], fb):
                raise RuntimeError(
                    f"Inconsistent factors in operands of {o.__class__.__name__}.\n"
                    f"{f0_symbol} != {fb_symbol}."
                )

    terminal = independent

    sum = linear
    indexed = linear
    grad = linear
    div = linear
    conj = linear
    index_sum = linear
    transposed = linear
    deviatoric = linear
    sym = linear
    trace = linear
    variable = linear
    coefficient_derivative = linear
    component_tensor = linear

    variable_derivative = division

    inner = product
    dot = product
    cross = product

    expr = inhomogeneous


def _transform_expr(expr: Expr, mapping: dict):
    transformer = UnitTransformer(mapping)
    return map_expr_dag(transformer, expr)


def _transform_form(form: Form, mapping: dict) -> Form:
    _integral_type_codim = {"cell": 0, "interior_facet": 1, "exterior_facet": 1}

    transformer = UnitTransformer(mapping)
    transformed_integrals = []

    for integral in form.integrals():
        # Transform the integrand
        transformed_integrand = map_expr_dag(transformer, integral.integrand())

        # Scale by measure change
        tdim = integral.ufl_domain().topological_dimension
        measure_dim = tdim - _integral_type_codim[integral.integral_type()]
        scaled_integrand = transformed_integrand * (transformer._mesh_scale**measure_dim)

        transformed_integrals.append(integral.reconstruct(scaled_integrand))

    return Form(transformed_integrals)


def transform(expr: Expr | Form | dict, mapping: dict):
    """Transform expressions or forms by applying unit mapping and mesh scaling.

    Parameters
    ----------
    expr
        Expression, form, or dictionary to transform
    mapping
        Mapping of quantities to their replacements

    Returns
    -------
    Transformed expression, form, or dictionary

    """
    if isinstance(expr, dict):
        return {key: transform(value, mapping) for key, value in expr.items()}
    if isinstance(expr, Form):
        return _transform_form(expr, mapping)
    if isinstance(expr, Expr):
        return _transform_expr(expr, mapping)

    raise TypeError(f"Unsupported type for unit transformation: {type(expr).__name__}")


@overload
def factorize(
    expr: dict,
    quantities: list["Quantity"],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> dict[str, FactorizedExpr]: ...


@overload
def factorize(
    expr: Expr | Form,
    quantities: list["Quantity"],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> FactorizedExpr: ...


def factorize(
    expr: Expr | Form | dict,
    quantities: list["Quantity"],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> FactorizedExpr | dict[str, FactorizedExpr]:
    """Factorize expressions, forms, or dictionaries to extract dimensional factors.

    Parameters
    ----------
    expr
        Expression, form, or dictionary to factorize
    quantities
        List of quantities to use for factorization
    mode
        Factorization mode: "factorize" or "check" (default: "factorize")
    mapping
        Optional mapping for unit transformation

    Returns
    -------
    FactorizedExpr | dict
        Factorized expression with dimensional factors, or dict of factorized items

    """
    if mapping is not None:
        expr = transform(expr, mapping)

    if isinstance(expr, dict):
        return {key: factorize(value, quantities, mode=mode) for key, value in expr.items()}
    if mode not in ("factorize", "check"):
        raise RuntimeError(f"{mode} is not a valid factorisation mode.")
    if isinstance(expr, Form):
        factorized_integrals = []
        factors = []
        for integral in expr.integrals():
            factorizer = QuantityFactorizer(quantities, mode=mode)
            factorized_integrals.append(
                integral.reconstruct(map_expr_dag(factorizer, integral.integrand()))
            )
            integral_root_expr = next(reversed(factorizer.factors), None)
            integral_root_factor = (
                factorizer.factors[integral_root_expr] if integral_root_expr is not None else None
            )
            factors.append(integral_root_factor)
        for i in range(len(factors) - 1):
            unit_system = quantities[0].unit_system
            dimsys = unit_system.get_dimension_system()
            fa_expr = expand(factors[i], [q.dimension for q in quantities]).simplify()  # type: ignore
            fb_expr = expand(factors[i + 1], [q.dimension for q in quantities]).simplify()  # type: ignore
            fa_symbol = expand(factors[i], [q.symbol for q in quantities])  # type: ignore
            fb_symbol = expand(factors[i + 1], [q.symbol for q in quantities])  # type: ignore
            if dimsys.equivalent_dims(fa_expr, fb_expr) is False:
                raise RuntimeError(
                    "Inconsistent dimensions across integrals in Form. \n"
                    f"Scales: {fa_symbol} != {fb_symbol}. \n"
                    f"{fa_expr} != {fb_expr}."
                )

            if mode == "factorize":
                fa_symbol = expand(factors[i], [q.symbol for q in quantities])  # type: ignore
                fb_symbol = expand(factors[i + 1], [q.symbol for q in quantities])  # type: ignore
                if not np.allclose(factors[i], factors[i + 1]):  # type: ignore
                    raise RuntimeError(
                        "Inconsistent factors across integrals in Form. \n"
                        f"{fa_symbol} != {fb_symbol}."
                    )

        factorized_expression = Form(factorized_integrals)
    else:
        factorizer = QuantityFactorizer(quantities, mode=mode)
        factorized_expression = map_expr_dag(factorizer, expr)

    root_expr = next(reversed(factorizer.factors), None)
    root_factor = factorizer.factors[root_expr] if root_expr is not None else None

    return FactorizedExpr(factorized_expression, root_factor)


def expand(factor: np.ndarray | list, quantities: list) -> sy.Expr:
    """Expand factor array into symbolic expression using quantities as base.

    Parameters
    ----------
    factor
        Array of exponents for each quantity
    quantities
        List of quantities/symbols to use as base

    Returns
    -------
    sy.Expr
        Symbolic expression with quantities raised to corresponding powers

    """
    return math.prod(q ** fractions.Fraction(f) for q, f in zip(quantities, factor))


def dimension_matrix(
    quantities: list[Quantity],
    unit_system: UnitSystem = sy.physics.units.systems.SI,
) -> tuple[sy.Matrix, list[Dimension]]:
    """Build dimension matrix for Buckingham Pi analysis.

    Returns matrix where each column represents a quantity's dimensional exponents
    with respect to base dimensions.
    """
    dimsys = unit_system.get_dimension_system()
    base_dims: list[Dimension] = dimsys.base_dims

    # Build matrix: each column is the exponents of base_dims for a quantity
    rows: list[list[int]] = []
    for dim in base_dims:
        row: list[int] = []
        for q in quantities:
            deps: dict[Dimension, int] = q.dimensional_dependencies
            row.append(deps.get(dim, 0))
        rows.append(row)

    matrix = sy.Matrix(rows)
    return matrix, base_dims


def buckingham_pi_analysis(
    quantities: list[Quantity],
    unit_system: UnitSystem = sy.physics.units.systems.SI,
    outlier_threshold: float = 1e-16,
) -> tuple[sy.Matrix, list[Dimension], list[sy.Matrix]]:
    """Perform Buckingham Pi analysis to find dimensionless groups.

    Parameters
    ----------
    quantities
        List of physical quantities to analyze
    unit_system
        Unit system for analysis (default: SI)
    outlier_threshold
        Threshold for detecting outlier values (default: 1e-3)

    Returns
    -------
        Dimension matrix, base dimensions, and Pi groups

    """
    dim_matrix, base_dims = dimension_matrix(quantities, unit_system)

    logger.info("")
    logger.info("=" * 50)
    logger.info("Buckingham Pi Analysis")
    logger.info("=" * 50)

    # Print quantities in a table
    rows: list[list[Any]] = []
    for q in quantities:
        rows.append(
            [
                str(q.symbol),
                str(q.expr.simplify().n(4)),
                f"{to_base_units(q.expr, unit_system).simplify().n(4)}",
            ]
        )

    print_table(rows, ["Symbol", "Expression", "Value (in base units)"])
    logger.info("")
    logger.info(f"Dimension matrix ({len(base_dims)} x {len(quantities)}):")
    dim_array = np.array(dim_matrix).astype(float)

    # Create header with quantity symbols
    header = ["Dimension"] + [str(q.symbol) for q in quantities]

    # Create rows with dimension names and their exponents for each quantity
    rows = []
    for i, dim in enumerate(base_dims):
        row = [str(dim.name)] + [f"{int(dim_array[i, j])}" for j in range(len(quantities))]
        rows.append(row)

    print_table(rows, header)
    logger.info("")

    pi_groups = dim_matrix.nullspace()

    logger.info(f"Dimensionless groups ({len(pi_groups)}):")
    rows.clear()
    outliers: list[tuple[int, sy.Expr, float]] = []

    average_group_value = np.mean(
        [
            np.prod(
                [
                    q.value ** float(pi_group[j])
                    for j, q in enumerate(quantities)
                    if pi_group[j] != 0
                ]
            )
            for pi_group in pi_groups
        ]
    )

    for i, pi_group in enumerate(pi_groups):
        expr = sy.simplify(expand(pi_group, [q.symbol for q in quantities]))
        numerical_value = np.prod(
            [q.value ** float(pi_group[j]) for j, q in enumerate(quantities) if pi_group[j] != 0]
        )

        # Check if numerical value is an outlier (too small or too large)
        is_outlier = (
            numerical_value < average_group_value * outlier_threshold
            or numerical_value > average_group_value / outlier_threshold
        )

        rows.append([f"Pi_{i + 1}", expr, f"{numerical_value:.3g}"])
        if is_outlier:
            outliers.append((i + 1, expr, numerical_value))

    # Print the dimensionless groups in a table
    print_table(rows, ["Group", "Expression", "Value"])

    if outliers:
        logger.warning("\nThe following dimensionless groups have outlier values:")
        for pi_num, expr, value in outliers:
            logger.warning(f"  Pi_{pi_num:2d} = {value:.3g}")

    logger.info("=" * 50)
    return dim_matrix, base_dims, pi_groups


def to_base_units(expr: sy.Expr, unit_system: UnitSystem = sy.physics.units.systems.SI) -> sy.Expr:
    """Convert expression to base units of the given unit system.

    Parameters
    ----------
    expr
        Expression containing units to be converted.
    unit_system
        Unit system to use for conversion. Default is SI system.

    Returns
    -------
        Expression with units converted to base units of the specified system.

    """
    base_units = unit_system._base_units
    return sy.physics.units.convert_to(expr, base_units)


def get_factor(
    scale: float | int,
    unit: sy.physics.units.Unit,
    unit_system: sy.physics.units.UnitSystem = sy.physics.units.si.SI,
) -> tuple[
    float, sy.physics.units.dimensions.Dimension, dict[sy.physics.units.dimensions.Dimension, int]
]:
    """Extract numerical factor, dimension, and dimensional dependencies from a scaled unit.

    Parameters
    ----------
    scale
        Numerical scaling factor
    unit
        Unit to analyze
    unit_system
        Unit system for conversion (default: SI)

    Returns
    -------
        (factor, dimension, dimensional_dependencies)

    """
    base_units = unit_system._base_units
    base_value = sy.physics.units.convert_to(scale * unit, base_units)
    strip_map = {unit: 1 for unit in base_units}
    factor = base_value.subs(strip_map)

    _, dimension = unit_system._collect_factor_and_dimension(unit)
    dimensional_dependencies = unit_system.get_dimension_system().get_dimensional_dependencies(
        dimension
    )

    if not factor.is_number:
        raise ValueError(f"Cannot convert {scale * unit} to base units {base_units}.")

    return factor, dimension, dimensional_dependencies


def get_dimension(
    expr: Expr | Form, quantities: list[Quantity], mapping: dict | None = None
) -> sy.Expr:
    """Get the physical dimension of an expression.

    Parameters
    ----------
    expr
        The expression to analyze for dimensional consistency.
    quantities
        List of quantities with their associated dimensions.
    mapping
        Optional mapping for variable substitution. Default is None.

    Returns
    -------
    unit
        The simplified physical dimension of the expression.

    """
    factor = factorize(expr, quantities, mode="check", mapping=mapping)
    assert isinstance(factor, FactorizedExpr)
    assert factor.factor is not None, "Factorized expression must have a non-None factor"

    unit = expand(factor.factor, [q.dimension for q in quantities]).simplify()
    return unit


def normalize(
    expr_dict: dict[str, FactorizedExpr],
    reference_key: str,
    quantities: list[Quantity],
) -> dict[str, Expr | Form]:
    """Normalize expressions or forms with respect to a reference expression.

    Parameters
    ----------
    expr_dict
        Dictionary of expressions or forms to normalize
    reference_key : str
        Key of the reference expression for normalization
    quantities
        List of quantities to use for factorization
    mapping
        Mapping for unit transformation

    Returns
    -------
        Dictionary of normalized expressions or forms

    """
    if reference_key not in expr_dict:
        raise KeyError(f"Reference key '{reference_key}' not found in expression dictionary")

    # Validate that expr_dict contains factorized expressions
    for key, value in expr_dict.items():
        if not isinstance(value, FactorizedExpr):
            raise TypeError(
                f"Expression '{key}' must be already factorized (FactorizedExpr),"
                f"got {type(value).__name__}."
            )
        if value.factor is None:
            raise ValueError(
                f"Expression '{key}' has a None factor. "
                "All expressions must have valid factors for normalization."
            )

    # Get reference factor
    ref_factor = expr_dict[reference_key].factor
    assert ref_factor is not None  # Already validated above

    logger.info("")
    logger.info("=" * 50)
    logger.info(f'Terms after normalization with "{reference_key}"')
    logger.info("=" * 50)

    # Print reference factor information
    ref_factor_sym = expand(ref_factor, [q.symbol for q in quantities])
    ref_factor_expr = to_base_units(expand(ref_factor, [q.expr for q in quantities]))

    logger.info(f"Reference factor from '{reference_key}':")
    ref_rows = [[reference_key, str(ref_factor_sym), ref_factor_expr.simplify().n(4)]]
    print_table(ref_rows, ["Term", "Factor", "Value (in base units)"])
    logger.info("")

    # Create table showing original and normalized factors
    rows = []
    normalized_dict = {}

    for key, factorized_expr in expr_dict.items():
        original_factor = factorized_expr.factor
        assert original_factor is not None  # Already validated above
        normalized_factor = original_factor - ref_factor

        ratio_sym = expand(normalized_factor, [q.symbol for q in quantities])
        ratio_expr = to_base_units(expand(normalized_factor, [q.expr for q in quantities]))

        rows.append([key, str(ratio_sym), ratio_expr.simplify().n(4)])

        # Create normalized expression by dividing by reference factor
        ref_value = expand(normalized_factor, quantities)
        if isinstance(factorized_expr.expr, Form):
            normalized_dict[key] = map_integrands(lambda x: x * ref_value, factorized_expr.expr)
        else:
            normalized_dict[key] = factorized_expr.expr / ref_value

    print_table(rows, ["Term", "Factor", "Value (in base units)"])
    logger.info("=" * 50)

    return normalized_dict


def collect_quantities(expr, mapping: dict | None = None) -> list[Quantity]:
    """Collect all Quantity instances from a UFL expression."""
    if mapping is not None:
        expr = transform(expr, mapping)

    quantities = set()

    class QuantityCollector(MultiFunction):
        def ufl_type(self, o, *args):
            return self.reuse_if_untouched(o, *args)

        def constant(self, o, *ops):
            if isinstance(o, Quantity):
                quantities.add(o)
            return self.reuse_if_untouched(o, *ops)

    if isinstance(expr, Form):
        for integral in expr.integrals():
            map_expr_dag(QuantityCollector(), integral.integrand())
        return list(quantities)
    elif isinstance(expr, Expr):
        map_expr_dag(QuantityCollector(), expr)
    else:
        raise TypeError(f"Unsupported type for collecting quantities: {type(expr).__name__}")

    return list(quantities)
