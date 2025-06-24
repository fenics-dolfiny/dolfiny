import logging

import basix
import dolfinx

import dolfiny


def interpolate(expr, target_func):
    """Interpolate UFL expression.

    Note:
    ----
    This method decides if interpolation is possible purely as linear combination
    of some underlying PETSc vectors. In such case this approach is chosen.

    """
    logger = logging.getLogger("dolfiny")

    linear_comb = []
    try:
        expr_float = dolfiny.expression.evaluate_constants(expr)
        dolfiny.expression.extract_linear_combination(expr_float, linear_comb)
    except RuntimeError:
        linear_comb = []
        pass

    if (
        len(linear_comb) > 0
        and all(
            [func.function_space == linear_comb[0][0].function_space for func, _ in linear_comb]
        )
        and target_func.function_space == linear_comb[0][0].function_space
    ):
        logger.info(f"Interpolating linear combination of vectors for {expr_float}")

        # If FunctionSpace of all donor and target functions are the same
        linear_comb_acc = {}

        # Accumulate all repeated occurrences of the same function
        for func, scalar in linear_comb:
            if func in linear_comb_acc:
                linear_comb_acc[func] += scalar
            else:
                linear_comb_acc[func] = scalar

        with target_func.x.petsc_vec.localForm() as target_local:
            target_local.set(0.0)

        for func, scalar in linear_comb_acc.items():
            with (
                target_func.x.petsc_vec.localForm() as target_local,
                func.x.petsc_vec.localForm() as func_local,
            ):
                target_local.axpy(scalar, func_local)
    else:
        T = target_func.function_space

        def is_quadrature_element(element):
            if isinstance(element, basix.ufl._QuadratureElement):
                return True
            elif element.block_size != 1:
                return isinstance(element.sub_elements[0], basix.ufl._QuadratureElement)
            else:
                return False

        try:
            # expr is a Function or Expression
            if isinstance(expr, dolfinx.fem.Function) and not (
                is_quadrature_element(expr.function_space.ufl_element())
                or is_quadrature_element(T.ufl_element())
            ):
                # proceed with Function as long as source/target are not QudratureElement
                logger.info("Interpolating given dolfinx.fem.Function")
                target_func.interpolate(expr)
            else:
                logger.info("Interpolating given ufl.Expr")
                e = dolfinx.fem.Expression(expr, T.element.interpolation_points)
                target_func.interpolate(e)
        except TypeError:
            # expr is callable
            assert callable(expr)
            target_func.interpolate(expr)

        target_func.x.scatter_forward()
