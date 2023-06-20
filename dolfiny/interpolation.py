import logging

import basix
import cffi
import dolfinx
import ffcx.element_interface
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np
import ufl
from mpi4py import MPI
from numba.typed import List
from petsc4py import PETSc

import dolfiny

# Load ffi import-time, needed to be binded
# into numba compiled code for interpolation
ffi = cffi.FFI()


class CompiledExpression:
    def __init__(self, expr, target_el, comm=MPI.COMM_WORLD):
        self.expr = expr
        self.target_el = target_el
        self.is_affine_map = False

        if target_el.value_size > 1:
            if isinstance(target_el, basix.ufl._BlockedElement) or \
               isinstance(target_el, basix.ufl._MixedElement):
                # For blocked/mixed elements fetch only one element
                # Saves computation for vector/tensor elements,
                # no need to evaluate at same points for each vector
                # component
                #
                # TODO: We can get unique subelements or unique
                #       points for evaluation
                sub_elements = target_el.sub_elements()

                # We can handle only all sub elements equal case
                assert all([sub_elements[0] == x for x in sub_elements])
                target_el = sub_elements[0]

        # Identify points at which to evaluate the expression
        # self.ffcx_element = ffcx.element_interface.convert_element(target_el)
        self.basix_element = target_el

        if isinstance(self.basix_element, basix.ufl._BlockedElement):
            mapping_types = [self.basix_element.sub_element.element.map_type]
            nodes = self.basix_element.sub_element.element.points
        elif isinstance(self.basix_element, basix.ufl._MixedElement):
            mapping_types = [e.element.map_type for e in self.basix_element.sub_elements]
            nodes = self.basix_element.sub_elements[0].element.points
        elif isinstance(self.basix_element, basix.ufl._BasixElement):
            mapping_types = [self.basix_element.element.map_type]
            nodes = self.basix_element.element.points
        elif isinstance(self.basix_element, ffcx.element_interface.QuadratureElement):
            mapping_types = [basix._basixcpp.MapType.identity]
            nodes = self.basix_element._points
        else:
            raise NotImplementedError("Unsupported element type")

        if all(x == basix._basixcpp.MapType.identity for x in mapping_types):
            self.is_affine_map = True

        module = dolfinx.jit.ffcx_jit(comm, (expr, nodes))
        self.module = module


def interpolate(expr, target_func):
    """Interpolate UFL expression.

    Note
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

    if (len(linear_comb) > 0
        and all([func.function_space == linear_comb[0][0].function_space for func, _ in linear_comb])
            and target_func.function_space == linear_comb[0][0].function_space):

        logger.info(f"Interpolating linear combination of vectors for {expr_float}")

        # If FunctionSpace of all donor and target functions are the same
        linear_comb_acc = {}

        # Accumulate all repeated occurences of the same function
        for func, scalar in linear_comb:
            if func in linear_comb_acc:
                linear_comb_acc[func] += scalar
            else:
                linear_comb_acc[func] = scalar

        with target_func.vector.localForm() as target_local:
            target_local.set(0.0)

        for func, scalar in linear_comb_acc.items():
            with target_func.vector.localForm() as target_local, func.vector.localForm() as func_local:
                target_local.axpy(scalar, func_local)
    else:
        compiled_expression = CompiledExpression(expr, target_func.function_space.ufl_element())
        interpolate_compiled(compiled_expression, target_func)


def interpolate_compiled(compiled_expression, target_func):
    """Compiled interpolation

    Interpolates UFL expression into target function using FFCx
    code generation and compilation.


    Note
    ----
    Works only for affine-mapped point-evaluation finite elements, e.g.
    lagrange/discontinuous lagrange of arbitrary order.

    """

    if not compiled_expression.is_affine_map:
        raise NotImplementedError("Only affine mapped function spaces supported")

    kernel = compiled_expression.module[0].tabulate_tensor_float64

    # Register complex types
    cffi_support.register_type(ffi.typeof('double _Complex'),
                               numba.types.complex128)
    cffi_support.register_type(ffi.typeof('float _Complex'),
                               numba.types.complex64)

    # Unpack mesh and dofmap data
    mesh = target_func.function_space.mesh
    geom_dofmap = mesh.geometry.dofmap
    geom = mesh.geometry.x

    dofmap = target_func.function_space.dofmap.list.flatten()

    # Prepare coefficients and their dofmaps
    # Global vectors and dofmaps are prepared here, local are
    # fetched inside hot cell-loop

    # Number of coefficients in ffcx-processed ufl form
    num_coeffs = compiled_expression.module[0].num_coefficients
    # Positions of ffcx-preprocessed coefficients in original form
    cpos = compiled_expression.module[0].original_coefficient_positions

    coeffs = ufl.algorithms.analysis.extract_coefficients(compiled_expression.expr)
    coeffs_dofmaps = List.empty_list(numba.types.Array(numba.typeof(dofmap[0]), 1, "C", readonly=True))
    coeffs_vectors = List.empty_list(numba.types.Array(numba.typeof(PETSc.ScalarType()), 1, "C", readonly=True))
    coeffs_bs = List.empty_list(numba.types.int_)

    for i in range(num_coeffs):
        coeffs_dofmaps.append(coeffs[cpos[i]].function_space.dofmap.list.flatten())
        coeffs_vectors.append(np.asarray(coeffs[cpos[i]].vector))
        coeffs_bs.append(coeffs[cpos[i]].function_space.dofmap.bs)

    coeffs_sizes = np.asarray([coeff.function_space.element.space_dimension
                               for coeff in coeffs], dtype=np.int_)

    # Prepare and pack constants
    constants = ufl.algorithms.analysis.extract_constants(compiled_expression.expr)
    constants_vector = np.array([], dtype=PETSc.ScalarType())
    if len(constants) > 0:
        constants_vector = np.hstack([c.value.flatten() for c in constants])

    value_size = int(np.product(compiled_expression.expr.ufl_shape))
    basix_space_dim = compiled_expression.basix_element.dim
    space_dim = target_func.function_space.element.space_dimension

    dofmap_bs = target_func.function_space.dofmap.bs
    element_bs = target_func.function_space.dofmap.dof_layout.block_size

    # Prepare mapping of subelements into the parent finite element
    # This mapping stores also how dofs are collapsed when symmetry to a TensorElement
    # is applied
    if hasattr(compiled_expression.target_el, "flattened_sub_element_mapping") and \
       compiled_expression.target_el._has_symmetry:
        subel_map = np.array(compiled_expression.target_el.flattened_sub_element_mapping())
    else:
        subel_map = np.array(range(value_size))

    num_coeffs = len(coeffs_vectors)

    with target_func.vector.localForm() as b:
        b.set(0.0)
        assemble_vector_ufc(np.asarray(b), kernel, (geom_dofmap, geom), dofmap,
                            coeffs_vectors, coeffs_dofmaps, coeffs_bs, constants_vector,
                            coeffs_sizes, basix_space_dim, space_dim,
                            value_size, subel_map, dofmap_bs, element_bs)


@numba.njit(fastmath=True)
def assemble_vector_ufc(b, kernel, mesh, dofmap, coeffs_vectors, coeffs_dofmaps, coeffs_bs,
                        const_vector, coeffs_sizes, fiat_space_dim,
                        space_dim, value_size, subel_map, dofmap_bs, element_bs):
    geom_dofmap, geom = mesh

    num_geometry_dofs = geom_dofmap.shape[1]

    # Coord dofs have shape (num_geometry_dofs, gdim)
    coordinate_dofs = np.zeros((num_geometry_dofs, 3))
    coeffs = np.zeros(np.sum(coeffs_sizes), dtype=PETSc.ScalarType)
    entity_index = np.array([0], dtype=np.intc)
    quad_perm = np.array([0], dtype=np.dtype("uint8"))

    # Allocate space for local element tensor
    # This has the size which generated code expects, not the local
    # dofmap of the actual element (these are different for symmetric spaces)
    b_local = np.zeros(fiat_space_dim * value_size, dtype=PETSc.ScalarType)

    for i, cell in enumerate(geom_dofmap):
        for j, dof in enumerate(cell):
            for k in range(3):
                coordinate_dofs[j, k] = geom[dof, k]
        b_local.fill(0.0)

        offset = 0
        for j in range(len(coeffs_vectors)):
            bs = coeffs_bs[j]
            local_dofsize = coeffs_sizes[j] // bs
            for k in range(local_dofsize):
                for m in range(bs):
                    coeffs[bs * k + m + offset] = coeffs_vectors[j][bs * coeffs_dofmaps[j][local_dofsize * i + k] + m]
            offset += coeffs_sizes[j]

        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(const_vector), ffi.from_buffer(coordinate_dofs),
               ffi.from_buffer(entity_index), ffi.from_buffer(quad_perm))

        for j in range(fiat_space_dim):
            for k in range(value_size):
                b[dofmap_bs * dofmap[i * fiat_space_dim + j] + subel_map[k]] = b_local[value_size * j + k]
