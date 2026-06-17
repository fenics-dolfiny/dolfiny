from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import pathlib
import typing
from collections.abc import Sequence

import dolfinx
import ffcx
import ufl
from dolfinx.fem import DirichletBC, Form, IntegralType
from dolfinx.fem.forms import _ufl_to_dolfinx_domain, get_integration_domains
from ffcx.compiler import compile_ufl_objects
from ffcx.options import get_options

import cppyy
import numpy as np
import numpy.typing as npt

import dolfiny
import dolfiny.cpp._cpp
from dolfiny.cpp._cpp import pack_coefficients as _pack_coefficients
from dolfiny.cpp._cpp import pack_constants as _pack_constants

if typing.TYPE_CHECKING:
    # import dolfinx.mesh just when doing type checking to avoid
    # circular import
    from dolfinx.mesh import EntityMap as _EntityMap

nptype_to_cpp = {np.float64: "double"}

# Alias type map is a workaround to pack custom running error type to a numpy array.
alias_type_map = {np.float64: np.complex128}

# Cache is used to avoid re-compiling the same form multiple times,
# since cppyy does not recognize that two identical declarations are
# the same and will raise a redefinition error.
_cppyy_form_cache: dict[str, cppyy._cpython_cppyy.Template] = {}


def assemble_scalar(M: Form) -> typing.Any:
    constants = _pack_constants(M._cpp_object)
    coeffs = _pack_coefficients(M._cpp_object)
    val = dolfiny.cpp._cpp.assemble_scalar(M._cpp_object, constants, coeffs)
    err = val.exact_error if hasattr(val, "exact_error") else val.err_bnd
    return complex(val.val, err)


def assemble_vector(L: Form) -> np.ndarray:
    constants = _pack_constants(L._cpp_object)
    coeffs = _pack_coefficients(L._cpp_object)

    V = L.function_spaces[0]
    b = np.zeros(
        (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs,
        dtype=np.complex128,
    )

    dolfiny.cpp._cpp.assemble_vector(b, L._cpp_object, constants, coeffs)
    return b


def create_matrix(
    a: Form, block_mode: dolfinx.la.BlockMode | None = None
) -> dolfiny.cpp._cpp.MatrixCSR_re_worst_double | dolfiny.cpp._cpp.MatrixCSR_re_exact_double:
    """Create a sparse matrix that is compatible with a bilinear form.

    Args:
        a: Bilinear form.
        block_mode: Block mode of the CSR matrix. If ``None``, default
            is used.

    Returns:
        A sparse matrix that the form can be assembled into.

    """
    sp = dolfiny.cpp._cpp.create_sparsity_pattern(a._cpp_object)
    sp.finalize()

    _type = (
        dolfiny.cpp._cpp.MatrixCSR_re_worst_double
        if isinstance(a._cpp_object, dolfiny.cpp._cpp.Form_re_worst_double)
        else dolfiny.cpp._cpp.MatrixCSR_re_exact_double
    )
    if block_mode is not None:
        return _type(sp, block_mode=block_mode)
    else:
        return _type(sp)


@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: Sequence[DirichletBC] | None = None,
    diag: complex = 1.0 + 0j,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
    block_mode: dolfinx.la.BlockMode | None = None,
) -> dolfiny.cpp._cpp.MatrixCSR_re_worst_double | dolfiny.cpp._cpp.MatrixCSR_re_exact_double:
    """Assemble bilinear form into a matrix."""
    bcs = [] if bcs is None else bcs
    A: dolfinx.la.MatrixCSR = create_matrix(a, block_mode)
    _assemble_matrix_csr(A, a, bcs, diag, constants, coeffs)
    return A


@assemble_matrix.register
def _assemble_matrix_csr(
    A: dolfinx.la.MatrixCSR,
    a: Form,
    bcs: Sequence[DirichletBC] | None = None,
    diag: complex = 1.0 + 0j,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> dolfiny.cpp._cpp.MatrixCSR_re_worst_double | dolfiny.cpp._cpp.MatrixCSR_re_exact_double:
    """Assemble bilinear form into a matrix."""
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]

    if constants is None:
        constants = _pack_constants(a._cpp_object)

    if coeffs is None:
        coeffs = _pack_coefficients(a._cpp_object)

    dolfiny.cpp._cpp.assemble_matrix(A, a._cpp_object, constants, coeffs, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    diag_t = (
        dolfiny.cpp._cpp.ReWorstDouble
        if isinstance(A, dolfiny.cpp._cpp.MatrixCSR_re_worst_double)
        else dolfiny.cpp._cpp.ReExactDouble
    )
    diag = diag_t(diag.real, diag.imag)
    if a.function_spaces[0] is a.function_spaces[1]:
        dolfiny.cpp._cpp.insert_diagonal(A, a.function_spaces[0], bcs, diag)
    return A


def form(
    form: ufl.Form | Sequence[ufl.Form] | Sequence[Sequence[ufl.Form]],
    dtype: npt.DTypeLike = np.float64,
    form_compiler_options: dict | None = None,
    entity_maps: Sequence[_EntityMap] | None = None,
    mode: str = "worst",
):
    form_compiler_options = form_compiler_options or {}

    if dtype != np.float64:
        raise NotImplementedError(f"Unsupported dtype: {dtype}. Only np.float64 is implemented.")

    _cpp = dolfiny.cpp._cpp
    geometry_type = np.float64

    if mode == "worst":
        cpp_type = "dolfiny::running_error_t<double, dolfiny::ErrorMode::WORST>"
        ftype = _cpp.Form_re_worst_double
        fn_type = _cpp.Function_re_worst_double
    elif mode == "exact":
        cpp_type = "dolfiny::running_error_t<double, dolfiny::ErrorMode::EXACT>"
        ftype = _cpp.Form_re_exact_double
        fn_type = _cpp.Function_re_exact_double
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'worst' or 'exact'.")

    def _form(form):
        sd = form.subdomain_data()
        (domain,) = sd.keys()

        for data in sd[domain].values():
            non_none = [d for d in data if d is not None]
            assert len(non_none) == 0 or all(d is non_none[0] for d in non_none)

        msh = domain.ufl_cargo()
        if msh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        ufcx_form_template = jit(form, form_compiler_options=form_compiler_options)
        ufcx_form = ufcx_form_template[cpp_type, nptype_to_cpp[geometry_type]]

        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]
        if form_compiler_options.get("part", "full") == "diagonal":
            V = [V[0]]

        # running_error_t<double> maps to complex128 in numpy: real=val, imag=err_bnd
        original_coeffs = form.coefficients()
        coeffs = []
        for i in range(ufcx_form.num_coefficients):
            c = original_coeffs[ufcx_form.original_coefficient_positions[i]]
            f_re = fn_type(c.ufl_function_space()._cpp_object)
            f_re.x.array[:] = c.x.array.astype(alias_type_map[dtype])
            coeffs.append(f_re)
        constants = [c._cpp_object for c in form.constants()]

        integral_offsets = [ufcx_form.form_integral_offsets[i] for i in range(6)]
        subdomain_ids = {}
        for i in range(len(integral_offsets) - 1):
            integral_type = IntegralType(i)
            subdomain_ids[integral_type.name] = [
                ufcx_form.form_integral_ids[j]
                for j in range(integral_offsets[i], integral_offsets[i + 1])
            ]

        subdomains = {
            _ufl_to_dolfinx_domain[key]: get_integration_domains(
                _ufl_to_dolfinx_domain[key], subdomain_data[0], subdomain_ids[key]
            )
            for key, subdomain_data in sd[domain].items()
        }

        _entity_maps = [entity_map._cpp_object for entity_map in entity_maps] if entity_maps else []

        # TODO: Maybe available in basix?
        facet_shapes = {
            "triangle": "interval",
            "tetrahedron": "triangle",
            "quadrilateral": "interval",
            "hexahedron": "quadrilateral",
        }

        def get_integral_shape(cell_type: str, integral_type: IntegralType):
            if integral_type == IntegralType.cell:
                return cell_type
            if integral_type in (IntegralType.exterior_facet, IntegralType.interior_facet):
                assert cell_type in facet_shapes, f"Unsupported cell type: {cell_type}"
                return facet_shapes[cell_type]
            raise RuntimeError(f"Unsupported integral type: {integral_type}")

        cell_name = msh.topology.cell_type.name
        active_coeffs = np.arange(len(coeffs), dtype=np.int32)
        integrals = {}

        # integrals maps IntegralType to a list of tuples:
        # (integral_index, kernel_address, entities, active_coeffs)
        for integral_type_idx, (offset_start, offset_end) in enumerate(
            itertools.pairwise(integral_offsets)
        ):
            if offset_end <= offset_start:
                continue

            integral_type = IntegralType(integral_type_idx)
            integral_shape = get_integral_shape(cell_name, integral_type)
            integrals[integral_type] = []

            for idx in range(offset_start, offset_end):
                integral_id = ufcx_form.form_integral_ids[idx]

                # ufcx names kernels integral_{shape}_all or integral_{shape}_id{id}
                suffix = "all" if integral_id == -1 else f"id{integral_id}"
                integral_attr = f"integral_{integral_shape}_{suffix}"
                integral_class = getattr(ufcx_form, integral_attr, None)
                if integral_class is None:
                    raise RuntimeError(f"Integral class '{integral_attr}' not found")

                kernel_address = integral_class.tabulate_tensor_addr()

                if integral_id == -1:
                    tdim = (
                        msh.topology.dim
                        if integral_type == IntegralType.cell
                        else msh.topology.dim - 1
                    )
                    msh.topology.create_entities(tdim)
                    entities = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
                else:
                    entities = np.array(subdomains.get(integral_type, []), dtype=np.int32)

                integrals[integral_type].append((idx, kernel_address, entities, active_coeffs))

        return Form(ftype(V, integrals, coeffs, constants, False, _entity_maps, msh))

    def _zero_form(form):
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]
        assert V, "Form must have at least one argument"

        return Form(
            ftype(
                spaces=V,
                integrals={},
                coefficients=[],
                constants=[],
                need_permutation_data=False,
                entity_maps=[],
                mesh=V[0].mesh,
            )
        )

    def _create_form(form):
        if isinstance(form, ufl.Form):
            return _form(form)
        if isinstance(form, ufl.ZeroBaseForm):
            return _zero_form(form)
        if isinstance(form, collections.abc.Iterable):
            return [_create_form(sub_form) for sub_form in form]
        return form

    return _create_form(form)


def jit(form, form_compiler_options: dict | None = None):
    opts = get_options({"language": "ffcx_backends.cpp"})
    opts.update(form_compiler_options or {})

    cache_key = hashlib.sha1(
        ";".join([form.signature(), ffcx.__version__, repr(sorted(opts.items()))]).encode()
    ).hexdigest()

    if cache_key in _cppyy_form_cache:
        return _cppyy_form_cache[cache_key]

    decl = compile_ufl_objects([form], opts)[0][0]

    ufcx_path = pathlib.Path(ffcx.__file__).parent / "codegeneration" / "ufcx.h"
    assert ufcx_path.exists(), f"Path does not exist: {ufcx_path}"
    cppyy.add_include_path(str(ufcx_path.parent))

    # Include dolfiny's running_error.h (brings in mpfr.h context) and load libmpfr
    dolfiny_cpp_path = pathlib.Path(dolfiny.__file__).parent / "cpp" / "running_error.h"
    cppyy.include(str(dolfiny_cpp_path))
    cppyy.load_library("mpfr")

    # Extract the form class name (content-hash based, stable) to retrieve the
    # compiled template from cppyy and to build a unique alias replacing form__0.
    import re as _re

    m = _re.search(r"\bclass (form_[0-9a-f]{40})\b", decl)
    if m is None:
        raise RuntimeError("Could not locate form class name in generated C++ declaration")
    form_class_name = m.group(1)
    form_content_hash = form_class_name[len("form_") :]

    # Rename the form__0 template alias to form__<content-hash> so that
    # compiling different forms in the same process never redefines the same alias.
    decl = decl.replace("form__0", f"form__{form_content_hash}")
    cppyy.cppdef(decl)

    compiled_form = getattr(cppyy.gbl, form_class_name)
    _cppyy_form_cache[cache_key] = compiled_form

    return compiled_form
