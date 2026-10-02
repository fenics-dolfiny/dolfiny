from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import pathlib
import re
import typing
from collections.abc import Sequence

import dolfinx
import dolfinx.cpp.la
import ffcx
import ufl
from dolfinx.fem import DirichletBC, Form, IntegralType
from dolfinx.fem.assemble import _bc_dof_markers, _owned_marked_rows
from dolfinx.fem.forms import _ufl_to_dolfinx_domain, get_integration_domains
from dolfinx.mesh import _mesh_from_ufl_domain
from ffcx.compiler import compile_ufl_objects
from ffcx.options import get_options

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

# Reduced-precision dofs on an fp64 mesh; dolfinx only instantiates matched pairs.
dolfinx.fem.DirichletBC.cpp_types[np.dtype(np.float16), np.dtype(np.float64)] = (
    dolfiny.cpp._cpp.DirichletBC_float16_float64
)
dolfinx.fem.DirichletBC.cpp_types[np.dtype(np.float32), np.dtype(np.float64)] = (
    dolfiny.cpp._cpp.DirichletBC_float32_float64
)
dolfinx.fem.Function.cpp_types[np.dtype(np.float16), np.dtype(np.float64)] = (
    dolfiny.cpp._cpp.Function_float16_float64
)
dolfinx.fem.Function.cpp_types[np.dtype(np.float32), np.dtype(np.float64)] = (
    dolfiny.cpp._cpp.Function_float32_float64
)

# Scalar dtypes of the running-error kernels.
re_dtypes = set(np.dtype(t) for t in (np.float64, np.float32, np.float16))

# Mesh geometry type. dolfinx and basix only instantiate double and float.
nptype_to_cpp = {np.float64: "double", np.float32: "float"}

# Opaque numpy carrier for a packed pair, of matching width. fp16 uses float32:
# numpy has no complex32.
nptype_to_carrier = {np.float64: np.complex128, np.float32: np.complex64, np.float16: np.float32}


def _packed_dtype(T: npt.DTypeLike) -> np.dtype:
    """Running-error scalar layout: value ``T`` at 0, error ``T`` at ``sizeof(T)``."""
    itemsize = np.dtype(T).itemsize
    return np.dtype(
        {
            "names": ["val", "err"],
            "formats": [T, T],
            "offsets": [0, itemsize],
            "itemsize": 2 * itemsize,
        }
    )


def _fill_packed(arr: np.ndarray, values: npt.ArrayLike, T: npt.DTypeLike, err=0) -> None:
    """Seed a packed running-error buffer with ``values`` and their error ``err``."""
    view = arr.view(_packed_dtype(T))
    view["val"] = values
    view["err"] = err


def split(arr: np.ndarray, T: npt.DTypeLike) -> tuple[np.ndarray, np.ndarray]:
    """Split a packed running-error array into value and error arrays, both ``T``."""
    s = arr.view(_packed_dtype(T))
    return s["val"].copy(), s["err"].copy()  # contiguous, decoupled


# Cache is used to avoid re-compiling the same form multiple times,
# since cppjit does not recognize that two identical declarations are
# the same and will raise a redefinition error.
_cppjit_form_cache: dict[str, typing.Any] = {}


def pack_coefficients(
    form: Form, errors: dict | None = None
) -> dict[tuple[IntegralType, int], npt.NDArray]:
    """Pack coefficients, re-seeding the packed mirrors from their sources first.

    ``errors`` maps a source Function to the error of its dofs, zero if not listed.
    """
    errors = errors or {}
    for src, mirror in zip(form._re_src_coefficients, form._cpp_object.coefficients, strict=True):  # type: ignore[attr-defined]
        _fill_packed(mirror.x.array, src.x.array, form._re_scalar_dtype, errors.get(src, 0))  # type: ignore[attr-defined]
    return _pack_coefficients(form._cpp_object)  # type: ignore


def pack_constants(form: Form, errors: dict | None = None) -> npt.NDArray:
    """Pack constants, re-seeding the packed mirrors from their sources first.

    ``errors`` maps a source Constant to the error of its value, zero if not listed.
    """
    errors = errors or {}
    for src, mirror in zip(form._re_src_constants, form._cpp_object.constants, strict=True):  # type: ignore[attr-defined]
        _fill_packed(mirror.value, src.value, form._re_scalar_dtype, errors.get(src, 0))  # type: ignore[attr-defined]
    return _pack_constants(form._cpp_object)  # type: ignore


def assemble_scalar(M: Form, errors: dict | None = None) -> typing.Any:
    constants = pack_constants(M, errors)
    coeffs = pack_coefficients(M, errors)
    val = dolfiny.cpp._cpp.assemble_scalar(M._cpp_object, constants, coeffs)

    return val.val, val.err


def assemble_vector(L: Form, errors: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    constants = pack_constants(L, errors)
    coeffs = pack_coefficients(L, errors)
    V = L.function_spaces[0]
    b = np.zeros(
        (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs,
        dtype=L.dtype,
    )

    dolfiny.cpp._cpp.assemble_vector(b, L._cpp_object, constants, coeffs)

    return split(b, L._re_scalar_dtype)  # type: ignore[attr-defined]


@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: Sequence[DirichletBC] | None = None,
    diag: complex = 1.0 + 0j,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
    block_mode: dolfinx.la.BlockMode | None = None,
    errors: dict | None = None,
) -> tuple[typing.Any, typing.Any]:
    """Assemble bilinear form into value and error matrices."""
    bcs = [] if bcs is None else bcs

    sp = dolfiny.cpp._cpp.create_sparsity_pattern(a._cpp_object)
    sp.finalize()
    block_mode = dolfinx.la.BlockMode.compact if block_mode is None else block_mode

    packed_type = getattr(dolfiny.cpp._cpp, f"MatrixCSR_{a._re_scalar_tag}")
    A: dolfinx.la.MatrixCSR = packed_type(sp, block_mode)
    _assemble_matrix_csr(A, a, bcs, diag, constants, coeffs, errors)

    # Value and error matrices, both in the studied dtype (fp16 only from dolfiny).
    dtype = a._re_scalar_dtype
    csr_type = getattr(dolfinx.cpp.la, f"MatrixCSR_{dtype.name}", None) or getattr(
        dolfiny.cpp._cpp, f"MatrixCSR_{dtype.name}"
    )
    A_val = csr_type(sp, block_mode)
    A_err = csr_type(sp, block_mode)
    packed = A.data.view(_packed_dtype(dtype))  # no copy; the assignments below are the copy
    A_val.data[:] = packed["val"]
    A_err.data[:] = packed["err"]
    return A_val, A_err


@assemble_matrix.register
def _assemble_matrix_csr(
    A: dolfinx.la.MatrixCSR,
    a: Form,
    bcs: Sequence[DirichletBC] | None = None,
    diag: complex = 1.0 + 0j,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
    errors: dict | None = None,
) -> typing.Any:
    """Assemble bilinear form into a matrix."""
    if constants is None:
        constants = pack_constants(a, errors)

    if coeffs is None:
        coeffs = pack_coefficients(a, errors)

    V0, V1 = a.function_spaces
    dof_marker0 = _bc_dof_markers(V0, bcs)
    dof_marker1 = _bc_dof_markers(V1, bcs)
    dolfiny.cpp._cpp.assemble_matrix(A, a._cpp_object, constants, coeffs, dof_marker0, dof_marker1)

    # If matrix is a 'diagonal' block, set diagonal entry for constrained
    # dofs. Insert, not add: adding to the zeroed entry would accrue a rounding error.
    diag = a._re_scalar_type(diag.real, diag.imag)  # type: ignore[attr-defined]
    if V0._cpp_object is V1._cpp_object:
        dolfiny.cpp._cpp.set_diagonal(
            A, _owned_marked_rows(V0, dof_marker0), diag, dolfinx.la.InsertMode.insert
        )
    return A


def form(
    form: ufl.Form | Sequence[ufl.Form] | Sequence[Sequence[ufl.Form]],
    dtype: npt.DTypeLike = np.float64,
    form_compiler_options: dict | None = None,
    entity_maps: Sequence[_EntityMap] | None = None,
    mode: str = "worst",
):
    form_compiler_options = form_compiler_options or {}

    # Normalized once, so that both np.float64 and np.dtype("float64") are accepted.
    dtype = np.dtype(dtype)
    if dtype not in re_dtypes:
        supported = sorted(t.name for t in re_dtypes)
        raise NotImplementedError(
            f"Unsupported dtype: {dtype.name}. Only {supported} are implemented."
        )

    if mode not in ("worst", "exact"):
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'worst' or 'exact'.")

    # (mode, dtype) picks one C++ instantiation, spelled twice: cpp_type as source text for
    # the form compiler, scalar_tag as the name bindings.cpp exported the bindings under.
    _cpp = dolfiny.cpp._cpp
    scalar_cpp = f"std::{dtype.name}_t"
    cpp_type = (
        f"running_error::running_error_t<{scalar_cpp}, running_error::ErrorMode::{mode.upper()}>"
    )
    scalar_tag = f"re_{mode}_{dtype.name}"
    # The scalar class bound in bindings.cpp, e.g. re_worst_float64 -> ReWorstFloat64.
    scalar_type = getattr(_cpp, "".join(part.capitalize() for part in scalar_tag.split("_")))

    def _re_types(msh):
        """Running-error C++ types for scalar ``dtype`` on this mesh geometry."""
        geometry_dtype = np.dtype(msh.geometry.x.dtype)
        # Form/Function are templated on geometry too: Form_re_worst_float64_float64,
        # against a scalar-only Constant_re_worst_float64.
        suffix = f"{scalar_tag}_{geometry_dtype.name}"
        try:
            ftype = getattr(_cpp, f"Form_{suffix}")
            fn_type = getattr(_cpp, f"Function_{suffix}")
            constant_type = getattr(_cpp, f"Constant_{scalar_tag}")
        except AttributeError:
            raise NotImplementedError(
                f"Running error not available for scalar "
                f"'{dtype.name}' on geometry '{geometry_dtype.name}'."
            ) from None
        return ftype, fn_type, constant_type, nptype_to_cpp[geometry_dtype.type]

    def _tagged(cpp_form, msh, V, src_coefficients=(), src_constants=()):
        """Wrap the C++ form, recording the scalar hidden by the packed ``Form.dtype``."""
        f = Form(cpp_form, msh, V)
        # Suffix of the bound C++ instantiations, for lookups: MatrixCSR_re_worst_float64.
        f._re_scalar_tag = scalar_tag
        # Studied precision, one half of a packed pair: split(b, float16) -> two fp16 arrays.
        f._re_scalar_dtype = dtype
        # Packed scalar class, to build a value/error pair: ReWorstFloat64(1.0, 0.0).
        f._re_scalar_type = scalar_type
        # Sources the mirrors are re-seeded from, in the order the mirrors are stored.
        f._re_src_coefficients = list(src_coefficients)
        f._re_src_constants = list(src_constants)
        return f

    def _form(form):
        sd = form.subdomain_data()
        (domain,) = sd.keys()

        for data in sd[domain].values():
            non_none = [d for d in data if d is not None]
            assert len(non_none) == 0 or all(d is non_none[0] for d in non_none)

        msh = _mesh_from_ufl_domain(domain)

        ftype, fn_type, constant_type, geom_cpp = _re_types(msh)

        ufcx_form_template = jit(form, form_compiler_options=form_compiler_options)
        ufcx_form = ufcx_form_template[cpp_type, geom_cpp]

        V = [arg.ufl_function_space() for arg in form.arguments()]
        if form_compiler_options.get("part", "full") == "diagonal":
            V = [V[0]]

        # Mirrors are left zeroed here, pack_coefficients/pack_constants seed them from the
        # sources on every assembly. ufcx orders coefficients its own way, the sources are
        # kept in that same order so the two lists pair up positionally.
        original_coeffs = form.coefficients()
        src_coefficients, coeffs = [], []
        for i in range(ufcx_form.num_coefficients):
            c = original_coeffs[ufcx_form.original_coefficient_positions[i]]
            src_coefficients.append(c)
            coeffs.append(fn_type(c.ufl_function_space()._cpp_object))

        src_constants = list(form.constants())
        constants = [
            constant_type(np.zeros_like(c._cpp_object.value, dtype=nptype_to_carrier[dtype.type]))
            for c in src_constants
        ]

        n_integral_types = len(IntegralType)
        assert n_integral_types == 5
        integral_offsets = [ufcx_form.form_integral_offsets[i] for i in range(n_integral_types + 1)]
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

        return _tagged(
            ftype(
                [_V._cpp_object for _V in V],
                integrals,
                coeffs,
                constants,
                False,
                _entity_maps,
                msh._cpp_object,
            ),
            msh,
            V,
            src_coefficients,
            src_constants,
        )

    def _zero_form(form):
        V = [arg.ufl_function_space() for arg in form.arguments()]
        assert V, "Form must have at least one argument"
        msh = V[0].mesh

        ftype, _, _, _ = _re_types(msh)
        return _tagged(
            ftype(
                spaces=[_V._cpp_object for _V in V],
                integrals={},
                coefficients=[],
                constants=[],
                need_permutation_data=False,
                entity_maps=[],
                mesh=msh._cpp_object,
            ),
            msh,
            V,
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

    # Also the salt below. cppdef is not idempotent, so each distinct declaration must be
    # defined exactly once, under names no other declaration uses.
    cache_key = hashlib.sha1(
        ";".join([form.signature(), ffcx.__version__, repr(sorted(opts.items()))]).encode()
    ).hexdigest()

    if cache_key in _cppjit_form_cache:
        return _cppjit_form_cache[cache_key]

    import cppjit

    decl = compile_ufl_objects([form], opts)[0][0]

    ufcx_path = pathlib.Path(ffcx.__file__).parent / "codegeneration" / "ufcx.h"
    assert ufcx_path.exists(), f"Path does not exist: {ufcx_path}"
    cppjit.add_include_path(str(ufcx_path.parent))

    dolfiny_cpp_path = pathlib.Path(dolfiny.cpp.get_include()) / "running_error.h"
    assert dolfiny_cpp_path.exists(), f"Path does not exist: {dolfiny_cpp_path}"
    cppjit.include(str(dolfiny_cpp_path))

    # ffcx hashes the form, not the options, so scalar_geometry=True yields a different
    # body under the same names. Salt every hash to keep the variants apart:
    # integral_60092046..._triangle -> integral_<sha1("60092046...;<cache_key>")>_triangle.
    decl = re.sub(
        r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])",
        lambda h: hashlib.sha1(f"{h.group(0)};{cache_key}".encode()).hexdigest(),
        decl,
    )

    m = re.search(r"\bclass (form_[0-9a-f]{40})\b", decl)
    if m is None:
        raise RuntimeError("Could not locate form class name in generated C++ declaration")
    form_class_name = m.group(1)

    # Drop ffcx's file-level alias.
    decl, n = re.subn(
        r"\n*// Alias name\s*template <typename T, typename U>\s*using form__\w+ =[^;]*;",
        "",
        decl,
    )
    if n != 1:
        raise RuntimeError(f"Expected one form alias in the generated declaration, found {n}")

    # #pragma once has no real file to track and crashes the interpreter
    decl = decl.replace("#pragma once", "")
    cppjit.cppdef(decl)

    compiled_form = getattr(cppjit.gbl, form_class_name)
    _cppjit_form_cache[cache_key] = compiled_form

    return compiled_form
