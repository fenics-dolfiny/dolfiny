# mypy: disable-error-code="attr-defined"

import collections
import hashlib
import itertools
import logging
import os

from petsc4py import PETSc

import dolfinx
from ffcx.codegeneration.utils import empty_void_pointer

import cffi
import numba
import numpy as np

from dolfiny.utils import pprint

# ruff: noqa: E501

ffi = cffi.FFI()
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8),
    numba.types.CPointer(numba.types.void),
)
formtype = dolfinx.fem.form_cpp_class(PETSc.ScalarType)  # type: ignore

KernelData = collections.namedtuple(
    "KernelData",
    (
        "kernel",
        "array",
        "w",
        "c",
        "coords",
        "entity_local_index",
        "permutation",
        "constants",
        "coefficients",
    ),
)
ElementData = collections.namedtuple(
    "ElementData", ("indices", "name", "begin", "end", "stacked_begin", "stacked_end")
)
UserKernel = collections.namedtuple("UserKernel", ("name", "code", "required_J"))


@numba.cfunc(c_signature, nopython=True)
def do_nothing(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL, custom_data=ffi.NULL):
    pass


do_nothing_cffi = ffi.cast(
    "void(*)(double *, double *, double *, double *, int *, uint8_t *, void *)", do_nothing.address
)


class LocalSolver:
    def __init__(
        self,
        function_spaces,
        local_spaces_id,
        F_integrals,
        J_integrals,
        local_integrals,
        local_update,
    ):
        """Local solver used to eliminate local degrees-of-freedom.

        Local solver is defined with 4 callback functions:
        1. assemble tangent for global dofs, parameter ``J_integrals``,
        2. assemble residual for global dofs, parameter ``F_integrals``,
        3. reconstruct local dofs for a fixed, known global dofs, parameter ``local_integrals``,
        4. update local dofs (globally), parameter ``local_update``.

        It supports use of C++ (via cppyy JIT compilation) or Numba kernels.

        The interface for ``foo_integrals`` follows integral storage in dolfinx,
        for example

        ```python
        F_integrals = [{dolfinx.fem.IntegralType.cell:
                         [(-1, sc_F_cell, cell_entities)],
                         dolfinx.fem.IntegralType.exterior_facet:
                         [(1, sc_F_exterior_facet, exterior_facet_entities)]}]
        ```

        defines an integral ``sc_F_cell`` which will be assembled over cells, has default
        marker ID ``-1`` and cell entities given by ``cell_entities``,
        and integral ``sc_F_exterior_facet`` which is assembled over exterior facets which are
        marked ``1`` and associated with facet entities in ``exterior_facet_entities``.

        C++ kernels
        -----
        Recommended approach. User writes C++ code string which has access to
        pre-tabulated residuals F0, F1, ..., Fn and J00, J01, J10, ..., Jnn tangents.
        These are kernel_data_t types (see localsolver.h for details).

        Numba kernels
        -----
        Numba kernels should be used for smaller problems and testing purposes due to
        several performance limitations. User provides Numba jitted callback with access
        to residual and tangent data structures via passed arguments.

        """
        self.function_spaces = function_spaces
        self.local_spaces_id = local_spaces_id
        self.J_integrals = J_integrals
        self.F_integrals = F_integrals
        self.local_integrals = local_integrals
        self.local_update = local_update

        self.global_spaces_id = [
            i for i in range(len(function_spaces)) if i not in self.local_spaces_id
        ]

        # Initialize forms (and form-related info) to None.
        # At this point localsolver has no information about forms, these
        # get provided later once the localsolver is attached
        # to SNESProblem.
        self.F_ufc = None
        self.J_ufc = None
        self.F_ufl = None
        self.J_ufl = None
        self.stacked_coefficients = None
        self.stacked_constants = None
        self.coefficients = None
        self.constants = None

        self.active_coeffs = None

    def reduced_F_forms(self):
        """Return list of forms used to assemble reduced residuals."""
        F_form = []

        for gi, i in enumerate(self.global_spaces_id):
            V = self.function_spaces[i]
            integrals = self.F_integrals[gi]
            for celltype, integral in integrals.items():
                for k in range(len(integral)):
                    integral[k] = (
                        integral[k][0],
                        self.wrap_kernel(integral[k][1], (i,), celltype),
                        np.array(integral[k][2], dtype=np.int32),
                        self.active_coeffs,
                    )

            cppform = formtype(
                [V._cpp_object],
                integrals,
                [c[0] for c in self.stacked_coefficients],
                [c[0] for c in self.stacked_constants],
                False,
                {},
                mesh=V.mesh._cpp_object,
            )
            cppform = dolfinx.fem.Form(cppform)
            F_form += [cppform]

        return F_form

    def reduced_J_forms(self):
        """Return list of list of forms used to assemble reduced tangents."""
        J_form = [
            [None for j in range(len(self.global_spaces_id))]
            for i in range(len(self.global_spaces_id))
        ]

        for gi, i in enumerate(self.global_spaces_id):
            V0 = self.function_spaces[i]
            for gj, j in enumerate(self.global_spaces_id):
                V1 = self.function_spaces[j]
                integrals = self.J_integrals[gi][gj]
                for celltype, integral in integrals.items():
                    for k in range(len(integral)):
                        integral[k] = (
                            integral[k][0],
                            self.wrap_kernel(integral[k][1], (i, j), celltype),
                            np.array(integral[k][2], dtype=np.int32),
                            self.active_coeffs,
                        )

                J_form[gi][gj] = formtype(
                    [V0._cpp_object, V1._cpp_object],
                    integrals,
                    [c[0] for c in self.stacked_coefficients],
                    [c[0] for c in self.stacked_constants],
                    False,
                    {},
                    mesh=V0.mesh._cpp_object,
                )
                J_form[gi][gj] = dolfinx.fem.Form(J_form[gi][gj])

        return J_form

    def local_form(self):
        """Return list of forms used to 'assemble' into local state vectors."""
        local_form = []

        for li, i in enumerate(self.local_spaces_id):
            V = self.function_spaces[i]

            integrals = self.local_integrals[li]
            for celltype, integral in integrals.items():
                for k in range(len(integral)):
                    integral[k] = (
                        integral[k][0],
                        self.wrap_kernel(integral[k][1], (i,), celltype),
                        np.array(integral[k][2], dtype=np.int32),
                        self.active_coeffs,
                    )

            cppform = formtype(
                [V._cpp_object],
                integrals,
                [c[0] for c in self.stacked_coefficients],
                [c[0] for c in self.stacked_constants],
                False,
                {},
                mesh=V.mesh._cpp_object,
            )
            cppform = dolfinx.fem.Form(cppform)
            local_form += [cppform]

        return local_form

    def wrap_kernel(self, kernel, indices, celltype):
        if isinstance(kernel, UserKernel):
            return self.wrap_kernel_cpp(kernel, indices, celltype)
        elif isinstance(kernel, numba.core.registry.CPUDispatcher):
            return self.wrap_kernel_numba(kernel, indices, celltype)
        else:
            raise RuntimeError(f"Unrecognised kernel type {type(kernel)}")

    def wrap_kernel_numba(self, kernel, indices, celltype):
        logging.warning(
            f"Compiling Numba kernel for indices {indices}. "
            "This has significant overhead, consider using C++ kernels."
        )

        sizes = []
        for i, fs in enumerate(self.function_spaces):
            sizes += [fs.element.space_dimension]

        sizes = np.array(sizes, dtype=int)

        def fetch_tabulated_tensor_integrals(ufcx_form, celltype):
            integral_offsets = ufcx_form.form_integral_offsets
            num_integrals_celltype = (
                integral_offsets[celltype.value + 1] - integral_offsets[celltype.value]
            )
            if num_integrals_celltype > 0:
                tab_integrals_celltype = ufcx_form.form_integrals[
                    integral_offsets[celltype.value]
                ].tabulate_tensor_float64
            else:
                tab_integrals_celltype = do_nothing_cffi
            return tab_integrals_celltype

        # Numba does not support "list" of fn pointers, must be "tuple"
        F_fn = tuple(
            fetch_tabulated_tensor_integrals(self.F_ufc[i].ufcx_form, celltype)
            for i in range(len(sizes))
        )
        J_fn = tuple(
            tuple(
                fetch_tabulated_tensor_integrals(self.J_ufc[i][j].ufcx_form, celltype)
                for j in range(len(sizes))
            )
            for i in range(len(sizes))
        )

        shape = (sizes[indices[0]], 1)
        if len(indices) == 2:
            shape = (sizes[indices[0]], sizes[indices[1]])

        stacked_coefficients_size = (
            self.stacked_coefficients[-1][1][1] if len(self.stacked_coefficients) > 0 else 0
        )
        stacked_constants_size = (
            self.stacked_constants[-1][1][1] if len(self.stacked_constants) > 0 else 0
        )

        num_coordinate_dofs = self.function_spaces[0].mesh.geometry.dofmap.shape[1]

        # Extract number of coeffs/consts for later use in compile-time branch
        # elimination
        num_constants = len(self.constants)
        num_coefficients = len(self.coefficients)

        # Weirdly, Numba cannot deduce these if accessed directly below,
        # need to force their evaluation beforehand
        constants = self.constants
        coefficients = self.coefficients

        @numba.cfunc(c_signature, nopython=True)
        def wrapped_kernel(
            A_, w_, c_, coords_, entity_local_index_, permutation_=ffi.NULL, custom_data=ffi.NULL
        ):
            A = numba.carray(A_, shape, dtype=PETSc.ScalarType)
            w = numba.carray(w_, (stacked_coefficients_size,), dtype=PETSc.ScalarType)
            c = numba.carray(c_, (stacked_constants_size,), dtype=PETSc.ScalarType)
            coords = numba.carray(coords_, (num_coordinate_dofs * 3,), dtype=np.double)
            entity_local_index = numba.carray(entity_local_index_, (1,), dtype=np.int32)
            permutation = numba.carray(permutation_, (1,), dtype=np.uint8)

            J = numba.typed.List()
            F = numba.typed.List()

            for i in range(len(sizes)):
                if num_constants > 0:
                    consts = [const for const in constants if const.indices == (i, -1)]
                    c_array = np.zeros(
                        (max([const.stacked_end for const in consts]) if len(consts) > 0 else 1,),
                        dtype=PETSc.ScalarType,
                    )
                else:
                    consts = [None]
                    c_array = np.zeros((1,), dtype=PETSc.ScalarType)

                if num_coefficients > 0:
                    coeffs = [coeff for coeff in coefficients if coeff.indices == (i, -1)]
                    w_array = np.zeros(
                        (max([coeff.stacked_end for coeff in coeffs]) if len(coeffs) > 0 else 1,),
                        dtype=PETSc.ScalarType,
                    )
                else:
                    coeffs = [None]
                    w_array = np.zeros((1,), dtype=PETSc.ScalarType)

                pos = 0
                if num_coefficients > 0 and len(coeffs) > 0:
                    for coeff in coeffs:
                        size = coeff.stacked_end - coeff.stacked_begin
                        w_array[pos : pos + size] = w[coeff.stacked_begin : coeff.stacked_end]
                        pos += size

                pos = 0
                if num_constants > 0 and len(consts) > 0:
                    for const in consts:
                        size = const.stacked_end - const.stacked_begin
                        c_array[pos : pos + size] = c[const.stacked_begin : const.stacked_end]
                        pos += size

                F_array = np.zeros((sizes[i], 1), dtype=PETSc.ScalarType)
                F_fn[i](
                    ffi.from_buffer(F_array),
                    ffi.from_buffer(w_array),
                    ffi.from_buffer(c_array),
                    ffi.from_buffer(coords),
                    ffi.from_buffer(entity_local_index),
                    ffi.from_buffer(permutation),
                    empty_void_pointer(),
                )

                F.append(
                    KernelData(
                        F_fn[i],
                        F_array,
                        w_array,
                        c_array,
                        coords,
                        entity_local_index,
                        permutation,
                        consts,
                        coeffs,
                    )
                )

                J_row = numba.typed.List()
                for j in range(len(sizes)):
                    if num_constants > 0:
                        consts = [const for const in constants if const.indices == (i, j)]
                        c_array = np.zeros(
                            (
                                max([const.stacked_end for const in consts])
                                if len(consts) > 0
                                else 1,
                            ),
                            dtype=PETSc.ScalarType,
                        )
                    else:
                        consts = [None]
                        c_array = np.zeros((1,), dtype=PETSc.ScalarType)

                    if num_coefficients > 0:
                        coeffs = [coeff for coeff in coefficients if coeff.indices == (i, j)]
                        w_array = np.zeros(
                            (
                                max([coeff.stacked_end for coeff in coeffs])
                                if len(coeffs) > 0
                                else 1,
                            ),
                            dtype=PETSc.ScalarType,
                        )
                    else:
                        coeffs = [None]
                        w_array = np.zeros((1,), dtype=PETSc.ScalarType)

                    # Copy coefficient data to local vector for this kernel
                    if num_coefficients > 0:
                        pos = 0
                        for coeff in coeffs:
                            size = coeff.stacked_end - coeff.stacked_begin
                            w_array[pos : pos + size] = w[coeff.stacked_begin : coeff.stacked_end]
                            pos += size

                    if num_constants > 0:
                        pos = 0
                        for const in consts:
                            size = const.stacked_end - const.stacked_begin
                            c_array[pos : pos + size] = c[const.stacked_begin : const.stacked_end]
                            pos += size

                    J_array = np.zeros((sizes[i], sizes[j]), dtype=PETSc.ScalarType)
                    J_fn[i][j](
                        ffi.from_buffer(J_array),
                        ffi.from_buffer(w_array),
                        ffi.from_buffer(c_array),
                        ffi.from_buffer(coords),
                        ffi.from_buffer(entity_local_index),
                        ffi.from_buffer(permutation),
                        empty_void_pointer(),
                    )
                    J_row.append(
                        KernelData(
                            J_fn[i][j],
                            J_array,
                            w_array,
                            c_array,
                            coords,
                            entity_local_index,
                            permutation,
                            consts,
                            coeffs,
                        )
                    )
                J.append(J_row)

            # Execute user kernel
            kernel(A, J, F)

        return int(wrapped_kernel.address)

    def wrap_kernel_cpp(self, kernel: UserKernel, indices, celltype):
        import cppyy
        import cppyy.ll

        cppyy.add_include_path("/usr/include/eigen3/")
        if getattr(cppyy.gbl, "kernel_data_t", None) is None:
            # Include header only once
            cppyy.include(os.path.join(os.path.dirname(__file__), "localsolver.h"))

        sizes = []
        for i, fs in enumerate(self.function_spaces):
            sizes += [fs.element.space_dimension]

        shape = (sizes[indices[0]], 1)
        if len(indices) == 2:
            shape = (sizes[indices[0]], sizes[indices[1]])

        code = ""
        num_coordinate_dofs = self.function_spaces[0].mesh.geometry.dofmap.shape[1]

        alloc_code = ""
        copy_code = ""

        for i in range(len(sizes)):
            # Find constants and coefficients required for (i, -1) block kernel
            consts = [const for const in self.constants if const.indices == (i, -1)]
            coeffs = [coeff for coeff in self.coefficients if coeff.indices == (i, -1)]

            ufcx_form = self.F_ufc[i].ufcx_form
            integral_offsets = ufcx_form.form_integral_offsets
            num_integrals_celltype = (
                integral_offsets[celltype.value + 1] - integral_offsets[celltype.value]
            )

            if num_integrals_celltype > 0:
                F_fn = ufcx_form.form_integrals[
                    integral_offsets[celltype.value]
                ].tabulate_tensor_float64

                alloc_code += f"""
                auto kernel_F{i} = (ufc_kernel_t){int(ffi.cast("intptr_t", F_fn))};
                """

                alloc_code += f"""
                kernel_data_t F{i} = {{
                    kernel_F{i},
                    Eigen::Matrix<double, {sizes[i]}, 1>(),
                    Eigen::Array<double,
                    {max([coeff.stacked_end for coeff in coeffs]) if len(coeffs) > 0 else 0}, 1>(),
                    Eigen::Array<double,
                    {max([const.stacked_end for const in consts]) if len(consts) > 0 else 0}, 1>(),
                    Eigen::Array<double, {num_coordinate_dofs * 3}, 1>(),
                    Eigen::Array<int32_t, 1, 1>(),
                    Eigen::Array<uint8_t, 1, 1>()
                }};
                """

                pos = 0
                for coeff in coeffs:
                    size = coeff.stacked_end - coeff.stacked_begin
                    copy_code += f"""
                    for (int j = 0; j < {size}; ++j){{
                        F{i}.w[{pos} + j] = w_[{coeff.stacked_begin} + j];
                    }}
                    """
                    pos += size

                pos = 0
                for const in consts:
                    size = const.stacked_end - const.stacked_begin
                    copy_code += f"""
                    for (int j = 0; j < {size}; ++j){{
                        F{i}.c[{pos} + j] = c_[{const.stacked_begin} + j];
                    }}
                    """
                    pos += size

                copy_code += f"""
                for (int k=0; k<{3 * num_coordinate_dofs}; ++k)
                {{
                    F{i}.coords[k] = coords_[k];
                }}
                """

                if celltype in [
                    dolfinx.fem.IntegralType.interior_facet,
                    dolfinx.fem.IntegralType.exterior_facet,
                ]:
                    copy_code += f"""
                    F{i}.entity_local_index[0] = eli_[0];
                    """

                copy_code += f"""
                F{i}.array.setZero();
                F{i}.kernel(F{i}.array.data(), F{i}.w.data(), F{i}.c.data(),
                            F{i}.coords.data(), F{i}.entity_local_index.data(),
                            F{i}.permutation.data(), nullptr);
                """

            for j in range(len(sizes)):
                if kernel.required_J is not None and (i, j) not in kernel.required_J:
                    continue
                consts = [const for const in self.constants if const.indices == (i, j)]
                coeffs = [coeff for coeff in self.coefficients if coeff.indices == (i, j)]

                ufcx_form = self.J_ufc[i][j].ufcx_form
                integral_offsets = ufcx_form.form_integral_offsets
                num_integrals_celltype = (
                    integral_offsets[celltype.value + 1] - integral_offsets[celltype.value]
                )

                if self.J_ufc[i][j] is not None and num_integrals_celltype > 0:
                    J_fn = ufcx_form.form_integrals[
                        integral_offsets[celltype.value]
                    ].tabulate_tensor_float64
                    alloc_code += f"""
                    auto kernel_J{i}{j} = (ufc_kernel_t){int(ffi.cast("intptr_t", J_fn))};
                    """

                    alloc_code += f"""
                    kernel_data_t J{i}{j} = {{
                        kernel_J{i}{j},
                        Eigen::Matrix<double, {sizes[i]}, {sizes[j]}>(),
                        Eigen::Array<double,
                        {max([coeff.stacked_end for coeff in coeffs]) if len(coeffs) > 0 else 0}, 1>(),
                        Eigen::Array<double,
                        {max([const.stacked_end for const in consts]) if len(consts) > 0 else 0}, 1>(),
                        Eigen::Array<double, {num_coordinate_dofs * 3}, 1>(),
                        Eigen::Array<int32_t, 1, 1>(),
                        Eigen::Array<uint8_t, 1, 1>()
                    }};
                    """

                    pos = 0
                    for coeff in coeffs:
                        size = coeff.stacked_end - coeff.stacked_begin
                        copy_code += f"""
                        for (int j = 0; j < {size}; ++j){{
                            J{i}{j}.w[{pos} + j] = w_[{coeff.stacked_begin} + j];
                        }}
                        """
                        pos += size

                    pos = 0
                    for const in consts:
                        size = const.stacked_end - const.stacked_begin
                        copy_code += f"""
                        for (int j = 0; j < {size}; ++j){{
                            J{i}{j}.c[{pos} + j] = c_[{const.stacked_begin} + j];
                        }}
                        """
                        pos += size

                    copy_code += f"""
                    for (int k=0; k<{3 * num_coordinate_dofs}; ++k)
                    {{
                        J{i}{j}.coords[k] = coords_[k];
                    }}
                    """

                    if celltype in [
                        dolfinx.fem.IntegralType.interior_facet,
                        dolfinx.fem.IntegralType.exterior_facet,
                    ]:
                        copy_code += f"""
                        J{i}{j}.entity_local_index[0] = eli_[0];
                        """

                    copy_code += f"""
                    // Cast UFC kernel address to callable kernel and execute
                    J{i}{j}.array.setZero();
                    J{i}{j}.kernel(J{i}{j}.array.data(), J{i}{j}.w.data(), J{i}{j}.c.data(),
                                   J{i}{j}.coords.data(), J{i}{j}.entity_local_index.data(),
                                   J{i}{j}.permutation.data(), nullptr);
                    """

        code += f"""
        // Allocate data structures with memory shared across all cells
        {alloc_code}

        {kernel.code}

        void kernel(double* __restrict__ A_, const double* __restrict__ w_,
                    const double* __restrict__ c_, const double* __restrict__ coords_,
                    void* __restrict__ eli_null, void* __restrict__ perm_null, void* custom_data_)
        {{
            // This is a hack for cppyy which cannot implicitly cast nullptr
            // passed for cell integrals into int32_t*
            auto eli_ = static_cast<const int32_t* >(eli_null);
            auto perm_ = static_cast<const uint8_t*>(perm_null);

            auto Arr = Eigen::Map<Eigen::Matrix<double, {shape[0]}, {shape[1]}>>(A_);

            // Copy coefficients and constants
            {copy_code}

            // Execute user kernel
            {kernel.name}(Arr);
        }};
        """

        # Compute caching hash and wrap in a namespace
        hsh = hashlib.sha1(code.encode("utf-8")).hexdigest()

        name = f"wrapped_{kernel.name}_{hsh}"
        code = f"""
        namespace {name} {{
            {code}
        }}
        """

        cppyy.cppdef(code)
        compiled_wrapper = getattr(cppyy.gbl, name)

        return cppyy.ll.cast["intptr_t"](compiled_wrapper.kernel)

    def stack_data(self):
        """Stack all Coefficients and Constants across all blocks in F and J forms."""
        stacked_coefficients = []
        stacked_constants = []
        coefficients = []
        constants = []

        # Stack all compiled and UFL forms
        stacked_forms_ufc = self.F_ufc.copy()
        stacked_forms_ufc += list(itertools.chain(*self.J_ufc))
        stacked_forms_ufc = [form.ufcx_form for form in stacked_forms_ufc if form is not None]

        stacked_forms_ufl = self.F_ufl.copy()
        stacked_forms_ufl += list(itertools.chain(*self.J_ufl))
        stacked_forms_ufl = [form for form in stacked_forms_ufl if form is not None]

        size = 0
        size_const = 0

        for i, form_ufl in enumerate(stacked_forms_ufl):
            original_coefficients = form_ufl.coefficients()
            for j in range(stacked_forms_ufc[i].num_coefficients):
                coeff = original_coefficients[
                    stacked_forms_ufc[i].original_coefficient_positions[j]
                ]._cpp_object
                if coeff not in [c[0] for c in stacked_coefficients]:
                    stacked_coefficients.append(
                        (coeff, (size, size + coeff.function_space.element.space_dimension))
                    )
                    size += coeff.function_space.element.space_dimension
            for const in form_ufl.constants():
                const = const._cpp_object
                if const not in [c[0] for c in stacked_constants]:
                    stacked_constants.append((const, (size_const, size_const + const.value.size)))
                    size_const += const.value.size

        for i in range(len(self.F_ufl)):
            if self.F_ufl[i] is None:
                continue
            original_coefficients = self.F_ufl[i].coefficients()

            coeff_pos = 0
            for c in range(self.F_ufc[i].ufcx_form.num_coefficients):
                coeff = original_coefficients[
                    self.F_ufc[i].ufcx_form.original_coefficient_positions[c]
                ]._cpp_object
                coeff_stacked_pos = next(
                    (c[1][0], c[1][1]) for c in stacked_coefficients if c[0] == coeff
                )
                coeff_size = coeff_stacked_pos[1] - coeff_stacked_pos[0]
                dat = ElementData(
                    (i, -1),
                    coeff.name,
                    coeff_pos,
                    coeff_pos + coeff_size,
                    coeff_stacked_pos[0],
                    coeff_stacked_pos[1],
                )
                coefficients.append(dat)
                coeff_pos += coeff_size

            const_pos = 0
            for const in self.F_ufl[i].constants():
                const_stacked_pos = next(
                    (c[1][0], c[1][1]) for c in stacked_constants if c[0] == const._cpp_object
                )
                const_size = const_stacked_pos[1] - const_stacked_pos[0]
                dat = ElementData(
                    (i, -1),
                    str(const),
                    const_pos,
                    const_pos + const_size,
                    const_stacked_pos[0],
                    const_stacked_pos[1],
                )
                constants.append(dat)
                const_pos += const_size

            for j in range(len(self.F_ufl)):
                if self.J_ufl[i][j] is None:
                    continue
                original_coefficients = self.J_ufl[i][j].coefficients()

                coeff_pos = 0
                for c in range(self.J_ufc[i][j].ufcx_form.num_coefficients):
                    coeff = original_coefficients[
                        self.J_ufc[i][j].ufcx_form.original_coefficient_positions[c]
                    ]._cpp_object
                    coeff_stacked_pos = next(
                        (c[1][0], c[1][1]) for c in stacked_coefficients if c[0] == coeff
                    )
                    coeff_size = coeff_stacked_pos[1] - coeff_stacked_pos[0]
                    dat = ElementData(
                        (i, j),
                        coeff.name,
                        coeff_pos,
                        coeff_pos + coeff_size,
                        coeff_stacked_pos[0],
                        coeff_stacked_pos[1],
                    )
                    coefficients.append(dat)
                    coeff_pos += coeff_size

                const_pos = 0
                for const in self.J_ufl[i][j].constants():
                    const_stacked_pos = next(
                        (c[1][0], c[1][1]) for c in stacked_constants if c[0] == const._cpp_object
                    )
                    const_size = const_stacked_pos[1] - const_stacked_pos[0]
                    dat = ElementData(
                        (i, j),
                        str(const),
                        const_pos,
                        const_pos + const_size,
                        const_stacked_pos[0],
                        const_stacked_pos[1],
                    )
                    constants.append(dat)
                    const_pos += const_size

        self.stacked_coefficients = stacked_coefficients
        self.stacked_constants = stacked_constants
        self.coefficients = tuple(coefficients)
        self.constants = tuple(constants)

        self.active_coeffs = np.arange(len(self.stacked_coefficients), dtype=np.int8)

    def view(self):
        """Shows information about kernels, sizes of blocks and positions of Coefficient DOFs."""
        pprint(79 * "#")
        pprint(79 * "*")
        for i in range(len(self.F_ufl)):
            rows = self.F_ufc[i].function_spaces[0].element.space_dimension
            pprint(f"F{i} ({rows}):")
            coeffs = [coeff for coeff in self.coefficients if coeff.indices == (i, -1)]
            for coeff in coeffs:
                pprint(f"\t{coeff.name} \t [{coeff.begin}, {coeff.end}]")
            pprint()

        for i in range(len(self.F_ufl)):
            for j in range(len(self.F_ufl)):
                form = self.J_ufc[i][j]
                if form is None:
                    continue
                rows = self.J_ufc[i][j].function_spaces[0].element.space_dimension
                cols = self.J_ufc[i][j].function_spaces[1].element.space_dimension
                pprint(f"J{i}{j} ({rows}, {cols}):")
                coeffs = [coeff for coeff in self.coefficients if coeff.indices == (i, j)]
                for coeff in coeffs:
                    pprint(f"\t{coeff.name} \t [{coeff.begin}, {coeff.end}]")
                pprint()

        pprint(79 * "*")
        pprint(79 * "#")
