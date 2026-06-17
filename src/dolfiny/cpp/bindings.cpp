// clang-format off
#include "running_error.h"
// clang-format on

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx_wrappers/assemble.h>
#include <dolfinx_wrappers/fem.h>
#include <dolfinx_wrappers/la.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <ufcx.h>

#include <format>

using re_worst_double_t = dolfiny::re_worst_t<double>;
using re_exact_double_t = dolfiny::re_exact_t<double>;

// --- dolfinx traits for running_error_t<double, ErrorMode::WORST> ---

template <>
struct dolfinx::is_custom_scalar<re_worst_double_t> : std::true_type {};

template <>
struct dolfinx::scalar_value<re_worst_double_t> {
  using type = double;
};

template <>
struct dolfinx::MPI::mpi_type_mapping<re_worst_double_t> {
  static inline MPI_Datatype type = []() {
    MPI_Datatype dt;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }();
};

namespace nanobind::detail {
// Overcome DLPack limitations for custom scalar types.
// Since our custom type consists of two doubles, we pack it as a complex128
// to facilitate array interoperability and data exchange with Python.
template <>
struct dtype_traits<re_worst_double_t> {
  static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Complex,
                                       128, 1};
  static constexpr auto name = const_name("complex128");
};
}  // namespace nanobind::detail

template <>
struct dolfinx_wrappers::numpy_dtype<re_worst_double_t> {
  static constexpr char value = 'D';
};

// --- dolfinx traits for running_error_t<double, ErrorMode::EXACT> ---

template <>
struct dolfinx::is_custom_scalar<re_exact_double_t> : std::true_type {};

template <>
struct dolfinx::scalar_value<re_exact_double_t> {
  using type = double;
};

template <>
struct dolfinx::MPI::mpi_type_mapping<re_exact_double_t> {
  static inline MPI_Datatype type = []() {
    MPI_Datatype dt;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }();
};

namespace nanobind::detail {
template <>
struct dtype_traits<re_exact_double_t> {
  static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Complex,
                                       128, 1};
  static constexpr auto name = const_name("complex128");
};
}  // namespace nanobind::detail

template <>
struct dolfinx_wrappers::numpy_dtype<re_exact_double_t> {
  static constexpr char value = 'D';
};

// --- Python bindings ---

namespace nb = nanobind;

NB_MODULE(_cpp, m) {
  m.doc() = "C++ extensions for dolfiny with running error tracking";

  nb::class_<re_worst_double_t>(m, "ReWorstDouble")
      .def(nb::init<double, double>(), nb::arg("val") = 0.0,
           nb::arg("err_bnd") = 0.0)
      .def_rw("val", &re_worst_double_t::val)
      .def_rw("err_bnd", &re_worst_double_t::err_bnd)
      .def("__repr__", [](const re_worst_double_t& d) {
        return std::format("ReWorstDouble(val={}, err_bnd={})", d.val,
                           d.err_bnd);
      });

  nb::class_<re_exact_double_t>(m, "ReExactDouble")
      .def(nb::init<double, double>(), nb::arg("val") = 0.0,
           nb::arg("exact_error") = 0.0)
      .def_rw("val", &re_exact_double_t::val)
      .def_rw("exact_error", &re_exact_double_t::exact_error)
      .def("__repr__", [](const re_exact_double_t& d) {
        return std::format("ReExactDouble(val={}, exact_error={})", d.val,
                           d.exact_error);
      });

  dolfinx_wrappers::declare_objects<re_worst_double_t>(m, "re_worst_double");
  dolfinx_wrappers::declare_form<re_worst_double_t>(m, "re_worst_double");
  dolfinx_wrappers::declare_assembly_functions<re_worst_double_t, double>(m);
  dolfinx_wrappers::declare_la_objects<re_worst_double_t>(m, "re_worst_double");

  dolfinx_wrappers::declare_objects<re_exact_double_t>(m, "re_exact_double");
  dolfinx_wrappers::declare_form<re_exact_double_t>(m, "re_exact_double");
  dolfinx_wrappers::declare_assembly_functions<re_exact_double_t, double>(m);
  dolfinx_wrappers::declare_la_objects<re_exact_double_t>(m, "re_exact_double");
}
