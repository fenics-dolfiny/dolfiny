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

#include <cstddef>
#include <format>
#include <stdfloat>
#include <type_traits>

using re_worst_float64_t = running_error::re_worst_t<std::float64_t>;
using re_exact_float64_t = running_error::re_exact_t<std::float64_t>;

using re_worst_float32_t = running_error::re_worst_t<std::float32_t>;
using re_exact_float32_t = running_error::re_exact_t<std::float32_t>;

using re_worst_float16_t = running_error::re_worst_t<std::float16_t>;
using re_exact_float16_t = running_error::re_exact_t<std::float16_t>;

// ---------------------------------------------------------------------------
// dolfinx / nanobind trait specializations
//
// Each scalar is a homogeneous SCALAR_T {value, error} pair of
// 2*sizeof(SCALAR_T) bytes, exposed to dolfinx/numpy as an opaque scalar of
// matching size. The float16 pair carries float32 rather than a complex:
// numpy has no complex32.
//
//   RE_TYPE     fully-qualified running-error type, e.g. re_worst_float64_t
//   SCALAR_T    underlying floating-point type, e.g. double
//   DLPACK_CODE DLPack dtype code of the packed scalar, e.g. Complex
//   DLPACK_BITS bit width of the packed scalar, e.g. 128
//   NP_NAME     numpy dtype name of the packed scalar, e.g. "complex128"
//   NP_CHAR     numpy dtype char of the packed scalar, e.g. 'D'
//
// ---------------------------------------------------------------------------

#define TRAITS(RE_TYPE, SCALAR_T, DLPACK_CODE, DLPACK_BITS, NP_NAME, NP_CHAR) \
  static_assert(sizeof(RE_TYPE) == 2 * sizeof(SCALAR_T),                      \
                #RE_TYPE " must be a value + error pair of " #SCALAR_T);      \
  static_assert(alignof(RE_TYPE) == alignof(SCALAR_T),                        \
                #RE_TYPE " must have the alignment of " #SCALAR_T);           \
  static_assert(offsetof(RE_TYPE, val) == 0,                                  \
                #RE_TYPE "::val must be the first member");                   \
  static_assert(offsetof(RE_TYPE, err) == sizeof(SCALAR_T),                   \
                #RE_TYPE "::err must sit directly after ::val");              \
                                                                              \
  template <>                                                                 \
  struct dolfinx::is_custom_scalar<RE_TYPE> : std::true_type {};              \
                                                                              \
  template <>                                                                 \
  struct dolfinx::scalar_value<RE_TYPE> {                                     \
    using type = SCALAR_T;                                                    \
  };                                                                          \
                                                                              \
  template <>                                                                 \
  MPI_Datatype dolfinx::MPI::mpi_datatype<RE_TYPE>() {                        \
    static MPI_Datatype dt = []() {                                           \
      MPI_Datatype datatype;                                                  \
      MPI_Type_contiguous(sizeof(RE_TYPE), MPI_BYTE, &datatype);              \
      MPI_Type_commit(&datatype);                                             \
      return datatype;                                                        \
    }();                                                                      \
    return dt;                                                                \
  }                                                                           \
                                                                              \
  namespace nanobind::detail {                                                \
  template <>                                                                 \
  struct dtype_traits<RE_TYPE> {                                              \
    static constexpr dlpack::dtype value{                                     \
        (uint8_t)dlpack::dtype_code::DLPACK_CODE, DLPACK_BITS, 1};            \
    static constexpr auto name = const_name(NP_NAME);                         \
  };                                                                          \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  struct dolfinx_wrappers::numpy_dtype<RE_TYPE> {                             \
    static constexpr char value = NP_CHAR;                                    \
  }

TRAITS(re_worst_float64_t, double, Complex, 128, "complex128", 'D');
TRAITS(re_exact_float64_t, double, Complex, 128, "complex128", 'D');
TRAITS(re_worst_float32_t, float, Complex, 64, "complex64", 'F');
TRAITS(re_exact_float32_t, float, Complex, 64, "complex64", 'F');
TRAITS(re_worst_float16_t, std::float16_t, Float, 32, "float32", 'f');
TRAITS(re_exact_float16_t, std::float16_t, Float, 32, "float32", 'f');

#undef TRAITS

// dolfinx instantiates nothing for std::float16_t.
template <>
MPI_Datatype dolfinx::MPI::mpi_datatype<std::float16_t>() {
  static MPI_Datatype dt = []() {
    MPI_Datatype datatype;
    MPI_Type_contiguous(sizeof(std::float16_t), MPI_BYTE, &datatype);
    MPI_Type_commit(&datatype);
    return datatype;
  }();
  return dt;
}

namespace nanobind::detail {
template <>
struct dtype_traits<std::float16_t> {
  static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Float, 16,
                                       1};
  static constexpr auto name = const_name("float16");
};
}  // namespace nanobind::detail

template <>
struct dolfinx_wrappers::numpy_dtype<std::float16_t> {
  static constexpr char value = 'e';
};

// --- Python bindings ---

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Register the Python class and the dolfinx object/form/assembly/la wrappers
// for a single running-error scalar type.
//
//   RE_TYPE    fully-qualified running-error type, e.g. re_worst_float64_t
//   SCALAR_T   underlying floating-point type, e.g. double
//   GEOM_T     mesh geometry type, e.g. double
//   PY_NAME    Python class name string literal, e.g. "ReWorstFloat64"
//   SCALAR_TAG name of the scalar, e.g. "re_worst_float64"
//   GEOM_TAG   name of the geometry, e.g. "float64"
//
// ---------------------------------------------------------------------------

#define BIND_GEOM(RE_TYPE, GEOM_T, SCALAR_TAG, GEOM_TAG)                       \
  dolfinx_wrappers::declare_objects<RE_TYPE, GEOM_T>(m,                        \
                                                     SCALAR_TAG "_" GEOM_TAG); \
  dolfinx_wrappers::declare_form<RE_TYPE, GEOM_T>(m, SCALAR_TAG "_" GEOM_TAG); \
  dolfinx_wrappers::declare_assembly_functions<RE_TYPE, GEOM_T>(m)

#define BIND_TYPE(RE_TYPE, SCALAR_T, PY_NAME)                         \
  nb::class_<RE_TYPE>(m, PY_NAME)                                     \
      .def(nb::init<SCALAR_T, SCALAR_T>(), nb::arg("val") = 0.0,      \
           nb::arg("err") = 0.0)                                      \
      .def_rw("val", &RE_TYPE::val)                                   \
      .def_rw("err", &RE_TYPE::err)                                   \
      .def_ro_static("eps", &RE_TYPE::eps)                            \
      .def("__repr__", [](const RE_TYPE& d) {                         \
        return std::format(PY_NAME "(val={}, err={})", d.val, d.err); \
      });

// BIND_TYPE registers nb::class_<RE_TYPE> and must run first: the dolfinx
// wrappers bake defaulted arguments of type RE_TYPE (e.g. MatrixCSR's
// `eliminate_zeros(tol=T(0))`) into their signatures at module-init time, which
// requires RE_TYPE to already be a known nanobind type.
#define BIND(RE_TYPE, SCALAR_T, GEOM_T, PY_NAME, SCALAR_TAG, GEOM_TAG) \
  BIND_TYPE(RE_TYPE, SCALAR_T, PY_NAME);                               \
  dolfinx_wrappers::declare_constant<RE_TYPE>(m, SCALAR_TAG);          \
  dolfinx_wrappers::declare_la_objects<RE_TYPE>(m, SCALAR_TAG);        \
  BIND_GEOM(RE_TYPE, GEOM_T, SCALAR_TAG, GEOM_TAG);

NB_MODULE(_cpp, m) {
  m.doc() = "C++ extensions for dolfiny with running error tracking";

  BIND(re_worst_float64_t, double, double, "ReWorstFloat64", "re_worst_float64",
       "float64");
  BIND(re_exact_float64_t, double, double, "ReExactFloat64", "re_exact_float64",
       "float64");
  BIND(re_worst_float32_t, float, float, "ReWorstFloat32", "re_worst_float32",
       "float32");
  BIND(re_exact_float32_t, float, float, "ReExactFloat32", "re_exact_float32",
       "float32");
  BIND(re_worst_float16_t, std::float16_t, double, "ReWorstFloat16",
       "re_worst_float16", "float64");
  BIND(re_exact_float16_t, std::float16_t, double, "ReExactFloat16",
       "re_exact_float16", "float64");

  // Only bind the geometry objects.
  BIND_GEOM(re_worst_float32_t, double, "re_worst_float32", "float64");
  BIND_GEOM(re_exact_float32_t, double, "re_exact_float32", "float64");

  // Bind some objects missing from dolfinx exports.
  dolfinx_wrappers::declare_objects<float, double>(m, "float32_float64");
  dolfinx_wrappers::declare_la_objects<std::float16_t>(m, "float16");
  dolfinx_wrappers::declare_objects<std::float16_t, double>(m,
                                                            "float16_float64");
}

#undef BIND
#undef BIND_GEOM
#undef BIND_TYPE