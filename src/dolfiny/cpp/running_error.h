#pragma once

#include <mpfr.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace dolfiny {

enum class ErrorMode { WORST, EXACT };

static constexpr mpfr_prec_t MPFR_DEFAULT_PREC = 256;

// ---------------------------------------------------------------------------
// WORST mode: running upper bound on absolute rounding error
// ---------------------------------------------------------------------------

template <typename T, ErrorMode Mode = ErrorMode::WORST>
struct running_error_t {
  using value_type = T;
  using re_t = running_error_t;

  T val;
  T err_bnd;

  constexpr running_error_t(T _val = T(0), T _err = T(0)) noexcept
      : val(_val), err_bnd(_err) {}

  //
  // Basic arithmetic operators, type homogeneous only
  //

  // f(a,b) = a + b:  ∂f/∂a = 1, ∂f/∂b = 1
  re_t operator+(const re_t& other) const {
    const T new_val = val + other.val;
    return re_t{new_val, err_bnd + other.err_bnd + eps * std::abs(new_val)};
  }

  // f(a,b) = a - b:  ∂f/∂a = 1, ∂f/∂b = -1
  re_t operator-(const re_t& other) const {
    const T new_val = val - other.val;
    return re_t{new_val, err_bnd + other.err_bnd + eps * std::abs(new_val)};
  }

  // Negation is exact in IEEE 754 (sign bit flip, no rounding).
  re_t operator-() const { return re_t{-val, err_bnd}; }

  // f(a,b) = a * b:  ∂f/∂a = b, ∂f/∂b = a
  re_t operator*(const re_t& other) const {
    const T new_val = val * other.val;
    return re_t{new_val, std::abs(other.val) * err_bnd +
                             std::abs(val) * other.err_bnd +
                             eps * std::abs(new_val)};
  }

  // f(a,b) = a / b:  ∂f/∂a = 1/b, ∂f/∂b = -a/b² = -new_val/b
  re_t operator/(const re_t& other) const {
    const T new_val = val / other.val;
    const T new_err = err_bnd / std::abs(other.val) +
                      std::abs(new_val / other.val) * other.err_bnd +
                      eps * std::abs(new_val);
    return re_t{new_val, new_err};
  }

  //
  // Compound assignment operators, type homogeneous only
  //

  re_t& operator+=(const re_t& other) { return *this = *this + other; }
  re_t& operator-=(const re_t& other) { return *this = *this - other; }
  re_t& operator*=(const re_t& other) { return *this = *this * other; }
  re_t& operator/=(const re_t& other) { return *this = *this / other; }

  //
  // Compound assignment with scalar
  //

  re_t& operator*=(T scalar) { return *this = re_t{scalar, T(0)} * *this; }

  //
  // Comparison operators
  //

  bool operator==(const re_t& other) const {
    return val == other.val && err_bnd == other.err_bnd;
  }
  bool operator!=(const re_t& other) const { return !(*this == other); }

 private:
  // Note: this is the machine epsilon (e.g. 2^-52 for double), i.e. twice the
  // unit roundoff u = 2^-53 of round-to-nearest. The local rounding term is
  // therefore a factor ~2 looser than strictly necessary. This keeps the bound
  // a safe (conservative) upper bound; use eps/2 if a tighter estimate is
  // preferred over guaranteed pessimism.
  static constexpr T eps = std::numeric_limits<T>::epsilon();
};

// ---------------------------------------------------------------------------
// EXACT mode: signed accumulated error via first-order derivative propagation
//             with MPFR-computed local rounding at each operation.
// ---------------------------------------------------------------------------

template <typename T>
struct running_error_t<T, ErrorMode::EXACT> {
  using value_type = T;
  using re_t = running_error_t;

  T val;
  T exact_error;

  constexpr running_error_t(T _val = T(0), T _err = T(0)) noexcept
      : val(_val), exact_error(_err) {}

  // Local rounding error introduced by the current operation, defined with the
  // same sign convention as the accumulated error: computed minus true, i.e.
  // (T result) - (MPFR result). This is added to the propagated error from the
  // inputs so that exact_error tracks (computed - true) throughout.
  static T local_rounding(mpfr_t mpfr_result, mpfr_t scratch, const T result) {
    mpfr_set_d(scratch, static_cast<double>(result), MPFR_RNDN);
    mpfr_sub(scratch, scratch, mpfr_result, MPFR_RNDN);
    return static_cast<T>(mpfr_get_d(scratch, MPFR_RNDN));
  }

  //
  // Basic arithmetic operators, type homogeneous only
  //

  // f(a,b) = a + b:  ∂f/∂a = 1, ∂f/∂b = 1
  re_t operator+(const re_t& other) const {
    const T new_val = val + other.val;
    mpfr_t a, b, r;
    mpfr_inits2(MPFR_DEFAULT_PREC, a, b, r, nullptr);
    mpfr_set_d(a, static_cast<double>(val), MPFR_RNDN);
    mpfr_set_d(b, static_cast<double>(other.val), MPFR_RNDN);
    mpfr_add(r, a, b, MPFR_RNDN);
    const T new_err =
        exact_error + other.exact_error + local_rounding(r, a, new_val);
    mpfr_clears(a, b, r, nullptr);
    return re_t{new_val, new_err};
  }

  // f(a,b) = a - b:  ∂f/∂a = 1, ∂f/∂b = -1
  re_t operator-(const re_t& other) const {
    const T new_val = val - other.val;
    mpfr_t a, b, r;
    mpfr_inits2(MPFR_DEFAULT_PREC, a, b, r, nullptr);
    mpfr_set_d(a, static_cast<double>(val), MPFR_RNDN);
    mpfr_set_d(b, static_cast<double>(other.val), MPFR_RNDN);
    mpfr_sub(r, a, b, MPFR_RNDN);
    const T new_err =
        exact_error - other.exact_error + local_rounding(r, a, new_val);
    mpfr_clears(a, b, r, nullptr);
    return re_t{new_val, new_err};
  }

  // Negation is exact in IEEE 754: ∂(-a)/∂a = -1, no local rounding.
  re_t operator-() const { return re_t{-val, -exact_error}; }

  // f(a,b) = a * b:  ∂f/∂a = b, ∂f/∂b = a
  re_t operator*(const re_t& other) const {
    const T new_val = val * other.val;
    mpfr_t a, b, r;
    mpfr_inits2(MPFR_DEFAULT_PREC, a, b, r, nullptr);
    mpfr_set_d(a, static_cast<double>(val), MPFR_RNDN);
    mpfr_set_d(b, static_cast<double>(other.val), MPFR_RNDN);
    mpfr_mul(r, a, b, MPFR_RNDN);
    const T new_err = other.val * exact_error + val * other.exact_error +
                      local_rounding(r, a, new_val);
    mpfr_clears(a, b, r, nullptr);
    return re_t{new_val, new_err};
  }

  // f(a,b) = a / b:  ∂f/∂a = 1/b, ∂f/∂b = -a/b² = -new_val/b
  re_t operator/(const re_t& other) const {
    const T new_val = val / other.val;
    mpfr_t a, b, r;
    mpfr_inits2(MPFR_DEFAULT_PREC, a, b, r, nullptr);
    mpfr_set_d(a, static_cast<double>(val), MPFR_RNDN);
    mpfr_set_d(b, static_cast<double>(other.val), MPFR_RNDN);
    mpfr_div(r, a, b, MPFR_RNDN);
    const T new_err = exact_error / other.val -
                      other.exact_error * new_val / other.val +
                      local_rounding(r, a, new_val);
    mpfr_clears(a, b, r, nullptr);
    return re_t{new_val, new_err};
  }

  //
  // Compound assignment operators, type homogeneous only
  //

  re_t& operator+=(const re_t& other) { return *this = *this + other; }
  re_t& operator-=(const re_t& other) { return *this = *this - other; }
  re_t& operator*=(const re_t& other) { return *this = *this * other; }
  re_t& operator/=(const re_t& other) { return *this = *this / other; }

  //
  // Compound assignment with scalar
  //

  re_t& operator*=(T scalar) { return *this = re_t{scalar, T(0)} * *this; }

  //
  // Comparison operators
  //

  bool operator==(const re_t& other) const {
    return val == other.val && exact_error == other.exact_error;
  }
  bool operator!=(const re_t& other) const { return !(*this == other); }
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

template <typename T>
using re_worst_t = dolfiny::running_error_t<T, ErrorMode::WORST>;
template <typename T>
using re_exact_t = dolfiny::running_error_t<T, ErrorMode::EXACT>;

static_assert(sizeof(re_worst_t<double>) == 2 * sizeof(double));
static_assert(sizeof(re_exact_t<double>) == 2 * sizeof(double));

// ---------------------------------------------------------------------------
// Free functions — WORST mode
//
// Free functions are used for inhomogeneous operations and to allow ADL-based
// math function overloading. Missing standard scalar operations are delegated
// to the homogeneous operators which seamlessly cover exactly the same math.
// ---------------------------------------------------------------------------

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator*(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>{static_cast<T>(lhs)} * rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator*(const re_worst_t<T>& lhs, S rhs) {
  return lhs * re_worst_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator+(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>{static_cast<T>(lhs)} + rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator+(const re_worst_t<T>& lhs, S rhs) {
  return lhs + re_worst_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator-(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>{static_cast<T>(lhs)} - rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator-(const re_worst_t<T>& lhs, S rhs) {
  return lhs - re_worst_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator/(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>{static_cast<T>(lhs)} / rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator/(const re_worst_t<T>& lhs, S rhs) {
  return lhs / re_worst_t<T>{static_cast<T>(rhs)};
}

template <typename T>
re_worst_t<T> abs(const re_worst_t<T>& x) {
  return re_worst_t<T>{std::abs(x.val), x.err_bnd};
}

template <typename T>
re_worst_t<T> sqrt(const re_worst_t<T>& x) {
  const T new_val = std::sqrt(x.val);
  const T new_err = (new_val != T(0)) ? x.err_bnd / (T(2) * new_val) +
                                            std::numeric_limits<T>::epsilon() *
                                                std::abs(new_val)
                                      : T(0);
  return re_worst_t<T>{new_val, new_err};
}

template <typename T>
re_worst_t<T> log(const re_worst_t<T>& x) {
  const T new_val = std::log(x.val);
  const T new_err =
      (x.val > T(0)) ? x.err_bnd / x.val +
                           std::numeric_limits<T>::epsilon() * std::abs(new_val)
                     : T(0);
  return re_worst_t<T>{new_val, new_err};
}

// pow(x, n): dy/dx = n*x^(n-1) = n*y/x, rewritten to avoid a second std::pow
// call.
template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, T n) {
  const T new_val = std::pow(x.val, n);
  const T deriv = (x.val != T(0)) ? std::abs(n * new_val / x.val) : T(0);
  return re_worst_t<T>{
      new_val, deriv * x.err_bnd +
                   std::numeric_limits<T>::epsilon() * std::abs(new_val)};
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, int n) {
  return pow(x, static_cast<T>(n));
}

// ---------------------------------------------------------------------------
// Free functions — EXACT mode
//
// Free functions are used for inhomogeneous operations and to allow ADL-based
// math function overloading.
//
// Scalar has no accumulated error, so derivative propagation is one-sided.
// Local rounding is always correctly computed down in the delegated homogeneous
// operators by treating the scalar as an exact mathematical `val` with 0 error.
// ---------------------------------------------------------------------------

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator*(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>{static_cast<T>(lhs)} * rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator*(const re_exact_t<T>& lhs, S rhs) {
  return lhs * re_exact_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator+(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>{static_cast<T>(lhs)} + rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator+(const re_exact_t<T>& lhs, S rhs) {
  return lhs + re_exact_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator-(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>{static_cast<T>(lhs)} - rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator-(const re_exact_t<T>& lhs, S rhs) {
  return lhs - re_exact_t<T>{static_cast<T>(rhs)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator/(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>{static_cast<T>(lhs)} / rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator/(const re_exact_t<T>& lhs, S rhs) {
  return lhs / re_exact_t<T>{static_cast<T>(rhs)};
}

// abs is exact in IEEE 754; derivative = sign(val)
template <typename T>
re_exact_t<T> abs(const re_exact_t<T>& x) {
  const T sign = (x.val > T(0)) ? T(1) : (x.val < T(0)) ? T(-1) : T(0);
  return re_exact_t<T>{std::abs(x.val), sign * x.exact_error};
}

// sqrt: ∂(√x)/∂x = 1/(2√x); local rounding via MPFR
template <typename T>
re_exact_t<T> sqrt(const re_exact_t<T>& x) {
  using re = re_exact_t<T>;
  const T new_val = std::sqrt(x.val);
  mpfr_t a, r;
  mpfr_inits2(MPFR_DEFAULT_PREC, a, r, nullptr);
  mpfr_set_d(a, static_cast<double>(x.val), MPFR_RNDN);
  mpfr_sqrt(r, a, MPFR_RNDN);
  const T local = re::local_rounding(r, a, new_val);
  const T deriv = (new_val != T(0)) ? T(1) / (T(2) * new_val) : T(0);
  mpfr_clears(a, r, nullptr);
  return re_exact_t<T>{new_val, deriv * x.exact_error + local};
}

// log: ∂(ln x)/∂x = 1/x; local rounding via MPFR
template <typename T>
re_exact_t<T> log(const re_exact_t<T>& x) {
  using re = re_exact_t<T>;
  const T new_val = std::log(x.val);
  mpfr_t a, r;
  mpfr_inits2(MPFR_DEFAULT_PREC, a, r, nullptr);
  mpfr_set_d(a, static_cast<double>(x.val), MPFR_RNDN);
  mpfr_log(r, a, MPFR_RNDN);
  const T local = re::local_rounding(r, a, new_val);
  const T deriv = (x.val > T(0)) ? T(1) / x.val : T(0);
  mpfr_clears(a, r, nullptr);
  return re_exact_t<T>{new_val, deriv * x.exact_error + local};
}

// pow(x, n): ∂(x^n)/∂x = n*x^(n-1) = n*new_val/x (signed); local rounding via
// MPFR
template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, T n) {
  using re = re_exact_t<T>;
  const T new_val = std::pow(x.val, n);
  mpfr_t a, exp_m, r;
  mpfr_inits2(MPFR_DEFAULT_PREC, a, exp_m, r, nullptr);
  mpfr_set_d(a, static_cast<double>(x.val), MPFR_RNDN);
  mpfr_set_d(exp_m, static_cast<double>(n), MPFR_RNDN);
  mpfr_pow(r, a, exp_m, MPFR_RNDN);
  const T local = re::local_rounding(r, a, new_val);
  const T deriv = (x.val != T(0)) ? n * new_val / x.val : T(0);
  mpfr_clears(a, exp_m, r, nullptr);
  return re_exact_t<T>{new_val, deriv * x.exact_error + local};
}

template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, int n) {
  using re = re_exact_t<T>;
  const T new_val = std::pow(x.val, n);
  mpfr_t a, r;
  mpfr_inits2(MPFR_DEFAULT_PREC, a, r, nullptr);
  mpfr_set_d(a, static_cast<double>(x.val), MPFR_RNDN);
  mpfr_pow_si(r, a, static_cast<long>(n), MPFR_RNDN);
  const T local = re::local_rounding(r, a, new_val);
  const T deriv = (x.val != T(0)) ? static_cast<T>(n) * new_val / x.val : T(0);
  mpfr_clears(a, r, nullptr);
  return re_exact_t<T>{new_val, deriv * x.exact_error + local};
}

}  // namespace dolfiny

// ---------------------------------------------------------------------------
// std:: overloads — WORST mode
// ---------------------------------------------------------------------------

namespace std {

template <typename T>
using re_worst_t = dolfiny::re_worst_t<T>;

template <typename T>
re_worst_t<T> abs(const re_worst_t<T>& x) {
  return dolfiny::abs(x);
}

template <typename T>
re_worst_t<T> sqrt(const re_worst_t<T>& x) {
  return dolfiny::sqrt(x);
}

template <typename T>
re_worst_t<T> log(const re_worst_t<T>& x) {
  return dolfiny::log(x);
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, T n) {
  return dolfiny::pow(x, n);
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, int n) {
  return dolfiny::pow(x, n);
}

// Complex-like interface (val is "real", no imaginary component)
template <typename T>
T real(const re_worst_t<T>& x) {
  return x.val;
}

template <typename T>
T imag(const re_worst_t<T>&) {
  return T(0);
}

template <typename T>
T norm(const re_worst_t<T>& x) {
  return x.val * x.val;
}

// ---------------------------------------------------------------------------
// std:: overloads — EXACT mode
// ---------------------------------------------------------------------------

template <typename T>
using re_exact_t = dolfiny::re_exact_t<T>;

template <typename T>
re_exact_t<T> abs(const re_exact_t<T>& x) {
  return dolfiny::abs(x);
}

template <typename T>
re_exact_t<T> sqrt(const re_exact_t<T>& x) {
  return dolfiny::sqrt(x);
}

template <typename T>
re_exact_t<T> log(const re_exact_t<T>& x) {
  return dolfiny::log(x);
}

template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, T n) {
  return dolfiny::pow(x, n);
}

template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, int n) {
  return dolfiny::pow(x, n);
}

template <typename T>
T real(const re_exact_t<T>& x) {
  return x.val;
}

template <typename T>
T imag(const re_exact_t<T>&) {
  return T(0);
}

template <typename T>
T norm(const re_exact_t<T>& x) {
  return x.val * x.val;
}

}  // namespace std
