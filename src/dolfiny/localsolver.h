// Vectorisation produces compiler error on my M1 machine
#define EIGEN_DONT_VECTORIZE
// C-style arrays for compatibility with FFCx
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>

using ufc_kernel_t = void(*)(double* __restrict__, const double* __restrict__, const double* __restrict__,
                             const double* __restrict__, int32_t* __restrict__, uint8_t* __restrict__, void*);

struct kernel_data_t {
    const ufc_kernel_t kernel;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> array;
    Eigen::Array<double, Eigen::Dynamic, 1> w;
    Eigen::Array<double, Eigen::Dynamic, 1> c;
    Eigen::Array<double, Eigen::Dynamic, 1> coords;
    Eigen::Array<int32_t, Eigen::Dynamic, 1> entity_local_index;
    Eigen::Array<uint8_t, Eigen::Dynamic, 1> permutation;
};
