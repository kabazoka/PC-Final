#ifndef CUDA_INTERPOLATION_CUH
#define CUDA_INTERPOLATION_CUH

#include <Eigen/Dense>

extern "C" void compute_barycentric_coordinates_cuda(Eigen::Vector4f& result, const float* points, const float* rhs);

#endif // CUDA_INTERPOLATION_CUH
