#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "cuda_interpolation.cuh"

__global__ void compute_barycentric_coordinates_kernel(float* d_bary_coords, const float* d_points, const float* d_rhs) {
    int idx = threadIdx.x;
    __shared__ float T[16];
    __shared__ float rhs[4];

    if (idx < 4) {
        rhs[idx] = d_rhs[idx];
        for (int i = 0; i < 3; ++i) {
            T[idx * 4 + i] = d_points[idx * 3 + i];
        }
        T[idx * 4 + 3] = 1.0f;
    }

    __syncthreads();

    if (idx < 4) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += T[idx * 4 + i] * rhs[i];
        }
        d_bary_coords[idx] = sum;
    }
}

extern "C" void compute_barycentric_coordinates_cuda(Eigen::Vector4f& result, const float* points, const float* rhs) {
    float h_bary_coords[4];

    float *d_points, *d_rhs, *d_bary_coords;
    cudaMalloc((void**)&d_points, 12 * sizeof(float));
    cudaMalloc((void**)&d_rhs, 4 * sizeof(float));
    cudaMalloc((void**)&d_bary_coords, 4 * sizeof(float));

    cudaMemcpy(d_points, points, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, rhs, 4 * sizeof(float), cudaMemcpyHostToDevice);

    compute_barycentric_coordinates_kernel<<<1, 4>>>(d_bary_coords, d_points, d_rhs);

    cudaMemcpy(h_bary_coords, d_bary_coords, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_rhs);
    cudaFree(d_bary_coords);

    for (int i = 0; i < 4; ++i) {
        result[i] = h_bary_coords[i];
    }
}
