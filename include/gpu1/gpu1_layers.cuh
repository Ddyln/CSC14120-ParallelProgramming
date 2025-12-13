#ifndef GPU1_LAYERS_CUH
#define GPU1_LAYERS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpu/gpu_layers.cuh"  // reuse baseline kernels and wrappers

// Fused Conv2D(3x3, padding=1, stride=1) + ReLU
// Kernel-level fusion: compute Conv2D and apply ReLU in single kernel without shared memory.
// Each thread computes one output element independently.
__global__ void conv2d_relu_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

// Host wrapper for fused Conv2D+ReLU (simple kernel-level fusion)
inline void gpu1_conv2d_relu_forward(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    int total_outputs = batch_size * out_channels * height * width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    conv2d_relu_forward_kernel<<<grid_size, block_size>>>(
        d_input,
        d_weights,
        d_bias,
        d_output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width
    );
    CUDA_CHECK(cudaGetLastError());
}

#endif // GPU1_LAYERS_CUH
