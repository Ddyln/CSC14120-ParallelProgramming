#ifndef GPU1_LAYERS_CUH
#define GPU1_LAYERS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpu/gpu_layers.cuh"  // reuse baseline kernels and wrappers

// Fused Conv2D(3x3, padding=1, stride=1) + ReLU
// Shared-memory tiling over spatial dimensions to reduce global reads.
// Each block processes an 8x8 output tile for a fixed (batch, out_channel).
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

// Host wrapper for fused Conv2D+ReLU with shared-memory tiling
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
    // Tile size must match constants inside kernel implementation
    const int TILE_W = 8;
    const int TILE_H = 8;

    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim(
        (width  + TILE_W - 1) / TILE_W,
        (height + TILE_H - 1) / TILE_H,
        batch_size * out_channels
    );

    conv2d_relu_forward_kernel<<<gridDim, blockDim>>>(
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
