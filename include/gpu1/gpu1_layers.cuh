#ifndef GPU1_LAYERS_CUH
#define GPU1_LAYERS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpu/gpu_layers.cuh"  // reuse baseline kernels and wrappers

// Fused Conv2D(3x3, padding=1, stride=1) + ReLU
// Kernel-level fusion with shared-memory tiling:
//  - Each block processes a spatial tile for one (batch, out_channel)
//  - Threads cooperatively load an input tile (with halo) into shared memory
//  - Then each thread computes one output element using the shared tile
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
    // 2D thread block over spatial dimensions, grid.z over (batch, out_channel)
    const int TILE_W = 16;
    const int TILE_H = 16;

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(
        (width  + TILE_W - 1) / TILE_W,
        (height + TILE_H - 1) / TILE_H,
        batch_size * out_channels
    );

    // Shared memory tile size: (TILE_H + 2*pad) x (TILE_W + 2*pad)
    const int pad = 1;
    size_t shared_mem_bytes = (TILE_W + 2 * pad) * (TILE_H + 2 * pad) * sizeof(float);

    conv2d_relu_forward_kernel<<<grid, block, shared_mem_bytes>>>(
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
