#ifndef GPU_LAYERS_V2_CUH
#define GPU_LAYERS_V2_CUH

#include <cuda_runtime.h>

// ============================================================================
// GPU Optimized Layers V2 - Kernel Optimization Bundle
// ============================================================================
// Techniques applied:
//   1. Kernel Fusion (Conv + Bias + ReLU) - Reduces global memory writes
//   2. Loop Unrolling (3x3 kernel) - Reduces loop overhead, increases ILP
//   3. Vectorized Memory Access (float4) - Increases memory bandwidth
// ============================================================================

namespace gpu_v2 {

// ============================================================================
// FUSED FORWARD KERNELS (Conv + Bias + ReLU in single kernel)
// ============================================================================

// Fused Conv2D + Bias + ReLU kernel
// Combines 3 operations into 1 kernel launch, reducing global memory traffic
__global__ void conv2d_bias_relu_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
);

// Fused Conv2D + Bias (without ReLU, for final decoder layer)
__global__ void conv2d_bias_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
);

// ============================================================================
// OPTIMIZED POOLING/UPSAMPLING KERNELS
// ============================================================================

// MaxPool with vectorized memory access where possible
__global__ void maxpool2d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int pool_size, int stride
);

// Upsample with vectorized memory writes
__global__ void upsample2d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int scale_factor
);

// ============================================================================
// OPTIMIZED LOSS KERNELS
// ============================================================================

// MSE Loss with vectorized reduction
__global__ void mse_loss_optimized_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size
);

// ============================================================================
// FUSED BACKWARD KERNELS
// ============================================================================

// Fused backward for Conv + ReLU (combines relu_backward + conv_backward_data)
__global__ void conv2d_relu_backward_fused_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ forward_output,  // For ReLU gradient
    float* __restrict__ grad_input,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding
);

// Optimized weight gradient computation with loop unrolling
__global__ void conv2d_weight_grad_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding
);

// Optimized bias gradient with parallel reduction
__global__ void bias_grad_optimized_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int out_channels, int height, int width
);

// MaxPool backward optimized
__global__ void maxpool2d_backward_optimized_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int pool_size, int stride
);

// Upsample backward optimized
__global__ void upsample2d_backward_optimized_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int scale_factor
);

// ============================================================================
// OPTIMIZED WEIGHT UPDATE
// ============================================================================

// SGD update with vectorized access (float4)
__global__ void sgd_update_vectorized_kernel(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float learning_rate,
    int size
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Wrapper functions for easier calling
void conv2d_bias_relu_forward_v2(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
);

void conv2d_bias_forward_v2(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
);

void maxpool2d_forward_v2(
    const float* input, float* output,
    int batch_size, int channels,
    int in_height, int in_width,
    int pool_size, int stride
);

void upsample2d_forward_v2(
    const float* input, float* output,
    int batch_size, int channels,
    int in_height, int in_width,
    int scale_factor
);

float mse_loss_v2(
    const float* output, const float* target,
    int batch_size, int channels, int height, int width
);

void sgd_update_v2(
    float* weights, const float* gradients,
    float learning_rate, int size
);

}  // namespace gpu_v2

#endif  // GPU_LAYERS_V2_CUH
