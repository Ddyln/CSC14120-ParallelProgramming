#ifndef GPU_LAYERS_CUH
#define GPU_LAYERS_CUH

#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ============================================================================
// Forward Pass Kernels
// ============================================================================

// Naive convolution 3x3 with padding=1, stride=1
// Each thread computes one output pixel
__global__ void conv2d_forward_kernel(
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

// ReLU activation: max(0, x)
// Each thread processes one element
__global__ void relu_forward_kernel(float* data, int size);

// Max pooling 2x2, stride=2
// Each thread computes one output element
__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Nearest neighbor upsampling (scale=2)
// Each thread computes one output pixel
__global__ void upsample2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// ============================================================================
// Backward Pass Kernels
// ============================================================================

// Compute dL/dinput, dL/dweights, dL/dbias for conv2d
__global__ void conv2d_backward_input_kernel(
    const float* weights,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

__global__ void conv2d_backward_weights_kernel(
    const float* input,
    const float* dL_doutput,
    float* dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

__global__ void conv2d_backward_bias_kernel(
    const float* dL_doutput,
    float* dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
);

// ReLU backward: gradient is passed through if output > 0
__global__ void relu_backward_kernel(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int size
);

// MaxPool backward: gradient goes to the max element position
__global__ void maxpool2d_backward_kernel(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Upsample backward: sum gradients from 2x2 output region
__global__ void upsample2d_backward_kernel(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// ============================================================================
// Loss Kernels
// ============================================================================

// MSE loss gradient: dL/doutput = 2*(output - target) / N
__global__ void mse_loss_gradient_kernel(
    const float* output,
    const float* target,
    float* dL_doutput,
    int size
);

// MSE loss value (partial sum per block)
__global__ void mse_loss_kernel(
    const float* output,
    const float* target,
    float* partial_sums,
    int size
);

// ============================================================================
// Weight Update Kernels
// ============================================================================

// SGD update with gradient clipping: w = w - lr * clip(grad, -clip_val, clip_val)
__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    float clip_value,
    int size
);

// Zero out array
__global__ void zero_kernel(float* data, int size);

// ============================================================================
// Wrapper Functions (Host Functions)
// ============================================================================

void gpu_conv2d_forward(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

void gpu_relu_forward(float* d_data, int size);

void gpu_maxpool2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_upsample2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_conv2d_backward(
    const float* d_input,
    const float* d_weights,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    float* d_dL_dweights,
    float* d_dL_dbias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

void gpu_relu_backward(
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int size
);

void gpu_maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_upsample2d_backward(
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

float gpu_mse_loss(const float* d_output, const float* d_target, int size);

void gpu_mse_loss_gradient(
    const float* d_output,
    const float* d_target,
    float* d_dL_doutput,
    int size
);

void gpu_sgd_update(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    float clip_value,
    int size
);

void gpu_zero(float* d_data, int size);

#endif  // GPU_LAYERS_CUH
