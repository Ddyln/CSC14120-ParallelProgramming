#ifndef GPU2_LAYERS_CUH
#define GPU2_LAYERS_CUH

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
// Constant Memory Configuration
// ============================================================================
// Store small, frequently accessed data in constant memory
// Total constant memory available: 64KB per GPU
// We'll use it for biases (small arrays)

#define MAX_BIAS_SIZE 256  // Maximum elements we can store in constant memory for biases

// Declare constant memory buffers (defined in gpu2_layers.cu)
extern __constant__ float d_const_b1[256];      // Conv1 bias
extern __constant__ float d_const_b2[128];      // Conv2 bias
extern __constant__ float d_const_b3[128];      // Conv3 bias
extern __constant__ float d_const_b4[256];      // Conv4 bias
extern __constant__ float d_const_b5[3];        // Conv5 bias

// ============================================================================
// Forward Pass Kernels with Optimizations
// ============================================================================
// Optimized kernel with constant memory
__global__ void conv2d_forward_kernel_opt(
    const float* input,
    const float* weights,
    const float* bias,      // Not used in this kernel - uses constant memory
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int bias_id              // 0=b1, 1=b2, 2=b3, 3=b4, 4=b5
);

__global__ void relu_forward_kernel(float* data, int size);

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

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

__global__ void relu_backward_kernel(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int size
);

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

__global__ void mse_loss_gradient_kernel(
    const float* output,
    const float* target,
    float* dL_doutput,
    int size
);

__global__ void mse_loss_kernel(
    const float* output,
    const float* target,
    float* partial_sums,
    int size
);

// ============================================================================
// Weight Update Kernels
// ============================================================================

__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    float clip_value,
    int size
);

__global__ void zero_kernel(float* data, int size);

// ============================================================================
// Wrapper Functions (Host Functions)
// ============================================================================

void gpu2_copy_bias_to_const_memory(const float* h_bias, int size, int bias_id);

void gpu2_conv2d_forward(
    const float* d_input,
    const float* d_weights,
    float* d_output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int bias_id
);

void gpu2_relu_forward(float* d_data, int size);

void gpu2_maxpool2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu2_upsample2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu2_conv2d_backward(
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

void gpu2_relu_backward(
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int size
);

void gpu2_maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu2_upsample2d_backward(
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

float gpu2_mse_loss(const float* d_output, const float* d_target, int size);

void gpu2_mse_loss_gradient(
    const float* d_output,
    const float* d_target,
    float* d_dL_doutput,
    int size
);

void gpu2_sgd_update(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    float clip_value,
    int size
);

void gpu2_zero(float* d_data, int size);

#endif  // GPU2_LAYERS_CUH
