#include "gpu2/gpu2_layers.cuh"
#include "gpu/gpu_layers.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// Constant Memory Definition
// ============================================================================
__constant__ float d_const_b1[256];      // Conv1 bias
__constant__ float d_const_b2[128];      // Conv2 bias
__constant__ float d_const_b3[128];      // Conv3 bias
__constant__ float d_const_b4[256];      // Conv4 bias
__constant__ float d_const_b5[3];        // Conv5 bias

// ============================================================================
// Constant Memory Implementation
// ============================================================================
// Copy biases to constant memory for fast broadcast access
// All threads in a warp can read the same value in a single instruction

void gpu2_copy_bias_to_const_memory(const float* h_bias, int size, int bias_id) {
    switch (bias_id) {
        case 0:
            CUDA_CHECK(cudaMemcpyToSymbol(d_const_b1, h_bias, size * sizeof(float)));
            break;
        case 1:
            CUDA_CHECK(cudaMemcpyToSymbol(d_const_b2, h_bias, size * sizeof(float)));
            break;
        case 2:
            CUDA_CHECK(cudaMemcpyToSymbol(d_const_b3, h_bias, size * sizeof(float)));
            break;
        case 3:
            CUDA_CHECK(cudaMemcpyToSymbol(d_const_b4, h_bias, size * sizeof(float)));
            break;
        case 4:
            CUDA_CHECK(cudaMemcpyToSymbol(d_const_b5, h_bias, size * sizeof(float)));
            break;
        default:
            fprintf(stderr, "Invalid bias_id: %d\n", bias_id);
    }
}

// Optimized Conv2D with constant memory
__global__ void conv2d_forward_kernel_opt(
    const float* input,
    const float* weights,
    const float* bias,      // Not used - uses constant memory instead
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int bias_id              // 0=b1, 1=b2, 2=b3, 3=b4, 4=b5
) {
    // 3x3 convolution with padding=1, stride=1
    const int kernel_size = 3;
    const int pad = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, oc, h, w) - optimized for coalescing
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);

    float sum = 0.0f;
    
    // Read bias from constant memory - fast broadcast access!
    if (bias_id == 0) sum = d_const_b1[oc];
    else if (bias_id == 1) sum = d_const_b2[oc];
    else if (bias_id == 2) sum = d_const_b3[oc];
    else if (bias_id == 3) sum = d_const_b4[oc];
    else if (bias_id == 4) sum = d_const_b5[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    // Memory coalescing: threads in warp access consecutive addresses
                    int input_idx = b * (in_channels * height * width) +
                                    ic * (height * width) + ih * width + iw;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

// ============================================================================
// GPU2 Wrapper Functions - Reuse GPU kernels + Constant Memory Optimization
// ============================================================================

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
) {
    // Use the proven opt kernel to avoid the oversized tiled launch that was
    // overwhelming the GPU for mid-sized feature maps. This keeps per-thread
    // work bounded and prevents block-level write conflicts.
    int total_outputs = batch_size * out_channels * height * width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    conv2d_forward_kernel_opt<<<grid_size, block_size>>>(
        d_input, d_weights, nullptr, d_output,
        batch_size, in_channels, out_channels, height, width, bias_id
    );

    CUDA_CHECK(cudaGetLastError());
}

void gpu2_relu_forward(float* d_data, int size) {
    gpu_relu_forward(d_data, size);
}

void gpu2_maxpool2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    gpu_maxpool2d_forward(d_input, d_output, batch_size, channels, in_height, in_width);
}

void gpu2_upsample2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    gpu_upsample2d_forward(d_input, d_output, batch_size, channels, in_height, in_width);
}

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
) {
    gpu_conv2d_backward(d_input, d_weights, d_dL_doutput, d_dL_dinput, 
                       d_dL_dweights, d_dL_dbias,
                       batch_size, in_channels, out_channels, height, width);
}

void gpu2_relu_backward(
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int size
) {
    gpu_relu_backward(d_output, d_dL_doutput, d_dL_dinput, size);
}

void gpu2_maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    gpu_maxpool2d_backward(d_input, d_output, d_dL_doutput, d_dL_dinput,
                          batch_size, channels, in_height, in_width);
}

void gpu2_upsample2d_backward(
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    gpu_upsample2d_backward(d_dL_doutput, d_dL_dinput, batch_size, channels, in_height, in_width);
}

float gpu2_mse_loss(const float* d_output, const float* d_target, int size) {
    return gpu_mse_loss(d_output, d_target, size);
}

void gpu2_mse_loss_gradient(
    const float* d_output,
    const float* d_target,
    float* d_dL_doutput,
    int size
) {
    gpu_mse_loss_gradient(d_output, d_target, d_dL_doutput, size);
}

void gpu2_sgd_update(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    float clip_value,
    int size
) {
    gpu_sgd_update(d_weights, d_gradients, learning_rate, clip_value, size);
}

void gpu2_zero(float* d_data, int size) {
    gpu_zero(d_data, size);
}
