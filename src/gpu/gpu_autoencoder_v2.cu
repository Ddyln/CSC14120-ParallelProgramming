// ============================================================================
// GPU Autoencoder V2 - Optimized Implementation
// ============================================================================

#include "gpu/gpu_autoencoder_v2.h"
#include "gpu/gpu_layers_v2.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <cstring>

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

GPUAutoencoderV2::GPUAutoencoderV2() : max_batch_size(0), current_batch_size(0) {
    // Initialize all pointers to nullptr
    d_w1 = d_b1 = d_w2 = d_b2 = d_w3 = d_b3 = nullptr;
    d_w4 = d_b4 = d_w5 = d_b5 = nullptr;
    d_grad_w1 = d_grad_b1 = d_grad_w2 = d_grad_b2 = nullptr;
    d_grad_w3 = d_grad_b3 = d_grad_w4 = d_grad_b4 = nullptr;
    d_grad_w5 = d_grad_b5 = nullptr;
    d_conv1_out = d_pool1_out = d_conv2_out = d_pool2_out = nullptr;
    d_conv3_out = d_up1_out = d_conv4_out = d_up2_out = d_conv5_out = nullptr;
    d_grad_conv5 = d_grad_up2 = d_grad_conv4 = d_grad_up1 = nullptr;
    d_grad_conv3 = d_grad_pool2 = d_grad_conv2 = d_grad_pool1 = d_grad_conv1 = nullptr;

    allocate_weights();
    init_weights();
    
    printf("[GPUAutoencoderV2] Initialized with %d parameters\n", TOTAL_PARAMS);
    printf("[GPUAutoencoderV2] Optimizations: Kernel Fusion + Loop Unrolling + Tuned Blocks\n");
}

GPUAutoencoderV2::~GPUAutoencoderV2() {
    free_memory();
}

// ============================================================================
// MEMORY ALLOCATION
// ============================================================================

void GPUAutoencoderV2::allocate_weights() {
    cudaMalloc(&d_w1, W1_SIZE * sizeof(float));
    cudaMalloc(&d_b1, B1_SIZE * sizeof(float));
    cudaMalloc(&d_w2, W2_SIZE * sizeof(float));
    cudaMalloc(&d_b2, B2_SIZE * sizeof(float));
    cudaMalloc(&d_w3, W3_SIZE * sizeof(float));
    cudaMalloc(&d_b3, B3_SIZE * sizeof(float));
    cudaMalloc(&d_w4, W4_SIZE * sizeof(float));
    cudaMalloc(&d_b4, B4_SIZE * sizeof(float));
    cudaMalloc(&d_w5, W5_SIZE * sizeof(float));
    cudaMalloc(&d_b5, B5_SIZE * sizeof(float));

    cudaMalloc(&d_grad_w1, W1_SIZE * sizeof(float));
    cudaMalloc(&d_grad_b1, B1_SIZE * sizeof(float));
    cudaMalloc(&d_grad_w2, W2_SIZE * sizeof(float));
    cudaMalloc(&d_grad_b2, B2_SIZE * sizeof(float));
    cudaMalloc(&d_grad_w3, W3_SIZE * sizeof(float));
    cudaMalloc(&d_grad_b3, B3_SIZE * sizeof(float));
    cudaMalloc(&d_grad_w4, W4_SIZE * sizeof(float));
    cudaMalloc(&d_grad_b4, B4_SIZE * sizeof(float));
    cudaMalloc(&d_grad_w5, W5_SIZE * sizeof(float));
    cudaMalloc(&d_grad_b5, B5_SIZE * sizeof(float));
}

void GPUAutoencoderV2::allocate_activations(int batch_size) {
    if (batch_size <= max_batch_size && d_conv1_out != nullptr) {
        return; // Already allocated for this batch size or larger
    }

    // Free old allocations if they exist
    if (d_conv1_out) {
        cudaFree(d_conv1_out);
        cudaFree(d_pool1_out);
        cudaFree(d_conv2_out);
        cudaFree(d_pool2_out);
        cudaFree(d_conv3_out);
        cudaFree(d_up1_out);
        cudaFree(d_conv4_out);
        cudaFree(d_up2_out);
        cudaFree(d_conv5_out);
    }

    max_batch_size = batch_size;

    cudaMalloc(&d_conv1_out, batch_size * CONV1_OUT * CONV1_H * CONV1_W * sizeof(float));
    cudaMalloc(&d_pool1_out, batch_size * CONV1_OUT * POOL1_H * POOL1_W * sizeof(float));
    cudaMalloc(&d_conv2_out, batch_size * CONV2_OUT * CONV2_H * CONV2_W * sizeof(float));
    cudaMalloc(&d_pool2_out, batch_size * CONV2_OUT * POOL2_H * POOL2_W * sizeof(float));
    cudaMalloc(&d_conv3_out, batch_size * CONV3_OUT * CONV3_H * CONV3_W * sizeof(float));
    cudaMalloc(&d_up1_out, batch_size * CONV3_OUT * UP1_H * UP1_W * sizeof(float));
    cudaMalloc(&d_conv4_out, batch_size * CONV4_OUT * CONV4_H * CONV4_W * sizeof(float));
    cudaMalloc(&d_up2_out, batch_size * CONV4_OUT * UP2_H * UP2_W * sizeof(float));
    cudaMalloc(&d_conv5_out, batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float));
}

void GPUAutoencoderV2::allocate_gradients(int batch_size) {
    if (d_grad_conv5 != nullptr) {
        if (batch_size <= max_batch_size) {
            return;
        }
        // Free old allocations
        cudaFree(d_grad_conv5);
        cudaFree(d_grad_up2);
        cudaFree(d_grad_conv4);
        cudaFree(d_grad_up1);
        cudaFree(d_grad_conv3);
        cudaFree(d_grad_pool2);
        cudaFree(d_grad_conv2);
        cudaFree(d_grad_pool1);
        cudaFree(d_grad_conv1);
    }

    cudaMalloc(&d_grad_conv5, batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float));
    cudaMalloc(&d_grad_up2, batch_size * CONV4_OUT * UP2_H * UP2_W * sizeof(float));
    cudaMalloc(&d_grad_conv4, batch_size * CONV4_OUT * CONV4_H * CONV4_W * sizeof(float));
    cudaMalloc(&d_grad_up1, batch_size * CONV3_OUT * UP1_H * UP1_W * sizeof(float));
    cudaMalloc(&d_grad_conv3, batch_size * CONV3_OUT * CONV3_H * CONV3_W * sizeof(float));
    cudaMalloc(&d_grad_pool2, batch_size * CONV2_OUT * POOL2_H * POOL2_W * sizeof(float));
    cudaMalloc(&d_grad_conv2, batch_size * CONV2_OUT * CONV2_H * CONV2_W * sizeof(float));
    cudaMalloc(&d_grad_pool1, batch_size * CONV1_OUT * POOL1_H * POOL1_W * sizeof(float));
    cudaMalloc(&d_grad_conv1, batch_size * CONV1_OUT * CONV1_H * CONV1_W * sizeof(float));
}

void GPUAutoencoderV2::init_weights() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);

    float scale1 = sqrtf(2.0f / (INPUT_C * 3 * 3));
    curandGenerateNormal(gen, d_w1, W1_SIZE, 0.0f, scale1);
    cudaMemset(d_b1, 0, B1_SIZE * sizeof(float));

    float scale2 = sqrtf(2.0f / (CONV1_OUT * 3 * 3));
    curandGenerateNormal(gen, d_w2, W2_SIZE, 0.0f, scale2);
    cudaMemset(d_b2, 0, B2_SIZE * sizeof(float));

    float scale3 = sqrtf(2.0f / (CONV2_OUT * 3 * 3));
    curandGenerateNormal(gen, d_w3, W3_SIZE, 0.0f, scale3);
    cudaMemset(d_b3, 0, B3_SIZE * sizeof(float));

    float scale4 = sqrtf(2.0f / (CONV3_OUT * 3 * 3));
    curandGenerateNormal(gen, d_w4, W4_SIZE, 0.0f, scale4);
    cudaMemset(d_b4, 0, B4_SIZE * sizeof(float));

    float scale5 = sqrtf(2.0f / (CONV4_OUT * 3 * 3));
    curandGenerateNormal(gen, d_w5, W5_SIZE, 0.0f, scale5);
    cudaMemset(d_b5, 0, B5_SIZE * sizeof(float));

    curandDestroyGenerator(gen);
}

void GPUAutoencoderV2::free_memory() {
    // Free weights
    if (d_w1) cudaFree(d_w1);
    if (d_b1) cudaFree(d_b1);
    if (d_w2) cudaFree(d_w2);
    if (d_b2) cudaFree(d_b2);
    if (d_w3) cudaFree(d_w3);
    if (d_b3) cudaFree(d_b3);
    if (d_w4) cudaFree(d_w4);
    if (d_b4) cudaFree(d_b4);
    if (d_w5) cudaFree(d_w5);
    if (d_b5) cudaFree(d_b5);

    // Free gradients
    if (d_grad_w1) cudaFree(d_grad_w1);
    if (d_grad_b1) cudaFree(d_grad_b1);
    if (d_grad_w2) cudaFree(d_grad_w2);
    if (d_grad_b2) cudaFree(d_grad_b2);
    if (d_grad_w3) cudaFree(d_grad_w3);
    if (d_grad_b3) cudaFree(d_grad_b3);
    if (d_grad_w4) cudaFree(d_grad_w4);
    if (d_grad_b4) cudaFree(d_grad_b4);
    if (d_grad_w5) cudaFree(d_grad_w5);
    if (d_grad_b5) cudaFree(d_grad_b5);

    // Free activations
    if (d_conv1_out) cudaFree(d_conv1_out);
    if (d_pool1_out) cudaFree(d_pool1_out);
    if (d_conv2_out) cudaFree(d_conv2_out);
    if (d_pool2_out) cudaFree(d_pool2_out);
    if (d_conv3_out) cudaFree(d_conv3_out);
    if (d_up1_out) cudaFree(d_up1_out);
    if (d_conv4_out) cudaFree(d_conv4_out);
    if (d_up2_out) cudaFree(d_up2_out);
    if (d_conv5_out) cudaFree(d_conv5_out);

    // Free gradient buffers
    if (d_grad_conv5) cudaFree(d_grad_conv5);
    if (d_grad_up2) cudaFree(d_grad_up2);
    if (d_grad_conv4) cudaFree(d_grad_conv4);
    if (d_grad_up1) cudaFree(d_grad_up1);
    if (d_grad_conv3) cudaFree(d_grad_conv3);
    if (d_grad_pool2) cudaFree(d_grad_pool2);
    if (d_grad_conv2) cudaFree(d_grad_conv2);
    if (d_grad_pool1) cudaFree(d_grad_pool1);
    if (d_grad_conv1) cudaFree(d_grad_conv1);
}

// ============================================================================
// FORWARD PASS - Using Fused Kernels (Device API)
// ============================================================================

void GPUAutoencoderV2::forward(const float* d_input, float* d_output, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);

    // ENCODER with FUSED kernels (Conv + Bias + ReLU in one kernel)
    
    // Conv1 + Bias + ReLU (FUSED)
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_input, d_w1, d_b1, d_conv1_out,
        batch_size, INPUT_C, CONV1_OUT, INPUT_H, INPUT_W, 3, 1, 1
    );

    // Pool1
    gpu_v2::maxpool2d_forward_v2(
        d_conv1_out, d_pool1_out,
        batch_size, CONV1_OUT, CONV1_H, CONV1_W, 2, 2
    );

    // Conv2 + Bias + ReLU (FUSED)
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_pool1_out, d_w2, d_b2, d_conv2_out,
        batch_size, CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1, 1
    );

    // Pool2 (Latent representation)
    gpu_v2::maxpool2d_forward_v2(
        d_conv2_out, d_pool2_out,
        batch_size, CONV2_OUT, CONV2_H, CONV2_W, 2, 2
    );

    // DECODER with FUSED kernels

    // Conv3 + Bias + ReLU (FUSED)
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_pool2_out, d_w3, d_b3, d_conv3_out,
        batch_size, CONV2_OUT, CONV3_OUT, POOL2_H, POOL2_W, 3, 1, 1
    );

    // Upsample1
    gpu_v2::upsample2d_forward_v2(
        d_conv3_out, d_up1_out,
        batch_size, CONV3_OUT, CONV3_H, CONV3_W, 2
    );

    // Conv4 + Bias + ReLU (FUSED)
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_up1_out, d_w4, d_b4, d_conv4_out,
        batch_size, CONV3_OUT, CONV4_OUT, UP1_H, UP1_W, 3, 1, 1
    );

    // Upsample2
    gpu_v2::upsample2d_forward_v2(
        d_conv4_out, d_up2_out,
        batch_size, CONV4_OUT, CONV4_H, CONV4_W, 2
    );

    // Conv5 + Bias (NO ReLU - final layer)
    gpu_v2::conv2d_bias_forward_v2(
        d_up2_out, d_w5, d_b5, d_output,
        batch_size, CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1, 1
    );
    
    // Store output for loss computation
    if (d_output != d_conv5_out) {
        cudaMemcpy(d_conv5_out, d_output,
                   batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// GET FEATURES - Extract latent representation (encoder only)
// ============================================================================

void GPUAutoencoderV2::get_features(const float* d_input, float* d_features, int batch_size) {
    allocate_activations(batch_size);

    // Run encoder only (same as forward but stop at pool2)
    
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_input, d_w1, d_b1, d_conv1_out,
        batch_size, INPUT_C, CONV1_OUT, INPUT_H, INPUT_W, 3, 1, 1
    );

    gpu_v2::maxpool2d_forward_v2(
        d_conv1_out, d_pool1_out,
        batch_size, CONV1_OUT, CONV1_H, CONV1_W, 2, 2
    );

    gpu_v2::conv2d_bias_relu_forward_v2(
        d_pool1_out, d_w2, d_b2, d_conv2_out,
        batch_size, CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1, 1
    );

    gpu_v2::maxpool2d_forward_v2(
        d_conv2_out, d_features,
        batch_size, CONV2_OUT, CONV2_H, CONV2_W, 2, 2
    );
}

// ============================================================================
// COMPUTE LOSS - Device API
// ============================================================================

float GPUAutoencoderV2::compute_loss(const float* d_output, const float* d_target, int batch_size) {
    return gpu_v2::mse_loss_v2(d_output, d_target, batch_size, CONV5_OUT, CONV5_H, CONV5_W);
}

// ============================================================================
// BACKWARD PASS - Device API with Optimized Kernels
// ============================================================================

// MSE gradient kernel (replaces slow CPU lambda)
__global__ void mse_gradient_kernel_v2(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = scale * (output[idx] - target[idx]);
    }
}

__global__ void conv2d_backward_data_v2(
    const float* __restrict__ grad_output,
    const float* __restrict__ weight,
    float* __restrict__ grad_input,
    int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size, int padding
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (w >= in_width || h >= in_height) return;
    
    int out_height = in_height;
    int out_width = in_width;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int oh = h + padding - kh;
                int ow = w + padding - kw;
                
                if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                    int grad_idx = oc * out_height * out_width + oh * out_width + ow;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                    c * kernel_size * kernel_size +
                                    kh * kernel_size + kw;
                    sum += grad_output[grad_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int input_idx = c * in_height * in_width + h * in_width + w;
    grad_input[input_idx] = sum;
}

void GPUAutoencoderV2::backward(const float* d_input, const float* d_target, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);
    allocate_gradients(batch_size);
    
    // Zero weight gradients
    cudaMemset(d_grad_w1, 0, W1_SIZE * sizeof(float));
    cudaMemset(d_grad_b1, 0, B1_SIZE * sizeof(float));
    cudaMemset(d_grad_w2, 0, W2_SIZE * sizeof(float));
    cudaMemset(d_grad_b2, 0, B2_SIZE * sizeof(float));
    cudaMemset(d_grad_w3, 0, W3_SIZE * sizeof(float));
    cudaMemset(d_grad_b3, 0, B3_SIZE * sizeof(float));
    cudaMemset(d_grad_w4, 0, W4_SIZE * sizeof(float));
    cudaMemset(d_grad_b4, 0, B4_SIZE * sizeof(float));
    cudaMemset(d_grad_w5, 0, W5_SIZE * sizeof(float));
    cudaMemset(d_grad_b5, 0, B5_SIZE * sizeof(float));

    int output_size = batch_size * CONV5_OUT * CONV5_H * CONV5_W;
    float scale = 2.0f / output_size;

    // Compute MSE gradient on GPU
    dim3 block(256);
    dim3 grid((output_size + 255) / 256);
    mse_gradient_kernel_v2<<<grid, block>>>(d_conv5_out, d_target, d_grad_conv5, scale, output_size);

    dim3 spatial_block(16, 16);
    dim3 wgrad_block(9);  // 3x3 kernel

    // Backward through Conv5 (no ReLU)
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::conv2d_weight_grad_optimized_kernel(
            d_up2_out + b * CONV4_OUT * UP2_H * UP2_W,
            d_grad_conv5 + b * CONV5_OUT * CONV5_H * CONV5_W,
            d_grad_w5, CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1
        );
        
        dim3 data_grid5((UP2_W + 15) / 16, (UP2_H + 15) / 16, CONV4_OUT);
        conv2d_backward_data_v2<<<data_grid5, spatial_block>>>(
            d_grad_conv5 + b * CONV5_OUT * CONV5_H * CONV5_W,
            d_w5, d_grad_up2 + b * CONV4_OUT * UP2_H * UP2_W,
            CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel(
        d_grad_conv5, d_grad_b5, CONV5_OUT, CONV5_H * batch_size, CONV5_W
    );

    // Backward through Upsample2
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::upsample2d_backward_optimized_kernel(
            d_grad_up2 + b * CONV4_OUT * UP2_H * UP2_W,
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            CONV4_OUT, CONV4_H, CONV4_W, UP2_H, UP2_W, 2
        );
    }

    // Backward through Conv4 + ReLU (FUSED)
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::conv2d_relu_backward_fused_kernel(
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_up1_out + b * CONV3_OUT * UP1_H * UP1_W,
            d_w4, d_conv4_out + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_grad_up1 + b * CONV3_OUT * UP1_H * UP1_W,
            CONV3_OUT, CONV4_OUT, UP1_H, UP1_W, 3, 1
        );
        
        gpu_v2::conv2d_weight_grad_optimized_kernel(
            d_up1_out + b * CONV3_OUT * UP1_H * UP1_W,
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_grad_w4, CONV3_OUT, CONV4_OUT, UP1_H, UP1_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel(
        d_grad_conv4, d_grad_b4, CONV4_OUT, CONV4_H * batch_size, CONV4_W
    );

    // Backward through Upsample1
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::upsample2d_backward_optimized_kernel(
            d_grad_up1 + b * CONV3_OUT * UP1_H * UP1_W,
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            CONV3_OUT, CONV3_H, CONV3_W, UP1_H, UP1_W, 2
        );
    }

    // Backward through Conv3 + ReLU (FUSED)
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::conv2d_relu_backward_fused_kernel(
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_w3, d_conv3_out + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_grad_pool2 + b * CONV2_OUT * POOL2_H * POOL2_W,
            CONV2_OUT, CONV3_OUT, POOL2_H, POOL2_W, 3, 1
        );
        
        gpu_v2::conv2d_weight_grad_optimized_kernel(
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_grad_w3, CONV2_OUT, CONV3_OUT, POOL2_H, POOL2_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel(
        d_grad_conv3, d_grad_b3, CONV3_OUT, CONV3_H * batch_size, CONV3_W
    );

    // Backward through MaxPool2
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::maxpool2d_backward_optimized_kernel(
            d_grad_pool2 + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_conv2_out + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            CONV2_OUT, CONV2_H, CONV2_W, POOL2_H, POOL2_W, 2, 2
        );
    }

    // Backward through Conv2 + ReLU (FUSED)
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::conv2d_relu_backward_fused_kernel(
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_w2, d_conv2_out + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_grad_pool1 + b * CONV1_OUT * POOL1_H * POOL1_W,
            CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1
        );
        
        gpu_v2::conv2d_weight_grad_optimized_kernel(
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_grad_w2, CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel(
        d_grad_conv2, d_grad_b2, CONV2_OUT, CONV2_H * batch_size, CONV2_W
    );

    // Backward through MaxPool1
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::maxpool2d_backward_optimized_kernel(
            d_grad_pool1 + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_conv1_out + b * CONV1_OUT * CONV1_H * CONV1_W,
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_grad_conv1 + b * CONV1_OUT * CONV1_H * CONV1_W,
            CONV1_OUT, CONV1_H, CONV1_W, POOL1_H, POOL1_W, 2, 2
        );
    }

    // Backward through Conv1 + ReLU (weights only)
    for (int b = 0; b < batch_size; ++b) {
        gpu_v2::conv2d_weight_grad_optimized_kernel(
            d_input + b * INPUT_C * INPUT_H * INPUT_W,
            d_grad_conv1 + b * CONV1_OUT * CONV1_H * CONV1_W,
            d_grad_w1, INPUT_C, CONV1_OUT, INPUT_H, INPUT_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel(
        d_grad_conv1, d_grad_b1, CONV1_OUT, CONV1_H * batch_size, CONV1_W
    );

    cudaDeviceSynchronize();
}

// ============================================================================
// UPDATE WEIGHTS - Using vectorized SGD
// ============================================================================

void GPUAutoencoderV2::update_weights(float learning_rate) {
    gpu_v2::sgd_update_v2(d_w1, d_grad_w1, learning_rate, W1_SIZE);
    gpu_v2::sgd_update_v2(d_b1, d_grad_b1, learning_rate, B1_SIZE);
    gpu_v2::sgd_update_v2(d_w2, d_grad_w2, learning_rate, W2_SIZE);
    gpu_v2::sgd_update_v2(d_b2, d_grad_b2, learning_rate, B2_SIZE);
    gpu_v2::sgd_update_v2(d_w3, d_grad_w3, learning_rate, W3_SIZE);
    gpu_v2::sgd_update_v2(d_b3, d_grad_b3, learning_rate, B3_SIZE);
    gpu_v2::sgd_update_v2(d_w4, d_grad_w4, learning_rate, W4_SIZE);
    gpu_v2::sgd_update_v2(d_b4, d_grad_b4, learning_rate, B4_SIZE);
    gpu_v2::sgd_update_v2(d_w5, d_grad_w5, learning_rate, W5_SIZE);
    gpu_v2::sgd_update_v2(d_b5, d_grad_b5, learning_rate, B5_SIZE);
}

// ============================================================================
// WEIGHT PERSISTENCE
// ============================================================================

void GPUAutoencoderV2::save_weights(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return;
    }

    float* h_temp = new float[W2_SIZE];  // Largest weight matrix

    // Save all weights
    cudaMemcpy(h_temp, d_w1, W1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), W1_SIZE, f);
    cudaMemcpy(h_temp, d_b1, B1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), B1_SIZE, f);

    cudaMemcpy(h_temp, d_w2, W2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), W2_SIZE, f);
    cudaMemcpy(h_temp, d_b2, B2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), B2_SIZE, f);

    cudaMemcpy(h_temp, d_w3, W3_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), W3_SIZE, f);
    cudaMemcpy(h_temp, d_b3, B3_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), B3_SIZE, f);

    cudaMemcpy(h_temp, d_w4, W4_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), W4_SIZE, f);
    cudaMemcpy(h_temp, d_b4, B4_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), B4_SIZE, f);

    cudaMemcpy(h_temp, d_w5, W5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), W5_SIZE, f);
    cudaMemcpy(h_temp, d_b5, B5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_temp, sizeof(float), B5_SIZE, f);

    delete[] h_temp;
    fclose(f);
    printf("[GPUAutoencoderV2] Weights saved to %s\n", filename);
}

void GPUAutoencoderV2::load_weights(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Cannot open file %s for reading\n", filename);
        return;
    }

    float* h_temp = new float[W2_SIZE];

    fread(h_temp, sizeof(float), W1_SIZE, f);
    cudaMemcpy(d_w1, h_temp, W1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    fread(h_temp, sizeof(float), B1_SIZE, f);
    cudaMemcpy(d_b1, h_temp, B1_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    fread(h_temp, sizeof(float), W2_SIZE, f);
    cudaMemcpy(d_w2, h_temp, W2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    fread(h_temp, sizeof(float), B2_SIZE, f);
    cudaMemcpy(d_b2, h_temp, B2_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    fread(h_temp, sizeof(float), W3_SIZE, f);
    cudaMemcpy(d_w3, h_temp, W3_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    fread(h_temp, sizeof(float), B3_SIZE, f);
    cudaMemcpy(d_b3, h_temp, B3_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    fread(h_temp, sizeof(float), W4_SIZE, f);
    cudaMemcpy(d_w4, h_temp, W4_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    fread(h_temp, sizeof(float), B4_SIZE, f);
    cudaMemcpy(d_b4, h_temp, B4_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    fread(h_temp, sizeof(float), W5_SIZE, f);
    cudaMemcpy(d_w5, h_temp, W5_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    fread(h_temp, sizeof(float), B5_SIZE, f);
    cudaMemcpy(d_b5, h_temp, B5_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_temp;
    fclose(f);
    printf("[GPUAutoencoderV2] Weights loaded from %s\n", filename);
}
