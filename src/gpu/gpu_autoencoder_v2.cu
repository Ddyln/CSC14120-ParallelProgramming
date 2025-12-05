// ============================================================================
// GPU Autoencoder V2 - Optimized Implementation
// ============================================================================
// Techniques:
//   1. Kernel Fusion (Conv + Bias + ReLU)
//   2. Loop Unrolling (3x3)
//   3. Vectorized Memory (float4)
// ============================================================================

#include "gpu/gpu_autoencoder_v2.h"
#include "gpu/gpu_layers_v2.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <random>

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

GPUAutoencoderV2::GPUAutoencoderV2() : max_batch_size(0), current_batch_size(0) {
    // Initialize all pointers to nullptr
    d_w1 = d_b1 = d_w2 = d_b2 = d_w3 = d_b3 = nullptr;
    d_w4 = d_b4 = d_w5 = d_b5 = nullptr;
    d_grad_w1 = d_grad_b1 = d_grad_w2 = d_grad_b2 = nullptr;
    d_grad_w3 = d_grad_b3 = d_grad_w4 = d_grad_b4 = nullptr;
    d_grad_w5 = d_grad_b5 = nullptr;
    d_input = d_target = nullptr;
    d_conv1_out = d_pool1_out = d_conv2_out = d_pool2_out = nullptr;
    d_conv3_out = d_up1_out = d_conv4_out = d_up2_out = d_conv5_out = nullptr;
    d_grad_conv5 = d_grad_up2 = d_grad_conv4 = d_grad_up1 = nullptr;
    d_grad_conv3 = d_grad_pool2 = d_grad_conv2 = d_grad_pool1 = d_grad_conv1 = nullptr;

    allocate_weights();
    init_weights();
    
    printf("[GPUAutoencoderV2] Initialized with %d parameters\n", TOTAL_PARAMS);
    printf("[GPUAutoencoderV2] Optimizations: Kernel Fusion, Loop Unrolling, float4\n");
}

GPUAutoencoderV2::~GPUAutoencoderV2() {
    free_memory();
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

void GPUAutoencoderV2::allocate_weights() {
    // Allocate weight memory
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

    // Allocate gradient memory
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
    if (batch_size <= max_batch_size) return;

    // Free old allocations if any
    if (max_batch_size > 0) {
        cudaFree(d_input);
        cudaFree(d_target);
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

    // Allocate input/target buffers for host API
    cudaMalloc(&d_input, batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&d_target, batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float));

    // Allocate activation buffers
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
    if (d_grad_conv5 != nullptr) return;  // Already allocated

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
    std::random_device rd;
    std::mt19937 gen(rd());

    auto init_layer = [&](float* d_w, float* d_b, int w_size, int b_size, int fan_in) {
        float std_dev = sqrtf(2.0f / fan_in);  // He initialization
        std::normal_distribution<float> dist(0.0f, std_dev);

        float* h_w = new float[w_size];
        float* h_b = new float[b_size];

        for (int i = 0; i < w_size; ++i) h_w[i] = dist(gen);
        for (int i = 0; i < b_size; ++i) h_b[i] = 0.0f;

        cudaMemcpy(d_w, h_w, w_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, b_size * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_w;
        delete[] h_b;
    };

    init_layer(d_w1, d_b1, W1_SIZE, B1_SIZE, INPUT_C * 9);
    init_layer(d_w2, d_b2, W2_SIZE, B2_SIZE, CONV1_OUT * 9);
    init_layer(d_w3, d_b3, W3_SIZE, B3_SIZE, CONV2_OUT * 9);
    init_layer(d_w4, d_b4, W4_SIZE, B4_SIZE, CONV3_OUT * 9);
    init_layer(d_w5, d_b5, W5_SIZE, B5_SIZE, CONV4_OUT * 9);
}

void GPUAutoencoderV2::free_memory() {
    // Free weights
    cudaFree(d_w1); cudaFree(d_b1);
    cudaFree(d_w2); cudaFree(d_b2);
    cudaFree(d_w3); cudaFree(d_b3);
    cudaFree(d_w4); cudaFree(d_b4);
    cudaFree(d_w5); cudaFree(d_b5);

    // Free weight gradients
    cudaFree(d_grad_w1); cudaFree(d_grad_b1);
    cudaFree(d_grad_w2); cudaFree(d_grad_b2);
    cudaFree(d_grad_w3); cudaFree(d_grad_b3);
    cudaFree(d_grad_w4); cudaFree(d_grad_b4);
    cudaFree(d_grad_w5); cudaFree(d_grad_b5);

    // Free input/target buffers
    if (d_input != nullptr) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if (d_target != nullptr) {
        cudaFree(d_target);
        d_target = nullptr;
    }

    // Free activations
    if (max_batch_size > 0) {
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

    // Free activation gradients
    if (d_grad_conv5 != nullptr) {
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
}

// ============================================================================
// FORWARD PASS - Host API (public)
// ============================================================================

void GPUAutoencoderV2::forward(const float* h_input, float* h_output, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);
    
    // Copy input from host to device
    size_t input_size = batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    // Call device forward
    forward_device(d_input, d_conv5_out, batch_size);
    
    // Copy output from device to host
    size_t output_size = batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float);
    cudaMemcpy(h_output, d_conv5_out, output_size, cudaMemcpyDeviceToHost);
}

// ============================================================================
// FORWARD PASS - Device API (public)
// ============================================================================

void GPUAutoencoderV2::forward_gpu(const float* d_in, float* d_out, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);
    forward_device(d_in, d_out, batch_size);
}

// ============================================================================
// FORWARD PASS - Device API (internal)
// ============================================================================

void GPUAutoencoderV2::forward_device(const float* d_in, float* d_out, int batch_size) {
    allocate_activations(batch_size);

    // ENCODER with FUSED kernels (Conv + Bias + ReLU in one kernel)
    
    // Conv1 + Bias + ReLU (FUSED)
    gpu_v2::conv2d_bias_relu_forward_v2(
        d_in, d_w1, d_b1, d_conv1_out,
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
        d_up2_out, d_w5, d_b5, d_conv5_out,
        batch_size, CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1, 1
    );

    // Copy to output if different from d_conv5_out
    if (d_out != d_conv5_out) {
        cudaMemcpy(d_out, d_conv5_out, 
                   batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// GET FEATURES (Encoder only) - Host API
// ============================================================================

void GPUAutoencoderV2::get_features(const float* h_input, float* h_features, int batch_size) {
    allocate_activations(batch_size);
    
    // Copy input from host to device
    size_t input_size = batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

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
        d_conv2_out, d_pool2_out,
        batch_size, CONV2_OUT, CONV2_H, CONV2_W, 2, 2
    );

    // Copy latent features to host
    cudaMemcpy(h_features, d_pool2_out,
               batch_size * LATENT_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
}

// ============================================================================
// COMPUTE LOSS - Host API
// ============================================================================

float GPUAutoencoderV2::compute_loss(const float* h_output, const float* h_target, int batch_size) {
    // Allocate temp device buffers if needed
    float* d_temp_output;
    float* d_temp_target;
    size_t size = batch_size * CONV5_OUT * CONV5_H * CONV5_W * sizeof(float);
    
    cudaMalloc(&d_temp_output, size);
    cudaMalloc(&d_temp_target, size);
    
    cudaMemcpy(d_temp_output, h_output, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_target, h_target, size, cudaMemcpyHostToDevice);
    
    float loss = gpu_v2::mse_loss_v2(d_temp_output, d_temp_target, batch_size, CONV5_OUT, CONV5_H, CONV5_W);
    
    cudaFree(d_temp_output);
    cudaFree(d_temp_target);
    
    return loss;
}

float GPUAutoencoderV2::compute_loss_gpu(const float* d_output, const float* d_target, int batch_size) {
    return gpu_v2::mse_loss_v2(d_output, d_target, batch_size, CONV5_OUT, CONV5_H, CONV5_W);
}

// ============================================================================
// BACKWARD PASS - Complete GPU implementation
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

// Conv backward data kernel (for layers without ReLU like Conv5)
__global__ void conv2d_backward_data_v2(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z;

    if (iw >= width || ih >= height || ic >= in_channels) return;

    float sum = 0.0f;

    #pragma unroll
    for (int oc = 0; oc < out_channels; ++oc) {
        int weight_base = (oc * in_channels + ic) * kernel_size * kernel_size;
        
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int oh = ih + padding - ky;
                int ow = iw + padding - kx;
                
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int out_idx = oc * height * width + oh * width + ow;
                    int flipped_ky = 2 - ky;
                    int flipped_kx = 2 - kx;
                    sum += grad_output[out_idx] * weights[weight_base + flipped_ky * 3 + flipped_kx];
                }
            }
        }
    }

    grad_input[ic * height * width + ih * width + iw] = sum;
}

void GPUAutoencoderV2::backward(const float* h_input, const float* h_target, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);
    allocate_gradients(batch_size);
    
    // Copy input and target from host to device
    size_t data_size = batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, data_size, cudaMemcpyHostToDevice);
    
    // Run forward pass to compute activations (needed for backward)
    forward_device(d_input, d_conv5_out, batch_size);
    
    // Call device backward
    backward_device(d_input, d_target, batch_size);
}

void GPUAutoencoderV2::backward_gpu(const float* d_in, const float* d_tgt, int batch_size) {
    current_batch_size = batch_size;
    allocate_activations(batch_size);
    allocate_gradients(batch_size);
    backward_device(d_in, d_tgt, batch_size);
}

void GPUAutoencoderV2::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
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

    // ========================================================================
    // Step 1: Compute MSE gradient on GPU (FAST!)
    // ========================================================================
    dim3 block(256);
    dim3 grid((output_size + 255) / 256);
    mse_gradient_kernel_v2<<<grid, block>>>(d_conv5_out, d_tgt, d_grad_conv5, scale, output_size);

    dim3 spatial_block(16, 16);
    dim3 wgrad_block(9);  // 3x3 kernel

    // ========================================================================
    // Step 2: Backward through Conv5 (no ReLU)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        // Weight gradients
        dim3 wgrad_grid5(CONV4_OUT, CONV5_OUT);
        gpu_v2::conv2d_weight_grad_optimized_kernel<<<wgrad_grid5, wgrad_block>>>(
            d_up2_out + b * CONV4_OUT * UP2_H * UP2_W,
            d_grad_conv5 + b * CONV5_OUT * CONV5_H * CONV5_W,
            d_grad_w5, CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1
        );
        
        // Data gradients → d_grad_up2
        dim3 data_grid5((UP2_W + 15) / 16, (UP2_H + 15) / 16, CONV4_OUT);
        conv2d_backward_data_v2<<<data_grid5, spatial_block>>>(
            d_grad_conv5 + b * CONV5_OUT * CONV5_H * CONV5_W,
            d_w5, d_grad_up2 + b * CONV4_OUT * UP2_H * UP2_W,
            CONV4_OUT, CONV5_OUT, UP2_H, UP2_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel<<<CONV5_OUT, 256, 256 * sizeof(float)>>>(
        d_grad_conv5, d_grad_b5, CONV5_OUT, CONV5_H * batch_size, CONV5_W
    );

    // ========================================================================
    // Step 3: Backward through Upsample2 (32x32 → 16x16)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 up_grid2((CONV4_W + 15) / 16, (CONV4_H + 15) / 16, CONV4_OUT);
        gpu_v2::upsample2d_backward_optimized_kernel<<<up_grid2, spatial_block>>>(
            d_grad_up2 + b * CONV4_OUT * UP2_H * UP2_W,
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            CONV4_OUT, CONV4_H, CONV4_W, UP2_H, UP2_W, 2
        );
    }

    // ========================================================================
    // Step 4: Backward through Conv4 + ReLU (FUSED)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        // Data gradients (fused with ReLU) → d_grad_up1
        dim3 data_grid4((UP1_W + 15) / 16, (UP1_H + 15) / 16, CONV3_OUT);
        gpu_v2::conv2d_relu_backward_fused_kernel<<<data_grid4, spatial_block>>>(
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_up1_out + b * CONV3_OUT * UP1_H * UP1_W,
            d_w4, d_conv4_out + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_grad_up1 + b * CONV3_OUT * UP1_H * UP1_W,
            CONV3_OUT, CONV4_OUT, UP1_H, UP1_W, 3, 1
        );
        
        // Weight gradients
        dim3 wgrad_grid4(CONV3_OUT, CONV4_OUT);
        gpu_v2::conv2d_weight_grad_optimized_kernel<<<wgrad_grid4, wgrad_block>>>(
            d_up1_out + b * CONV3_OUT * UP1_H * UP1_W,
            d_grad_conv4 + b * CONV4_OUT * CONV4_H * CONV4_W,
            d_grad_w4, CONV3_OUT, CONV4_OUT, UP1_H, UP1_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel<<<CONV4_OUT, 256, 256 * sizeof(float)>>>(
        d_grad_conv4, d_grad_b4, CONV4_OUT, CONV4_H * batch_size, CONV4_W
    );

    // ========================================================================
    // Step 5: Backward through Upsample1 (16x16 → 8x8)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 up_grid1((CONV3_W + 15) / 16, (CONV3_H + 15) / 16, CONV3_OUT);
        gpu_v2::upsample2d_backward_optimized_kernel<<<up_grid1, spatial_block>>>(
            d_grad_up1 + b * CONV3_OUT * UP1_H * UP1_W,
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            CONV3_OUT, CONV3_H, CONV3_W, UP1_H, UP1_W, 2
        );
    }

    // ========================================================================
    // Step 6: Backward through Conv3 + ReLU (FUSED)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 data_grid3((POOL2_W + 15) / 16, (POOL2_H + 15) / 16, CONV2_OUT);
        gpu_v2::conv2d_relu_backward_fused_kernel<<<data_grid3, spatial_block>>>(
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_w3, d_conv3_out + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_grad_pool2 + b * CONV2_OUT * POOL2_H * POOL2_W,
            CONV2_OUT, CONV3_OUT, POOL2_H, POOL2_W, 3, 1
        );
        
        dim3 wgrad_grid3(CONV2_OUT, CONV3_OUT);
        gpu_v2::conv2d_weight_grad_optimized_kernel<<<wgrad_grid3, wgrad_block>>>(
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_grad_conv3 + b * CONV3_OUT * CONV3_H * CONV3_W,
            d_grad_w3, CONV2_OUT, CONV3_OUT, POOL2_H, POOL2_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel<<<CONV3_OUT, 256, 256 * sizeof(float)>>>(
        d_grad_conv3, d_grad_b3, CONV3_OUT, CONV3_H * batch_size, CONV3_W
    );

    // ========================================================================
    // Step 7: Backward through MaxPool2 (16x16 → 8x8)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 pool_grid2((CONV2_W + 15) / 16, (CONV2_H + 15) / 16, CONV2_OUT);
        gpu_v2::maxpool2d_backward_optimized_kernel<<<pool_grid2, spatial_block>>>(
            d_grad_pool2 + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_conv2_out + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_pool2_out + b * CONV2_OUT * POOL2_H * POOL2_W,
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            CONV2_OUT, CONV2_H, CONV2_W, POOL2_H, POOL2_W, 2, 2
        );
    }

    // ========================================================================
    // Step 8: Backward through Conv2 + ReLU (FUSED)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 data_grid2((POOL1_W + 15) / 16, (POOL1_H + 15) / 16, CONV1_OUT);
        gpu_v2::conv2d_relu_backward_fused_kernel<<<data_grid2, spatial_block>>>(
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_w2, d_conv2_out + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_grad_pool1 + b * CONV1_OUT * POOL1_H * POOL1_W,
            CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1
        );
        
        dim3 wgrad_grid2(CONV1_OUT, CONV2_OUT);
        gpu_v2::conv2d_weight_grad_optimized_kernel<<<wgrad_grid2, wgrad_block>>>(
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_grad_conv2 + b * CONV2_OUT * CONV2_H * CONV2_W,
            d_grad_w2, CONV1_OUT, CONV2_OUT, POOL1_H, POOL1_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel<<<CONV2_OUT, 256, 256 * sizeof(float)>>>(
        d_grad_conv2, d_grad_b2, CONV2_OUT, CONV2_H * batch_size, CONV2_W
    );

    // ========================================================================
    // Step 9: Backward through MaxPool1 (32x32 → 16x16)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 pool_grid1((CONV1_W + 15) / 16, (CONV1_H + 15) / 16, CONV1_OUT);
        gpu_v2::maxpool2d_backward_optimized_kernel<<<pool_grid1, spatial_block>>>(
            d_grad_pool1 + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_conv1_out + b * CONV1_OUT * CONV1_H * CONV1_W,
            d_pool1_out + b * CONV1_OUT * POOL1_H * POOL1_W,
            d_grad_conv1 + b * CONV1_OUT * CONV1_H * CONV1_W,
            CONV1_OUT, CONV1_H, CONV1_W, POOL1_H, POOL1_W, 2, 2
        );
    }

    // ========================================================================
    // Step 10: Backward through Conv1 + ReLU (weights only, no input grad needed)
    // ========================================================================
    for (int b = 0; b < batch_size; ++b) {
        dim3 wgrad_grid1(INPUT_C, CONV1_OUT);
        gpu_v2::conv2d_weight_grad_optimized_kernel<<<wgrad_grid1, wgrad_block>>>(
            d_input + b * INPUT_C * INPUT_H * INPUT_W,
            d_grad_conv1 + b * CONV1_OUT * CONV1_H * CONV1_W,
            d_grad_w1, INPUT_C, CONV1_OUT, INPUT_H, INPUT_W, 3, 1
        );
    }
    gpu_v2::bias_grad_optimized_kernel<<<CONV1_OUT, 256, 256 * sizeof(float)>>>(
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
// SAVE/LOAD WEIGHTS
// ============================================================================

void GPUAutoencoderV2::save_weights(const std::string& filename) {
    float* h_w1 = new float[W1_SIZE];
    float* h_b1 = new float[B1_SIZE];
    float* h_w2 = new float[W2_SIZE];
    float* h_b2 = new float[B2_SIZE];
    float* h_w3 = new float[W3_SIZE];
    float* h_b3 = new float[B3_SIZE];
    float* h_w4 = new float[W4_SIZE];
    float* h_b4 = new float[B4_SIZE];
    float* h_w5 = new float[W5_SIZE];
    float* h_b5 = new float[B5_SIZE];

    cudaMemcpy(h_w1, d_w1, W1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, d_b1, B1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2, d_w2, W2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, d_b2, B2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w3, d_w3, W3_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3, d_b3, B3_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w4, d_w4, W4_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b4, d_b4, B4_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w5, d_w5, W5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b5, d_b5, B5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen(filename.c_str(), "wb");
    if (f) {
        fwrite(h_w1, sizeof(float), W1_SIZE, f);
        fwrite(h_b1, sizeof(float), B1_SIZE, f);
        fwrite(h_w2, sizeof(float), W2_SIZE, f);
        fwrite(h_b2, sizeof(float), B2_SIZE, f);
        fwrite(h_w3, sizeof(float), W3_SIZE, f);
        fwrite(h_b3, sizeof(float), B3_SIZE, f);
        fwrite(h_w4, sizeof(float), W4_SIZE, f);
        fwrite(h_b4, sizeof(float), B4_SIZE, f);
        fwrite(h_w5, sizeof(float), W5_SIZE, f);
        fwrite(h_b5, sizeof(float), B5_SIZE, f);
        fclose(f);
        printf("[GPUAutoencoderV2] Saved weights to %s\n", filename.c_str());
    }

    delete[] h_w1; delete[] h_b1;
    delete[] h_w2; delete[] h_b2;
    delete[] h_w3; delete[] h_b3;
    delete[] h_w4; delete[] h_b4;
    delete[] h_w5; delete[] h_b5;
}

void GPUAutoencoderV2::load_weights(const std::string& filename) {
    float* h_w1 = new float[W1_SIZE];
    float* h_b1 = new float[B1_SIZE];
    float* h_w2 = new float[W2_SIZE];
    float* h_b2 = new float[B2_SIZE];
    float* h_w3 = new float[W3_SIZE];
    float* h_b3 = new float[B3_SIZE];
    float* h_w4 = new float[W4_SIZE];
    float* h_b4 = new float[B4_SIZE];
    float* h_w5 = new float[W5_SIZE];
    float* h_b5 = new float[B5_SIZE];

    FILE* f = fopen(filename.c_str(), "rb");
    if (f) {
        size_t read_count = 0;
        read_count += fread(h_w1, sizeof(float), W1_SIZE, f);
        read_count += fread(h_b1, sizeof(float), B1_SIZE, f);
        read_count += fread(h_w2, sizeof(float), W2_SIZE, f);
        read_count += fread(h_b2, sizeof(float), B2_SIZE, f);
        read_count += fread(h_w3, sizeof(float), W3_SIZE, f);
        read_count += fread(h_b3, sizeof(float), B3_SIZE, f);
        read_count += fread(h_w4, sizeof(float), W4_SIZE, f);
        read_count += fread(h_b4, sizeof(float), B4_SIZE, f);
        read_count += fread(h_w5, sizeof(float), W5_SIZE, f);
        read_count += fread(h_b5, sizeof(float), B5_SIZE, f);
        fclose(f);

        cudaMemcpy(d_w1, h_w1, W1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, B1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, h_w2, W2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, B2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w3, h_w3, W3_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b3, h_b3, B3_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w4, h_w4, W4_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b4, h_b4, B4_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w5, h_w5, W5_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b5, h_b5, B5_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        printf("[GPUAutoencoderV2] Loaded weights from %s (%zu values)\n", 
               filename.c_str(), read_count);
    } else {
        printf("[GPUAutoencoderV2] Failed to open %s\n", filename.c_str());
    }

    delete[] h_w1; delete[] h_b1;
    delete[] h_w2; delete[] h_b2;
    delete[] h_w3; delete[] h_b3;
    delete[] h_w4; delete[] h_b4;
    delete[] h_w5; delete[] h_b5;
}
