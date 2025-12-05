// ============================================================================
// GPU Layer Operations V2 - Optimized Kernels
// Optimizations:
//   - Kernel Fusion: Conv+Bias+ReLU in single kernel
//   - Loop Unrolling: Manual unrolling for 3x3 convolutions
//   - Optimized Block Dimensions: Tuned for Tesla T4
// ============================================================================

#include "gpu/gpu_layers_v2.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

namespace gpu_v2 {

// ============================================================================
// FUSED FORWARD KERNELS (Conv + Bias + ReLU)
// ============================================================================

__global__ void conv2d_bias_relu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    
    if (idx >= total) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[oc];
    
    // LOOP UNROLLING for 3x3 kernel
    if (kernel_size == 3) {
        for (int ic = 0; ic < in_channels; ic++) {
            // Unroll 3x3 manually
            int ih, iw, input_idx, weight_idx;
            
            // Row 0
            ih = h - padding; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 0;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 1;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 2;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            // Row 1
            ih = h - padding + 1; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 3;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 1; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 4;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 1; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 5;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            // Row 2
            ih = h - padding + 2; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 6;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 2; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 7;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 2; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 8;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // FUSED ReLU activation
    output[idx] = (sum > 0.0f) ? sum : 0.0f;
}

__global__ void conv2d_bias_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    
    if (idx >= total) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[oc];
    
    // LOOP UNROLLING for 3x3
    if (kernel_size == 3) {
        for (int ic = 0; ic < in_channels; ic++) {
            int ih, iw, input_idx, weight_idx;
            
            // Unroll 3x3 (same as above but no ReLU)
            ih = h - padding; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 0;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 1;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 2;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 3;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 4;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 5;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 6;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 7;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 8;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    output[idx] = sum;  // No ReLU
}

void conv2d_bias_relu_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
) {
    int total = batch_size * out_channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv2d_bias_relu_forward_kernel<<<blocks, threads>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding
    );
}

void conv2d_bias_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
) {
    int total = batch_size * out_channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv2d_bias_forward_kernel<<<blocks, threads>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding
    );
}

// ============================================================================
// MAXPOOL2D FORWARD
// ============================================================================

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = in_height / stride;
    int out_width = in_width / stride;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    float max_val = -1e38f;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = h * stride + ph;
            int iw = w * stride + pw;
            int input_idx = b * (channels * in_height * in_width) +
                           c * (in_height * in_width) + ih * in_width + iw;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }
    
    output[idx] = max_val;
}

void maxpool2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width,
    int pool_size, int stride
) {
    int out_height = height / stride;
    int out_width = width / stride;
    int total = batch_size * channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    maxpool2d_forward_kernel<<<blocks, threads>>>(
        d_input, d_output, batch_size, channels, height, width, pool_size, stride
    );
}

// ============================================================================
// UPSAMPLE2D FORWARD
// ============================================================================

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width, int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    int ih = h / scale_factor;
    int iw = w / scale_factor;
    int input_idx = b * (channels * in_height * in_width) +
                   c * (in_height * in_width) + ih * in_width + iw;
    
    output[idx] = input[input_idx];
}

void upsample2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width, int scale_factor
) {
    int out_height = height * scale_factor;
    int out_width = width * scale_factor;
    int total = batch_size * channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    upsample2d_forward_kernel<<<blocks, threads>>>(
        d_input, d_output, batch_size, channels, height, width, scale_factor
    );
}

// ============================================================================
// BACKWARD HELPERS (REUSE BASELINE-STYLE KERNELS WITH BATCH DIMENSION)
// ============================================================================

// Compute dL/doutput for MSE loss: 2*(y - y_hat)/N
__global__ void mse_gradient_kernel_v2(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float diff = output[idx] - target[idx];
        grad_output[idx] = 2.0f * diff / total_size;
    }
}

// Thin wrappers that mirror the baseline GPU implementation but keep them
// inside the gpu_v2 namespace for better reuse from GPUAutoencoderV2.

void conv2d_backward_input_v2(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    conv2d_backward_input(
        d_weights,
        d_grad_output,
        d_grad_input,
        batch_size,
        in_channels,
        out_channels,
        height,
        width
    );
}

void conv2d_backward_full_v2(
    const float* d_input,
    const float* d_weights,
    const float* d_conv_output,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    // Use baseline conv2d_backward which computes both input and weight grads.
    conv2d_backward(
        d_input,
        d_weights,
        d_grad_output,
        d_grad_input,
        d_grad_weights,
        d_grad_bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width
    );
}

void maxpool2d_backward_v2(
    const float* d_grad_output,
    const float* d_input,
    const float* d_output,
    float* d_grad_input,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pool_size,
    int stride
) {
    maxpool2d_backward(
        d_input,
        d_output,
        d_grad_output,
        d_grad_input,
        batch_size,
        channels,
        in_height,
        in_width
    );
}

void upsample2d_backward_v2(
    const float* d_grad_output,
    float* d_grad_input,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int scale_factor
) {
    upsample2d_backward(
        d_grad_output,
        d_grad_input,
        batch_size,
        channels,
        in_height,
        in_width
    );
}

// ============================================================================
// LOSS AND OPTIMIZER
// ============================================================================

__global__ void mse_loss_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ partial_loss,
    int size
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float diff = (idx < size) ? (output[idx] - target[idx]) : 0.0f;
    shared[tid] = diff * diff;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = shared[0];
    }
}

float mse_loss_v2(
    const float* d_output, const float* d_target,
    int batch_size, int channels, int height, int width
) {
    int size = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    mse_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_output, d_target, d_partial, size
    );
    
    float* h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_loss += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_partial);
    
    return total_loss / size;
}

__global__ void sgd_update_kernel(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

void sgd_update_v2(float* d_weights, const float* d_gradients, float lr, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(d_weights, d_gradients, lr, size);
}

} // namespace gpu_v2
