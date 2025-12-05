// ============================================================================
// GPU Optimized Layers V2 - Kernel Optimization Bundle
// ============================================================================
// Techniques applied:
//   1. Kernel Fusion (Conv + Bias + ReLU) - Reduces global memory writes
//   2. Loop Unrolling (3x3 kernel) - Reduces loop overhead, increases ILP
//   3. Vectorized Memory Access (float4) - Increases memory bandwidth
// ============================================================================

#include "gpu/gpu_layers_v2.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace gpu_v2 {

// ============================================================================
// TECHNIQUE 1: KERNEL FUSION - Conv2D + Bias + ReLU
// ============================================================================
// Benefits:
//   - Reduces 3 kernel launches to 1
//   - Eliminates 2 global memory write/read cycles
//   - Better cache utilization
// ============================================================================

__global__ void conv2d_bias_relu_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
) {
    // Output position
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (ow >= width || oh >= height || oc >= out_channels) return;

    float sum = 0.0f;
    int half_k = kernel_size / 2;

    // TECHNIQUE 2: Loop Unrolling for 3x3 kernel
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        // Manually unrolled 3x3 convolution
        int input_base = ic * height * width;
        int weight_base = (oc * in_channels + ic) * kernel_size * kernel_size;

        // Row 0: ky = 0
        int iy0 = oh * stride - padding + 0;
        if (iy0 >= 0 && iy0 < height) {
            // kx = 0
            int ix00 = ow * stride - padding + 0;
            if (ix00 >= 0 && ix00 < width) {
                sum += input[input_base + iy0 * width + ix00] * weights[weight_base + 0];
            }
            // kx = 1
            int ix01 = ow * stride - padding + 1;
            if (ix01 >= 0 && ix01 < width) {
                sum += input[input_base + iy0 * width + ix01] * weights[weight_base + 1];
            }
            // kx = 2
            int ix02 = ow * stride - padding + 2;
            if (ix02 >= 0 && ix02 < width) {
                sum += input[input_base + iy0 * width + ix02] * weights[weight_base + 2];
            }
        }

        // Row 1: ky = 1
        int iy1 = oh * stride - padding + 1;
        if (iy1 >= 0 && iy1 < height) {
            // kx = 0
            int ix10 = ow * stride - padding + 0;
            if (ix10 >= 0 && ix10 < width) {
                sum += input[input_base + iy1 * width + ix10] * weights[weight_base + 3];
            }
            // kx = 1
            int ix11 = ow * stride - padding + 1;
            if (ix11 >= 0 && ix11 < width) {
                sum += input[input_base + iy1 * width + ix11] * weights[weight_base + 4];
            }
            // kx = 2
            int ix12 = ow * stride - padding + 2;
            if (ix12 >= 0 && ix12 < width) {
                sum += input[input_base + iy1 * width + ix12] * weights[weight_base + 5];
            }
        }

        // Row 2: ky = 2
        int iy2 = oh * stride - padding + 2;
        if (iy2 >= 0 && iy2 < height) {
            // kx = 0
            int ix20 = ow * stride - padding + 0;
            if (ix20 >= 0 && ix20 < width) {
                sum += input[input_base + iy2 * width + ix20] * weights[weight_base + 6];
            }
            // kx = 1
            int ix21 = ow * stride - padding + 1;
            if (ix21 >= 0 && ix21 < width) {
                sum += input[input_base + iy2 * width + ix21] * weights[weight_base + 7];
            }
            // kx = 2
            int ix22 = ow * stride - padding + 2;
            if (ix22 >= 0 && ix22 < width) {
                sum += input[input_base + iy2 * width + ix22] * weights[weight_base + 8];
            }
        }
    }

    // FUSED: Add bias and apply ReLU in same kernel
    sum += bias[oc];
    sum = fmaxf(0.0f, sum);  // ReLU

    output[oc * height * width + oh * width + ow] = sum;
}

// Conv2D + Bias without ReLU (for final decoder layer)
__global__ void conv2d_bias_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (ow >= width || oh >= height || oc >= out_channels) return;

    float sum = 0.0f;

    // TECHNIQUE 2: Loop Unrolling for 3x3 kernel
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        int input_base = ic * height * width;
        int weight_base = (oc * in_channels + ic) * kernel_size * kernel_size;

        // Unrolled 3x3 convolution (same as above but without ReLU)
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int iy = oh * stride - padding + ky;
            if (iy >= 0 && iy < height) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = ow * stride - padding + kx;
                    if (ix >= 0 && ix < width) {
                        sum += input[input_base + iy * width + ix] 
                             * weights[weight_base + ky * 3 + kx];
                    }
                }
            }
        }
    }

    // Add bias only (no ReLU)
    sum += bias[oc];
    output[oc * height * width + oh * width + ow] = sum;
}

// ============================================================================
// OPTIMIZED MAXPOOL KERNEL
// ============================================================================

__global__ void maxpool2d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int pool_size, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (ow >= out_width || oh >= out_height || c >= channels) return;

    int in_row_start = oh * stride;
    int in_col_start = ow * stride;

    float max_val = -1e30f;
    int input_base = c * in_height * in_width;

    // TECHNIQUE 2: Unroll 2x2 pooling
    #pragma unroll
    for (int ph = 0; ph < 2; ++ph) {
        #pragma unroll
        for (int pw = 0; pw < 2; ++pw) {
            int ih = in_row_start + ph;
            int iw = in_col_start + pw;
            if (ih < in_height && iw < in_width) {
                float val = input[input_base + ih * in_width + iw];
                max_val = fmaxf(max_val, val);
            }
        }
    }

    output[c * out_height * out_width + oh * out_width + ow] = max_val;
}

// ============================================================================
// OPTIMIZED UPSAMPLE KERNEL
// ============================================================================

__global__ void upsample2d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int scale_factor
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (ow >= out_width || oh >= out_height || c >= channels) return;

    // Nearest neighbor: map output to input
    int ih = oh / scale_factor;
    int iw = ow / scale_factor;

    float val = input[c * in_height * in_width + ih * in_width + iw];
    output[c * out_height * out_width + oh * out_width + ow] = val;
}

// ============================================================================
// TECHNIQUE 3: VECTORIZED MSE LOSS WITH float4
// ============================================================================

__global__ void mse_loss_optimized_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;  // Process 4 elements per thread
    
    float thread_sum = 0.0f;
    
    // TECHNIQUE 3: Vectorized access - process 4 elements at a time
    int vec_size = size / 4;
    if (idx < vec_size) {
        float4 out_vec = reinterpret_cast<const float4*>(output)[idx];
        float4 tgt_vec = reinterpret_cast<const float4*>(target)[idx];
        
        float diff0 = out_vec.x - tgt_vec.x;
        float diff1 = out_vec.y - tgt_vec.y;
        float diff2 = out_vec.z - tgt_vec.z;
        float diff3 = out_vec.w - tgt_vec.w;
        
        thread_sum = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    
    // Handle remaining elements
    int remaining_start = vec_size * 4;
    int remaining_idx = remaining_start + tid;
    if (remaining_idx < size && tid < (size - remaining_start)) {
        float diff = output[remaining_idx] - target[remaining_idx];
        thread_sum += diff * diff;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

// ============================================================================
// FUSED BACKWARD: Conv + ReLU Backward
// ============================================================================

__global__ void conv2d_relu_backward_fused_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ forward_output,
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
    int half_k = kernel_size / 2;

    #pragma unroll
    for (int oc = 0; oc < out_channels; ++oc) {
        int weight_base = (oc * in_channels + ic) * kernel_size * kernel_size;
        
        // Unrolled 3x3
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int oh = ih + padding - ky;
                int ow = iw + padding - kx;
                
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int out_idx = oc * height * width + oh * width + ow;
                    
                    // FUSED: ReLU backward (gradient is 0 if output was 0)
                    float relu_grad = (forward_output[out_idx] > 0.0f) ? 1.0f : 0.0f;
                    float grad = grad_output[out_idx] * relu_grad;
                    
                    // Flip kernel for transpose convolution
                    int flipped_ky = 2 - ky;
                    int flipped_kx = 2 - kx;
                    
                    sum += grad * weights[weight_base + flipped_ky * 3 + flipped_kx];
                }
            }
        }
    }

    grad_input[ic * height * width + ih * width + iw] = sum;
}

// ============================================================================
// OPTIMIZED WEIGHT GRADIENT
// ============================================================================

__global__ void conv2d_weight_grad_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding
) {
    int kx = threadIdx.x % kernel_size;
    int ky = threadIdx.x / kernel_size;
    int ic = blockIdx.x;
    int oc = blockIdx.y;

    if (kx >= kernel_size || ky >= kernel_size) return;

    float sum = 0.0f;

    // Sum over all spatial positions
    for (int oh = 0; oh < height; ++oh) {
        for (int ow = 0; ow < width; ++ow) {
            int ih = oh - padding + ky;
            int iw = ow - padding + kx;
            
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                float in_val = input[ic * height * width + ih * width + iw];
                float grad_val = grad_output[oc * height * width + oh * width + ow];
                sum += in_val * grad_val;
            }
        }
    }

    int weight_idx = (oc * in_channels + ic) * kernel_size * kernel_size + ky * kernel_size + kx;
    atomicAdd(&grad_weights[weight_idx], sum);
}

// ============================================================================
// OPTIMIZED BIAS GRADIENT WITH PARALLEL REDUCTION
// ============================================================================

__global__ void bias_grad_optimized_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int out_channels, int height, int width
) {
    extern __shared__ float sdata[];
    
    int oc = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    
    // Each thread sums multiple elements
    float thread_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        thread_sum += grad_output[oc * spatial_size + i];
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&grad_bias[oc], sdata[0]);
    }
}

// ============================================================================
// MAXPOOL BACKWARD
// ============================================================================

__global__ void maxpool2d_backward_optimized_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int pool_size, int stride
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (iw >= in_width || ih >= in_height || c >= channels) return;

    // Find which output position this input contributes to
    int oh = ih / stride;
    int ow = iw / stride;

    if (oh >= out_height || ow >= out_width) {
        grad_input[c * in_height * in_width + ih * in_width + iw] = 0.0f;
        return;
    }

    // Check if this was the max element
    float in_val = input[c * in_height * in_width + ih * in_width + iw];
    float out_val = output[c * out_height * out_width + oh * out_width + ow];
    
    float grad = 0.0f;
    if (fabsf(in_val - out_val) < 1e-6f) {
        grad = grad_output[c * out_height * out_width + oh * out_width + ow];
    }

    grad_input[c * in_height * in_width + ih * in_width + iw] = grad;
}

// ============================================================================
// UPSAMPLE BACKWARD
// ============================================================================

__global__ void upsample2d_backward_optimized_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int scale_factor
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (iw >= in_width || ih >= in_height || c >= channels) return;

    // Sum gradients from all output positions that map to this input
    float sum = 0.0f;
    int out_base = c * out_height * out_width;
    
    #pragma unroll
    for (int sy = 0; sy < 2; ++sy) {
        #pragma unroll
        for (int sx = 0; sx < 2; ++sx) {
            int oh = ih * scale_factor + sy;
            int ow = iw * scale_factor + sx;
            if (oh < out_height && ow < out_width) {
                sum += grad_output[out_base + oh * out_width + ow];
            }
        }
    }

    grad_input[c * in_height * in_width + ih * in_width + iw] = sum;
}

// ============================================================================
// TECHNIQUE 3: VECTORIZED SGD UPDATE
// ============================================================================

__global__ void sgd_update_vectorized_kernel(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float learning_rate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    // Process 4 elements at a time using float4
    if (vec_idx + 3 < size) {
        float4* w_ptr = reinterpret_cast<float4*>(weights + vec_idx);
        const float4* g_ptr = reinterpret_cast<const float4*>(gradients + vec_idx);
        
        float4 w = *w_ptr;
        float4 g = *g_ptr;
        
        w.x -= learning_rate * g.x;
        w.y -= learning_rate * g.y;
        w.z -= learning_rate * g.z;
        w.w -= learning_rate * g.w;
        
        *w_ptr = w;
    } else {
        // Handle remaining elements
        for (int i = vec_idx; i < size && i < vec_idx + 4; ++i) {
            weights[i] -= learning_rate * gradients[i];
        }
    }
}

// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================

void conv2d_bias_relu_forward_v2(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        out_channels
    );

    for (int b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * in_channels * height * width;
        float* batch_output = output + b * out_channels * height * width;
        
        conv2d_bias_relu_fused_kernel<<<grid, block>>>(
            batch_input, weights, bias, batch_output,
            in_channels, out_channels,
            height, width,
            kernel_size, padding, stride
        );
    }
    cudaDeviceSynchronize();
}

void conv2d_bias_forward_v2(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_size, int padding, int stride
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        out_channels
    );

    for (int b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * in_channels * height * width;
        float* batch_output = output + b * out_channels * height * width;
        
        conv2d_bias_fused_kernel<<<grid, block>>>(
            batch_input, weights, bias, batch_output,
            in_channels, out_channels,
            height, width,
            kernel_size, padding, stride
        );
    }
    cudaDeviceSynchronize();
}

void maxpool2d_forward_v2(
    const float* input, float* output,
    int batch_size, int channels,
    int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = in_height / stride;
    int out_width = in_width / stride;

    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        channels
    );

    for (int b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * channels * in_height * in_width;
        float* batch_output = output + b * channels * out_height * out_width;
        
        maxpool2d_optimized_kernel<<<grid, block>>>(
            batch_input, batch_output,
            channels, in_height, in_width,
            out_height, out_width,
            pool_size, stride
        );
    }
    cudaDeviceSynchronize();
}

void upsample2d_forward_v2(
    const float* input, float* output,
    int batch_size, int channels,
    int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;

    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        channels
    );

    for (int b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * channels * in_height * in_width;
        float* batch_output = output + b * channels * out_height * out_width;
        
        upsample2d_optimized_kernel<<<grid, block>>>(
            batch_input, batch_output,
            channels, in_height, in_width,
            out_height, out_width,
            scale_factor
        );
    }
    cudaDeviceSynchronize();
}

float mse_loss_v2(
    const float* output, const float* target,
    int batch_size, int channels, int height, int width
) {
    int size = batch_size * channels * height * width;
    
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));
    
    int block_size = 256;
    int num_blocks = (size / 4 + block_size - 1) / block_size;
    int shared_mem = block_size * sizeof(float);
    
    mse_loss_optimized_kernel<<<num_blocks, block_size, shared_mem>>>(
        output, target, d_loss, size
    );
    
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    
    return h_loss / size;
}

void sgd_update_v2(
    float* weights, const float* gradients,
    float learning_rate, int size
) {
    int block_size = 256;
    int num_elements = (size + 3) / 4;  // Number of float4 elements
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    sgd_update_vectorized_kernel<<<num_blocks, block_size>>>(
        weights, gradients, learning_rate, size
    );
    cudaDeviceSynchronize();
}

}  // namespace gpu_v2
