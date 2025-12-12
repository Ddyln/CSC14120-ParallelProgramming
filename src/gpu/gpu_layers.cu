#include "gpu/gpu_layers.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// Forward Pass Kernels
// ============================================================================

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
) {
    // 3x3 convolution with padding=1, stride=1
    const int kernel_size = 3;
    const int pad = 1;

    // Calculate global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, oc, h, w)
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
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

__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0.0f) ? data[idx] : 0.0f;
    }
}

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int pool_size = 2;
    int out_height = in_height / 2;
    int out_width = in_width / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, c, h, w)
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float max_val = -1e38f;

    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = h * pool_size + ph;
            int iw = w * pool_size + pw;
            int input_idx = b * (channels * in_height * in_width) +
                            c * (in_height * in_width) + ih * in_width + iw;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }

    output[idx] = max_val;
}

__global__ void upsample2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, c, h, w)
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    // Nearest neighbor: map output coord to input coord
    int ih = h / 2;
    int iw = w / 2;

    int input_idx = b * (channels * in_height * in_width) +
                    c * (in_height * in_width) + ih * in_width + iw;

    output[idx] = input[input_idx];
}

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
) {
    const int kernel_size = 3;
    const int pad = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_channels * height * width;

    if (idx >= total_inputs) return;

    // Decompose linear index into (b, ic, ih, iw)
    int iw = idx % width;
    int ih = (idx / width) % height;
    int ic = (idx / (width * height)) % in_channels;
    int b = idx / (width * height * in_channels);

    float sum = 0.0f;

    // For each output channel and kernel position that affects this input
    for (int oc = 0; oc < out_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Output position that used this input with this kernel position
                int oh = ih - kh + pad;
                int ow = iw - kw + pad;

                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int output_idx = b * (out_channels * height * width) +
                                     oc * (height * width) + oh * width + ow;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += dL_doutput[output_idx] * weights[weight_idx];
                }
            }
        }
    }

    dL_dinput[idx] = sum;
}

__global__ void conv2d_backward_weights_kernel(
    const float* input,
    const float* dL_doutput,
    float* dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int kernel_size = 3;
    const int pad = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;

    if (idx >= total_weights) return;

    // Decompose linear index into (oc, ic, kh, kw)
    int kw = idx % kernel_size;
    int kh = (idx / kernel_size) % kernel_size;
    int ic = (idx / (kernel_size * kernel_size)) % in_channels;
    int oc = idx / (kernel_size * kernel_size * in_channels);

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = b * (in_channels * height * width) +
                                    ic * (height * width) + ih * width + iw;
                    int output_idx = b * (out_channels * height * width) +
                                     oc * (height * width) + oh * width + ow;
                    sum += input[input_idx] * dL_doutput[output_idx];
                }
            }
        }
    }

    dL_dweights[idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(
    const float* dL_doutput,
    float* dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= out_channels) return;

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = b * (out_channels * height * width) +
                          oc * (height * width) + h * width + w;
                sum += dL_doutput[idx];
            }
        }
    }

    dL_dbias[oc] = sum;
}

__global__ void relu_backward_kernel(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_dinput[idx] = (output[idx] > 0.0f) ? dL_doutput[idx] : 0.0f;
    }
}

__global__ void maxpool2d_backward_kernel(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int pool_size = 2;
    int out_height = in_height / 2;
    int out_width = in_width / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, c, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float out_val = output[idx];
    float grad = dL_doutput[idx];

    // Find which input position had the max value and pass gradient there
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * pool_size + ph;
            int iw = ow * pool_size + pw;
            int input_idx = b * (channels * in_height * in_width) +
                            c * (in_height * in_width) + ih * in_width + iw;
            
            // Pass gradient to the position that had the max value
            if (input[input_idx] == out_val) {
                atomicAdd(&dL_dinput[input_idx], grad);
            }
        }
    }
}

__global__ void upsample2d_backward_kernel(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * channels * in_height * in_width;

    if (idx >= total_inputs) return;

    // Decompose linear index into (b, c, ih, iw)
    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int c = (idx / (in_width * in_height)) % channels;
    int b = idx / (in_width * in_height * channels);

    // Sum gradients from the 4 output positions that came from this input
    float sum = 0.0f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int oh = ih * 2 + dh;
            int ow = iw * 2 + dw;
            int output_idx = b * (channels * out_height * out_width) +
                             c * (out_height * out_width) + oh * out_width + ow;
            sum += dL_doutput[output_idx];
        }
    }

    dL_dinput[idx] = sum;
}

// ============================================================================
// Loss Kernels
// ============================================================================

__global__ void mse_loss_gradient_kernel(
    const float* output,
    const float* target,
    float* dL_doutput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_doutput[idx] = 2.0f * (output[idx] - target[idx]) / size;
    }
}

__global__ void mse_loss_kernel(
    const float* output,
    const float* target,
    float* partial_sum,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        float val = diff * diff;
        atomicAdd(partial_sum, val);
    }
}

// ============================================================================
// Weight Update Kernels
// ============================================================================

__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    float clip_value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = gradients[idx];
        // Gradient clipping
        if (grad > clip_value) grad = clip_value;
        if (grad < -clip_value) grad = -clip_value;
        weights[idx] -= learning_rate * grad;
    }
}

__global__ void zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

// ============================================================================
// Wrapper Functions
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
) {
    int total_outputs = batch_size * out_channels * height * width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    conv2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_forward(float* d_data, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_forward_kernel<<<grid_size, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    maxpool2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    upsample2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

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
) {
    int block_size = 256;

    // Backward for input gradients
    int total_inputs = batch_size * in_channels * height * width;
    int grid_size_input = (total_inputs + block_size - 1) / block_size;
    conv2d_backward_input_kernel<<<grid_size_input, block_size>>>(
        d_weights, d_dL_doutput, d_dL_dinput,
        batch_size, in_channels, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward for weight gradients
    int total_weights = out_channels * in_channels * 9;  // 3x3 kernel
    int grid_size_weights = (total_weights + block_size - 1) / block_size;
    conv2d_backward_weights_kernel<<<grid_size_weights, block_size>>>(
        d_input, d_dL_doutput, d_dL_dweights,
        batch_size, in_channels, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward for bias gradients
    int grid_size_bias = (out_channels + block_size - 1) / block_size;
    conv2d_backward_bias_kernel<<<grid_size_bias, block_size>>>(
        d_dL_doutput, d_dL_dbias,
        batch_size, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward(
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_backward_kernel<<<grid_size, block_size>>>(
        d_output, d_dL_doutput, d_dL_dinput, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    // Zero the input gradient first
    int total_inputs = batch_size * channels * in_height * in_width;
    gpu_zero(d_dL_dinput, total_inputs);

    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    maxpool2d_backward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_dL_doutput, d_dL_dinput,
        batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_backward(
    const float* d_dL_doutput,
    float* d_dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int total_inputs = batch_size * channels * in_height * in_width;
    int block_size = 256;
    int grid_size = (total_inputs + block_size - 1) / block_size;

    upsample2d_backward_kernel<<<grid_size, block_size>>>(
        d_dL_doutput, d_dL_dinput,
        batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

float gpu_mse_loss(const float* d_output, const float* d_target, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Single global accumulator to avoid shared memory
    float* d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(float)));

    mse_loss_kernel<<<grid_size, block_size>>>(
        d_output, d_target, d_partial_sum, size
    );
    CUDA_CHECK(cudaGetLastError());

    float h_total = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_total, d_partial_sum, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_partial_sum));

    return h_total / size;
}

void gpu_mse_loss_gradient(
    const float* d_output,
    const float* d_target,
    float* d_dL_doutput,
    int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    mse_loss_gradient_kernel<<<grid_size, block_size>>>(
        d_output, d_target, d_dL_doutput, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_sgd_update(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    float clip_value,
    int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    sgd_update_kernel<<<grid_size, block_size>>>(
        d_weights, d_gradients, learning_rate, clip_value, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_zero(float* d_data, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    zero_kernel<<<grid_size, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}
