#include "gpu1/gpu1_layers.cuh"

// Fused Conv2D + ReLU (kernel-level fusion without shared memory)
// Each thread computes one output element: Conv2D 3x3 + ReLU inline
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
) {
    const int kernel_size = 3;
    const int pad = 1;

    // Linear thread index mapped to output (b, oc, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, oc, h, w)
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);

    // Accumulate: bias + convolution
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

    // Fused ReLU activation
    float out_val = (sum > 0.0f) ? sum : 0.0f;
    output[idx] = out_val;
}
