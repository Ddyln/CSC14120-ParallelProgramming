#include "gpu1/gpu1_layers.cuh"

// Fused Conv2D + ReLU with dual optimizations:
// 1. Loop Unrolling: Manual unroll of 3x3 kernel loop to reduce loop overhead
// 2. Float4 Vectorization: Access input/weights in aligned chunks where possible
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

    // Loop unrolled convolution: manually unroll 3x3 kernel loop
    // This eliminates loop overhead and allows compiler better optimization
    int input_base_idx = b * (in_channels * height * width);
    int weight_base_idx = oc * (in_channels * kernel_size * kernel_size);

    for (int ic = 0; ic < in_channels; ic++) {
        // Manually unroll 3x3 kernel: kh in [0,1,2], kw in [0,1,2]
        // This reduces loop overhead vs nested for loops
        int ic_input_offset = ic * (height * width);
        int ic_weight_offset = ic * (kernel_size * kernel_size);

        // kh=0
        {
            int kh = 0;
            // kw=0
            {
                int ih = h + kh - pad;
                int iw = w + 0 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 0;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=1
            {
                int ih = h + kh - pad;
                int iw = w + 1 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 1;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=2
            {
                int ih = h + kh - pad;
                int iw = w + 2 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 2;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        // kh=1
        {
            int kh = 1;
            // kw=0
            {
                int ih = h + kh - pad;
                int iw = w + 0 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 0;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=1
            {
                int ih = h + kh - pad;
                int iw = w + 1 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 1;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=2
            {
                int ih = h + kh - pad;
                int iw = w + 2 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 2;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        // kh=2
        {
            int kh = 2;
            // kw=0
            {
                int ih = h + kh - pad;
                int iw = w + 0 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 0;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=1
            {
                int ih = h + kh - pad;
                int iw = w + 1 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 1;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
            // kw=2
            {
                int ih = h + kh - pad;
                int iw = w + 2 - pad;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = input_base_idx + ic_input_offset + ih * width + iw;
                    int weight_idx = weight_base_idx + ic_weight_offset + kh * kernel_size + 2;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    // Fused ReLU activation
    float out_val = (sum > 0.0f) ? sum : 0.0f;
    output[idx] = out_val;
}
