#include "gpu1/gpu1_layers.cuh"

// Fused Conv2D + ReLU implementation with shared-memory tiling
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

    // Tile size (must match wrapper in gpu1_layers.cuh)
    const int TILE_W = 8;
    const int TILE_H = 8;

    // 2D thread indices within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Number of tiles along height (per image)
    int tiles_y = (height + TILE_H - 1) / TILE_H;

    // Output tile origin (top-left corner)
    int ow_base = blockIdx.x * TILE_W;

    // Decode batch and tile_y from blockIdx.y, and output channel from blockIdx.z
    int oc = blockIdx.z;                 // [0, out_channels)
    int b  = blockIdx.y / tiles_y;       // batch index
    int tile_y = blockIdx.y % tiles_y;   // tile index along height
    int oh_base = tile_y * TILE_H;

    // Actual output coordinate handled by this thread
    int ow = ow_base + tx;
    int oh = oh_base + ty;

    if (b >= batch_size) return;

    // Shared-memory tile for one input channel, with halo for 3x3 kernel
    const int SH_W = TILE_W + 2;  // +2 for pad on both sides
    const int SH_H = TILE_H + 2;
    __shared__ float sh_input[SH_W * SH_H];

    // Accumulator over all input channels
    float sum = 0.0f;

    // Start with bias for this output channel
    if (oh < height && ow < width) {
        sum = bias[oc];
    }

    // Loop over input channels, reusing the same shared tile
    for (int ic = 0; ic < in_channels; ic++) {
        // Cooperative load of the (TILE_H+2) x (TILE_W+2) patch for this (b, ic)
        for (int sh_y = ty; sh_y < SH_H; sh_y += blockDim.y) {
            int ih = oh_base + sh_y - pad;  // global input row
            for (int sh_x = tx; sh_x < SH_W; sh_x += blockDim.x) {
                int iw = ow_base + sh_x - pad;  // global input col

                float val = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = b * (in_channels * height * width) +
                                    ic * (height * width) + ih * width + iw;
                    val = input[input_idx];
                }

                sh_input[sh_y * SH_W + sh_x] = val;
            }
        }

        __syncthreads();

        // Each thread computes its output pixel using the shared tile
        if (oh < height && ow < width) {
            float partial = 0.0f;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    float v = sh_input[(ty + kh) * SH_W + (tx + kw)];
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    partial += v * weights[weight_idx];
                }
            }
            sum += partial;
        }

        __syncthreads();
    }

    // Apply ReLU in the same kernel (kernel fusion) and write output
    if (oh < height && ow < width) {
        float out_val = (sum > 0.0f) ? sum : 0.0f;
        int out_idx = b * (out_channels * height * width) +
                      oc * (height * width) + oh * width + ow;
        output[out_idx] = out_val;
    }
}
