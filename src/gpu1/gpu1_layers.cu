#include "gpu1/gpu1_layers.cuh"

// Fused Conv2D + ReLU using shared-memory tiling.
// Each block processes a spatial tile for a single (batch, out_channel) pair.
// Threads cooperatively load an input tile (with padding halo) into shared memory
// and then each thread computes one output element from that tile.
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

    // 2D thread index within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Output spatial coordinates handled by this thread
    const int out_x = blockIdx.x * blockDim.x + tx;
    const int out_y = blockIdx.y * blockDim.y + ty;

    // Map blockIdx.z to (batch, out_channel)
    int bo = blockIdx.z; // combined batch & out_channel
    const int oc = bo % out_channels;
    const int b  = bo / out_channels;
    if (b >= batch_size) {
        return;
    }

    // Valid output flag (threads outside image still participate in __syncthreads)
    const bool valid = (out_x < width) && (out_y < height);

    // Shared memory tile: (blockDim.y + 2*pad) x (blockDim.x + 2*pad)
    extern __shared__ float sh_input[];
    const int tile_w = blockDim.x + 2 * pad;
    const int tile_h = blockDim.y + 2 * pad;

    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    const int tile_size = tile_w * tile_h;

    // Base indices for this (batch, out_channel)
    const int input_batch_base  = b  * (in_channels * height * width);
    const int weight_oc_base    = oc * (in_channels * kernel_size * kernel_size);

    // Accumulate: bias + convolution
    float sum = bias[oc];

    // Loop over input channels, reusing the shared tile each time
    for (int ic = 0; ic < in_channels; ic++) {
        const int ic_input_offset = ic * (height * width);
        const int weight_ic_base  = weight_oc_base + ic * (kernel_size * kernel_size);

        // Cooperative load of input tile for this (b, ic)
        for (int t = tid; t < tile_size; t += num_threads) {
            int local_y = t / tile_w;
            int local_x = t % tile_w;

            int global_y = blockIdx.y * blockDim.y + local_y - pad;
            int global_x = blockIdx.x * blockDim.x + local_x - pad;

            float val = 0.0f;
            if (global_y >= 0 && global_y < height &&
                global_x >= 0 && global_x < width) {
                int input_idx = input_batch_base + ic_input_offset + global_y * width + global_x;
                val = input[input_idx];
            }
            sh_input[t] = val;
        }

        __syncthreads();

        if (valid) {
            // Center of this thread's 3x3 window in shared memory
            const int center_y = ty + pad;
            const int center_x = tx + pad;

            float tmp = 0.0f;

            // Unrolled 3x3 convolution over shared tile
            // Row -1
            {
                const int row = (center_y - 1) * tile_w;
                tmp += sh_input[row + (center_x - 1)] * weights[weight_ic_base + 0];
                tmp += sh_input[row + (center_x    )] * weights[weight_ic_base + 1];
                tmp += sh_input[row + (center_x + 1)] * weights[weight_ic_base + 2];
            }
            // Row 0
            {
                const int row = center_y * tile_w;
                tmp += sh_input[row + (center_x - 1)] * weights[weight_ic_base + 3];
                tmp += sh_input[row + (center_x    )] * weights[weight_ic_base + 4];
                tmp += sh_input[row + (center_x + 1)] * weights[weight_ic_base + 5];
            }
            // Row +1
            {
                const int row = (center_y + 1) * tile_w;
                tmp += sh_input[row + (center_x - 1)] * weights[weight_ic_base + 6];
                tmp += sh_input[row + (center_x    )] * weights[weight_ic_base + 7];
                tmp += sh_input[row + (center_x + 1)] * weights[weight_ic_base + 8];
            }

            sum += tmp;
        }

        __syncthreads(); // ensure all threads finished using current tile before next ic
    }

    if (valid) {
        // Fused ReLU activation
        float out_val = (sum > 0.0f) ? sum : 0.0f;

        int output_idx =
            b  * (out_channels * height * width) +
            oc * (height * width) +
            out_y * width + out_x;

        output[output_idx] = out_val;
    }
}
