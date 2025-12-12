#include "gpu2/gpu2_autoencoder.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <cstring>
#include <random>

#include "gpu2/gpu2_layers.cuh"

// Weight sizes
constexpr int W1_SIZE = 256 * 3 * 3 * 3;      // 6,912
constexpr int B1_SIZE = 256;
constexpr int W2_SIZE = 128 * 256 * 3 * 3;    // 294,912
constexpr int B2_SIZE = 128;
constexpr int W3_SIZE = 128 * 128 * 3 * 3;    // 147,456
constexpr int B3_SIZE = 128;
constexpr int W4_SIZE = 256 * 128 * 3 * 3;    // 294,912
constexpr int B4_SIZE = 256;
constexpr int W5_SIZE = 3 * 256 * 3 * 3;      // 6,912
constexpr int B5_SIZE = 3;

// Xavier weight initialization
static void init_weights_xavier(float* weights, int in_channels, int out_channels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (in_channels + out_channels));
    std::uniform_real_distribution<float> dis(-limit, limit);

    int kernel_size = 3 * 3;
    int total_weights = out_channels * in_channels * kernel_size;

    for (int i = 0; i < total_weights; i++) {
        weights[i] = dis(gen);
    }
}

GPU2Autoencoder::GPU2Autoencoder() {
    // Host pointers - will use pinned memory
    h_w1 = h_b1 = h_w2 = h_b2 = h_w3 = h_b3 = h_w4 = h_b4 = h_w5 = h_b5 = nullptr;

    // Device weight pointers
    d_w1 = d_b1 = d_w2 = d_b2 = d_w3 = d_b3 = d_w4 = d_b4 = d_w5 = d_b5 = nullptr;

    // Device gradient pointers
    d_dw1 = d_db1 = d_dw2 = d_db2 = d_dw3 = d_db3 = d_dw4 = d_db4 = d_dw5 = d_db5 = nullptr;

    // Device activation pointers
    d_input = d_target = nullptr;
    d_act1 = d_pool1 = d_act2 = d_act3 = nullptr;
    d_conv3_out = d_up1 = d_act4 = d_up2 = d_act5 = nullptr;

    // Device gradient buffers
    d_dL_dact5 = d_dL_dup2 = d_dL_dact4 = d_dL_dup1 = nullptr;
    d_dL_dconv3 = d_dL_dact3 = d_dL_dact2 = d_dL_dpool1 = nullptr;
    d_dL_dact1 = d_dL_dinput = nullptr;

    // Initialize CUDA streams for multi-stream pipeline
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    current_batch_size = 0;
    max_batch_size = 64;  // Default max batch size
    memory_allocated = false;
}

GPU2Autoencoder::~GPU2Autoencoder() {
    free_device_memory();
    free_host_memory();

    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

void GPU2Autoencoder::allocate_host_memory() {
    // Use pinned (page-locked) memory for faster H2D/D2H transfers
    CUDA_CHECK(cudaMallocHost(&h_w1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b1, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_w2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b2, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_w3, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b3, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_w4, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b4, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_w5, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b5, B5_SIZE * sizeof(float)));
}

void GPU2Autoencoder::free_host_memory() {
    // Free pinned host memory
    if (h_w1) CUDA_CHECK(cudaFreeHost(h_w1)); h_w1 = nullptr;
    if (h_b1) CUDA_CHECK(cudaFreeHost(h_b1)); h_b1 = nullptr;
    if (h_w2) CUDA_CHECK(cudaFreeHost(h_w2)); h_w2 = nullptr;
    if (h_b2) CUDA_CHECK(cudaFreeHost(h_b2)); h_b2 = nullptr;
    if (h_w3) CUDA_CHECK(cudaFreeHost(h_w3)); h_w3 = nullptr;
    if (h_b3) CUDA_CHECK(cudaFreeHost(h_b3)); h_b3 = nullptr;
    if (h_w4) CUDA_CHECK(cudaFreeHost(h_w4)); h_w4 = nullptr;
    if (h_b4) CUDA_CHECK(cudaFreeHost(h_b4)); h_b4 = nullptr;
    if (h_w5) CUDA_CHECK(cudaFreeHost(h_w5)); h_w5 = nullptr;
    if (h_b5) CUDA_CHECK(cudaFreeHost(h_b5)); h_b5 = nullptr;
}

void GPU2Autoencoder::allocate_device_memory(int batch_size) {
    if (memory_allocated && batch_size <= max_batch_size) {
        current_batch_size = batch_size;
        return;
    }

    if (memory_allocated) {
        free_device_memory();
    }

    max_batch_size = batch_size;
    current_batch_size = batch_size;

    // Allocate device weights
    CUDA_CHECK(cudaMalloc(&d_w1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5, B5_SIZE * sizeof(float)));

    // Allocate device gradients
    CUDA_CHECK(cudaMalloc(&d_dw1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw4, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db4, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw5, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db5, B5_SIZE * sizeof(float)));

    // Allocate device activations
    CUDA_CHECK(cudaMalloc(&d_input, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act4, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act5, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    // Allocate gradient buffers for backward pass
    CUDA_CHECK(cudaMalloc(&d_dL_dact5, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact4, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dconv3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dpool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dinput, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    memory_allocated = true;
}

void GPU2Autoencoder::free_device_memory() {
    if (!memory_allocated) return;

    // Free device weights
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

    // Free device gradients
    if (d_dw1) cudaFree(d_dw1);
    if (d_db1) cudaFree(d_db1);
    if (d_dw2) cudaFree(d_dw2);
    if (d_db2) cudaFree(d_db2);
    if (d_dw3) cudaFree(d_dw3);
    if (d_db3) cudaFree(d_db3);
    if (d_dw4) cudaFree(d_dw4);
    if (d_db4) cudaFree(d_db4);
    if (d_dw5) cudaFree(d_dw5);
    if (d_db5) cudaFree(d_db5);

    // Free device activations
    if (d_input) cudaFree(d_input);
    if (d_target) cudaFree(d_target);
    if (d_act1) cudaFree(d_act1);
    if (d_pool1) cudaFree(d_pool1);
    if (d_act2) cudaFree(d_act2);
    if (d_act3) cudaFree(d_act3);
    if (d_conv3_out) cudaFree(d_conv3_out);
    if (d_up1) cudaFree(d_up1);
    if (d_act4) cudaFree(d_act4);
    if (d_up2) cudaFree(d_up2);
    if (d_act5) cudaFree(d_act5);

    // Free gradient buffers
    if (d_dL_dact5) cudaFree(d_dL_dact5);
    if (d_dL_dup2) cudaFree(d_dL_dup2);
    if (d_dL_dact4) cudaFree(d_dL_dact4);
    if (d_dL_dup1) cudaFree(d_dL_dup1);
    if (d_dL_dconv3) cudaFree(d_dL_dconv3);
    if (d_dL_dact3) cudaFree(d_dL_dact3);
    if (d_dL_dact2) cudaFree(d_dL_dact2);
    if (d_dL_dpool1) cudaFree(d_dL_dpool1);
    if (d_dL_dact1) cudaFree(d_dL_dact1);
    if (d_dL_dinput) cudaFree(d_dL_dinput);

    memory_allocated = false;
}

void GPU2Autoencoder::copy_weights_to_device() {
    // Use main stream (stream 0) for weight transfers
    CUDA_CHECK(cudaMemcpyAsync(d_w1, h_w1, W1_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b1, h_b1, B1_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_w2, h_w2, W2_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b2, h_b2, B2_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_w3, h_w3, W3_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b3, h_b3, B3_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_w4, h_w4, W4_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b4, h_b4, B4_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_w5, h_w5, W5_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b5, h_b5, B5_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    
    // Wait for weight transfers to complete
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
}

void GPU2Autoencoder::copy_weights_to_host() {
    CUDA_CHECK(cudaMemcpy(h_w1, d_w1, W1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, d_b1, B1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2, d_w2, W2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, d_b2, B2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3, d_w3, W3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3, d_b3, B3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w4, d_w4, W4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b4, d_b4, B4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w5, d_w5, W5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b5, d_b5, B5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU2Autoencoder::copy_biases_to_const_memory() {
    // Copy biases to constant memory for fast broadcast access
    gpu2_copy_bias_to_const_memory(h_b1, B1_SIZE, 0);
    gpu2_copy_bias_to_const_memory(h_b2, B2_SIZE, 1);
    gpu2_copy_bias_to_const_memory(h_b3, B3_SIZE, 2);
    gpu2_copy_bias_to_const_memory(h_b4, B4_SIZE, 3);
    gpu2_copy_bias_to_const_memory(h_b5, B5_SIZE, 4);
}

void GPU2Autoencoder::initialize() {
    allocate_host_memory();

    // Initialize weights using Xavier initialization
    init_weights_xavier(h_w1, 3, 256);
    init_weights_xavier(h_w2, 256, 128);
    init_weights_xavier(h_w3, 128, 128);
    init_weights_xavier(h_w4, 128, 256);
    init_weights_xavier(h_w5, 256, 3);

    // Initialize biases to zero
    memset(h_b1, 0, B1_SIZE * sizeof(float));
    memset(h_b2, 0, B2_SIZE * sizeof(float));
    memset(h_b3, 0, B3_SIZE * sizeof(float));
    memset(h_b4, 0, B4_SIZE * sizeof(float));
    memset(h_b5, 0, B5_SIZE * sizeof(float));

    // Allocate device memory and copy weights
    allocate_device_memory(max_batch_size);
    copy_weights_to_device();
    copy_biases_to_const_memory();
}

void GPU2Autoencoder::forward_device(const float* d_in, int batch_size) {
    current_batch_size = batch_size;

    // Encoder - all operations use main stream (stream 0)
    gpu2_conv2d_forward(d_in, d_w1, d_act1, batch_size, 3, 256, 32, 32, 0);
    gpu2_relu_forward(d_act1, batch_size * 256 * 32 * 32);

    gpu2_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    gpu2_conv2d_forward(d_pool1, d_w2, d_act2, batch_size, 256, 128, 16, 16, 1);
    gpu2_relu_forward(d_act2, batch_size * 128 * 16 * 16);

    gpu2_maxpool2d_forward(d_act2, d_act3, batch_size, 128, 16, 16);

    // Decoder
    gpu2_conv2d_forward(d_act3, d_w3, d_conv3_out, batch_size, 128, 128, 8, 8, 2);
    gpu2_relu_forward(d_conv3_out, batch_size * 128 * 8 * 8);

    gpu2_upsample2d_forward(d_conv3_out, d_up1, batch_size, 128, 8, 8);

    gpu2_conv2d_forward(d_up1, d_w4, d_act4, batch_size, 128, 256, 16, 16, 3);
    gpu2_relu_forward(d_act4, batch_size * 256 * 16 * 16);

    gpu2_upsample2d_forward(d_act4, d_up2, batch_size, 256, 16, 16);

    gpu2_conv2d_forward(d_up2, d_w5, d_act5, batch_size, 256, 3, 32, 32, 4);
}

void GPU2Autoencoder::forward(const float* h_input, float* h_output, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Asynchronous H2D transfer on stream 1
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[1]));

    // Synchronize H2D transfer before computation
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    
    // Run forward pass on stream 0 (using synchronized input)
    forward_device(d_input, batch_size);

    // Asynchronous D2H transfer on stream 2
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_act5, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyDeviceToHost, streams[2]));

    // Synchronize all streams before returning
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
}

void GPU2Autoencoder::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    // 1. Compute gradient at output
    gpu2_mse_loss_gradient(d_act5, d_tgt, d_dL_dact5, output_size);

    // 2. Backward through Conv5: 256->3, 32x32
    gpu2_conv2d_backward(d_up2, d_w5, d_dL_dact5, d_dL_dup2, d_dw5, d_db5,
                        batch_size, 256, 3, 32, 32);

    // 3. Backward through Upsample2: 16x16->32x32
    gpu2_upsample2d_backward(d_dL_dup2, d_dL_dact4, batch_size, 256, 16, 16);

    // 4. Backward through ReLU4
    gpu2_relu_backward(d_act4, d_dL_dact4, d_dL_dact4, batch_size * 256 * 16 * 16);

    // 5. Backward through Conv4: 128->256, 16x16
    gpu2_conv2d_backward(d_up1, d_w4, d_dL_dact4, d_dL_dup1, d_dw4, d_db4,
                        batch_size, 128, 256, 16, 16);

    // 6. Backward through Upsample1: 8x8->16x16
    gpu2_upsample2d_backward(d_dL_dup1, d_dL_dconv3, batch_size, 128, 8, 8);

    // 7. Backward through ReLU3
    gpu2_relu_backward(d_conv3_out, d_dL_dconv3, d_dL_dconv3, batch_size * 128 * 8 * 8);

    // 8. Backward through Conv3: 128->128, 8x8
    gpu2_conv2d_backward(d_act3, d_w3, d_dL_dconv3, d_dL_dact3, d_dw3, d_db3,
                        batch_size, 128, 128, 8, 8);

    // 9. Backward through MaxPool2: 16x16->8x8
    gpu2_maxpool2d_backward(d_act2, d_act3, d_dL_dact3, d_dL_dact2,
                           batch_size, 128, 16, 16);

    // 10. Backward through ReLU2
    gpu2_relu_backward(d_act2, d_dL_dact2, d_dL_dact2, batch_size * 128 * 16 * 16);

    // 11. Backward through Conv2: 256->128, 16x16
    gpu2_conv2d_backward(d_pool1, d_w2, d_dL_dact2, d_dL_dpool1, d_dw2, d_db2,
                        batch_size, 256, 128, 16, 16);

    // 12. Backward through MaxPool1: 32x32->16x16
    gpu2_maxpool2d_backward(d_act1, d_pool1, d_dL_dpool1, d_dL_dact1,
                           batch_size, 256, 32, 32);

    // 13. Backward through ReLU1
    gpu2_relu_backward(d_act1, d_dL_dact1, d_dL_dact1, batch_size * 256 * 32 * 32);

    // 14. Backward through Conv1: 3->256, 32x32
    gpu2_conv2d_backward(d_in, d_w1, d_dL_dact1, d_dL_dinput, d_dw1, d_db1,
                        batch_size, 3, 256, 32, 32);
}

void GPU2Autoencoder::backward(const float* h_input, const float* h_target, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Asynchronous transfers on streams 1 and 2
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaMemcpyAsync(d_target, h_target, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[2]));

    // Wait for transfers to complete
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));

    // Run backward pass
    backward_device(d_input, d_target, batch_size);
}

void GPU2Autoencoder::update_weights(float learning_rate) {
    const float clip_value = 1.0f;

    // Update all weights and biases using SGD with gradient clipping
    gpu2_sgd_update(d_w1, d_dw1, learning_rate, clip_value, W1_SIZE);
    gpu2_sgd_update(d_b1, d_db1, learning_rate, clip_value, B1_SIZE);
    gpu2_sgd_update(d_w2, d_dw2, learning_rate, clip_value, W2_SIZE);
    gpu2_sgd_update(d_b2, d_db2, learning_rate, clip_value, B2_SIZE);
    gpu2_sgd_update(d_w3, d_dw3, learning_rate, clip_value, W3_SIZE);
    gpu2_sgd_update(d_b3, d_db3, learning_rate, clip_value, B3_SIZE);
    gpu2_sgd_update(d_w4, d_dw4, learning_rate, clip_value, W4_SIZE);
    gpu2_sgd_update(d_b4, d_db4, learning_rate, clip_value, B4_SIZE);
    gpu2_sgd_update(d_w5, d_dw5, learning_rate, clip_value, W5_SIZE);
    gpu2_sgd_update(d_b5, d_db5, learning_rate, clip_value, B5_SIZE);
}

float GPU2Autoencoder::compute_loss(const float* h_target, int batch_size) {
    // Copy target to device asynchronously
    CUDA_CHECK(cudaMemcpyAsync(d_target, h_target, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    
    int size = batch_size * 3 * 32 * 32;
    return gpu2_mse_loss(d_act5, d_target, size);
}

void GPU2Autoencoder::extract_features(const float* h_input, float* h_features, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Asynchronous H2D transfer
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // Run encoder only
    gpu2_conv2d_forward(d_input, d_w1, d_act1, batch_size, 3, 256, 32, 32, 0);
    gpu2_relu_forward(d_act1, batch_size * 256 * 32 * 32);

    gpu2_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    gpu2_conv2d_forward(d_pool1, d_w2, d_act2, batch_size, 256, 128, 16, 16, 1);
    gpu2_relu_forward(d_act2, batch_size * 128 * 16 * 16);

    gpu2_maxpool2d_forward(d_act2, d_act3, batch_size, 128, 16, 16);

    // Asynchronous D2H transfer
    CUDA_CHECK(cudaMemcpyAsync(h_features, d_act3, batch_size * 128 * 8 * 8 * sizeof(float), 
                               cudaMemcpyDeviceToHost, streams[2]));
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
}

void GPU2Autoencoder::save_weights(const std::string& filepath) {
    // Copy weights from device to host
    copy_weights_to_host();

    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return;
    }

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
    printf("GPU2 Model weights saved to: %s\n", filepath.c_str());
}

void GPU2Autoencoder::load_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filepath.c_str());
        return;
    }

    if (!h_w1) {
        allocate_host_memory();
    }

    fread(h_w1, sizeof(float), W1_SIZE, f);
    fread(h_b1, sizeof(float), B1_SIZE, f);
    fread(h_w2, sizeof(float), W2_SIZE, f);
    fread(h_b2, sizeof(float), B2_SIZE, f);
    fread(h_w3, sizeof(float), W3_SIZE, f);
    fread(h_b3, sizeof(float), B3_SIZE, f);
    fread(h_w4, sizeof(float), W4_SIZE, f);
    fread(h_b4, sizeof(float), B4_SIZE, f);
    fread(h_w5, sizeof(float), W5_SIZE, f);
    fread(h_b5, sizeof(float), B5_SIZE, f);

    fclose(f);

    // Allocate device memory if needed and copy weights
    if (!memory_allocated) {
        allocate_device_memory(max_batch_size);
    }
    copy_weights_to_device();
    copy_biases_to_const_memory();

    printf("GPU2 Model weights loaded from: %s\n", filepath.c_str());
}
