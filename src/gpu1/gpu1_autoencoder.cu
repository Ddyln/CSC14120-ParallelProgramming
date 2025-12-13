#include "gpu1/gpu1_autoencoder.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <cstring>
#include <random>

#include "gpu/gpu_layers.cuh"   // baseline GPU kernels
#include "gpu1/gpu1_layers.cuh" // fused Conv2D+ReLU for v1

// Weight sizes
constexpr int W1_SIZE_1 = 256 * 3 * 3 * 3;
constexpr int B1_SIZE_1 = 256;
constexpr int W2_SIZE_1 = 128 * 256 * 3 * 3;
constexpr int B2_SIZE_1 = 128;
constexpr int W3_SIZE_1 = 128 * 128 * 3 * 3;
constexpr int B3_SIZE_1 = 128;
constexpr int W4_SIZE_1 = 256 * 128 * 3 * 3;
constexpr int B4_SIZE_1 = 256;
constexpr int W5_SIZE_1 = 3 * 256 * 3 * 3;
constexpr int B5_SIZE_1 = 3;

static void init_weights_xavier_v1(float* weights, int in_channels, int out_channels) {
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

GPU1Autoencoder::GPU1Autoencoder() {
    h_w1 = h_b1 = h_w2 = h_b2 = h_w3 = h_b3 = h_w4 = h_b4 = h_w5 = h_b5 = nullptr;
    h_input_pinned = h_target_pinned = nullptr;

    d_w1 = d_b1 = d_w2 = d_b2 = d_w3 = d_b3 = d_w4 = d_b4 = d_w5 = d_b5 = nullptr;
    d_dw1 = d_db1 = d_dw2 = d_db2 = d_dw3 = d_db3 = d_dw4 = d_db4 = d_dw5 = d_db5 = nullptr;

    d_input = d_target = nullptr;
    d_act1 = d_pool1 = d_act2 = d_act3 = nullptr;
    d_conv3_out = d_up1 = d_act4 = d_up2 = d_act5 = nullptr;

    d_dL_dact5 = d_dL_dup2 = d_dL_dact4 = d_dL_dup1 = nullptr;
    d_dL_dconv3 = d_dL_dact3 = d_dL_dact2 = d_dL_dpool1 = nullptr;
    d_dL_dact1 = d_dL_dinput = nullptr;

    current_batch_size = 0;
    max_batch_size = 64;
    memory_allocated = false;
}

GPU1Autoencoder::~GPU1Autoencoder() {
    free_device_memory();
    free_host_memory();
}

void GPU1Autoencoder::allocate_host_memory() {
    h_w1 = new float[W1_SIZE_1];
    h_b1 = new float[B1_SIZE_1];
    h_w2 = new float[W2_SIZE_1];
    h_b2 = new float[B2_SIZE_1];
    h_w3 = new float[W3_SIZE_1];
    h_b3 = new float[B3_SIZE_1];
    h_w4 = new float[W4_SIZE_1];
    h_b4 = new float[B4_SIZE_1];
    h_w5 = new float[W5_SIZE_1];
    h_b5 = new float[B5_SIZE_1];

    // Allocate pinned memory for input/target buffers (faster H2D/D2H)
    // Max batch size = 64, 3 channels, 32x32 = 196608 floats per buffer
    CUDA_CHECK(cudaMallocHost(&h_input_pinned, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_target_pinned, max_batch_size * 3 * 32 * 32 * sizeof(float)));
}

void GPU1Autoencoder::free_host_memory() {
    delete[] h_w1; h_w1 = nullptr;
    delete[] h_b1; h_b1 = nullptr;
    delete[] h_w2; h_w2 = nullptr;
    delete[] h_b2; h_b2 = nullptr;
    delete[] h_w3; h_w3 = nullptr;
    delete[] h_b3; h_b3 = nullptr;
    delete[] h_w4; h_w4 = nullptr;
    delete[] h_b4; h_b4 = nullptr;
    delete[] h_w5; h_w5 = nullptr;
    delete[] h_b5; h_b5 = nullptr;

    // Free pinned memory
    if (h_input_pinned) {
        CUDA_CHECK(cudaFreeHost(h_input_pinned));
        h_input_pinned = nullptr;
    }
    if (h_target_pinned) {
        CUDA_CHECK(cudaFreeHost(h_target_pinned));
        h_target_pinned = nullptr;
    }
}

void GPU1Autoencoder::allocate_device_memory(int batch_size) {
    if (memory_allocated && batch_size <= max_batch_size) {
        current_batch_size = batch_size;
        return;
    }

    if (memory_allocated) {
        free_device_memory();
    }

    max_batch_size = batch_size;
    current_batch_size = batch_size;

    // Weights & biases
    CUDA_CHECK(cudaMalloc(&d_w1, W1_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, B1_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, W2_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, B2_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, W3_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, B3_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4, W4_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4, B4_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5, W5_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5, B5_SIZE_1 * sizeof(float)));

    // Gradients
    CUDA_CHECK(cudaMalloc(&d_dw1, W1_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, B1_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, W2_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, B2_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3, W3_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, B3_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw4, W4_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db4, B4_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw5, W5_SIZE_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db5, B5_SIZE_1 * sizeof(float)));

    // Activations
    CUDA_CHECK(cudaMalloc(&d_input,  max_batch_size * 3   * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, max_batch_size * 3   * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act1,   max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1,  max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act2,   max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act3,   max_batch_size * 128 *  8 *  8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1,    max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act4,   max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2,    max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act5,   max_batch_size * 3   * 32 * 32 * sizeof(float)));

    // Backward buffers
    CUDA_CHECK(cudaMalloc(&d_dL_dact5, max_batch_size * 3   * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup2,  max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact4, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup1,  max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dconv3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dpool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dinput, max_batch_size * 3  * 32 * 32 * sizeof(float)));

    memory_allocated = true;
}

void GPU1Autoencoder::free_device_memory() {
    if (!memory_allocated) return;

    auto f = [](float*& p) { if (p) cudaFree(p); p = nullptr; };

    f(d_w1); f(d_b1); f(d_w2); f(d_b2); f(d_w3); f(d_b3); f(d_w4); f(d_b4); f(d_w5); f(d_b5);
    f(d_dw1); f(d_db1); f(d_dw2); f(d_db2); f(d_dw3); f(d_db3); f(d_dw4); f(d_db4); f(d_dw5); f(d_db5);

    f(d_input); f(d_target); f(d_act1); f(d_pool1); f(d_act2); f(d_act3);
    f(d_conv3_out); f(d_up1); f(d_act4); f(d_up2); f(d_act5);

    f(d_dL_dact5); f(d_dL_dup2); f(d_dL_dact4); f(d_dL_dup1);
    f(d_dL_dconv3); f(d_dL_dact3); f(d_dL_dact2); f(d_dL_dpool1);
    f(d_dL_dact1); f(d_dL_dinput);

    memory_allocated = false;
}

void GPU1Autoencoder::copy_weights_to_device() {
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, W1_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, B1_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, W2_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, B2_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3, W3_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3, B3_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4, h_w4, W4_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b4, h_b4, B4_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5, h_w5, W5_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b5, h_b5, B5_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
}

void GPU1Autoencoder::copy_weights_to_host() {
    CUDA_CHECK(cudaMemcpy(h_w1, d_w1, W1_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, d_b1, B1_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2, d_w2, W2_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, d_b2, B2_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3, d_w3, W3_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3, d_b3, B3_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w4, d_w4, W4_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b4, d_b4, B4_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w5, d_w5, W5_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b5, d_b5, B5_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU1Autoencoder::initialize() {
    allocate_host_memory();

    init_weights_xavier_v1(h_w1, 3, 256);
    init_weights_xavier_v1(h_w2, 256, 128);
    init_weights_xavier_v1(h_w3, 128, 128);
    init_weights_xavier_v1(h_w4, 128, 256);
    init_weights_xavier_v1(h_w5, 256, 3);

    std::memset(h_b1, 0, B1_SIZE_1 * sizeof(float));
    std::memset(h_b2, 0, B2_SIZE_1 * sizeof(float));
    std::memset(h_b3, 0, B3_SIZE_1 * sizeof(float));
    std::memset(h_b4, 0, B4_SIZE_1 * sizeof(float));
    std::memset(h_b5, 0, B5_SIZE_1 * sizeof(float));

    allocate_device_memory(max_batch_size);
    copy_weights_to_device();
}

void GPU1Autoencoder::forward_device(const float* d_in, int batch_size) {
    current_batch_size = batch_size;

    // Encoder
    // Conv1 + ReLU (fused)
    gpu1_conv2d_relu_forward(d_in, d_w1, d_b1, d_act1, batch_size, 3, 256, 32, 32);

    gpu_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    // Conv2 + ReLU (fused)
    gpu1_conv2d_relu_forward(d_pool1, d_w2, d_b2, d_act2, batch_size, 256, 128, 16, 16);

    gpu_maxpool2d_forward(d_act2, d_act3, batch_size, 128, 16, 16);

    // Decoder
    // Decoder
    // Conv3 + ReLU (fused)
    gpu1_conv2d_relu_forward(d_act3, d_w3, d_b3, d_conv3_out, batch_size, 128, 128, 8, 8);

    gpu_upsample2d_forward(d_conv3_out, d_up1, batch_size, 128, 8, 8);

    // Conv4 + ReLU (fused)
    gpu1_conv2d_relu_forward(d_up1, d_w4, d_b4, d_act4, batch_size, 128, 256, 16, 16);

    gpu_upsample2d_forward(d_act4, d_up2, batch_size, 256, 16, 16);

    // Conv5 (no activation)
    gpu_conv2d_forward(d_up2, d_w5, d_b5, d_act5, batch_size, 256, 3, 32, 32);
}

void GPU1Autoencoder::forward(const float* h_input, float* h_output, int batch_size) {
    allocate_device_memory(batch_size);

    // Use pinned memory for faster H2D transfer (optimization: Pinned Host Memory)
    int input_size = batch_size * 3 * 32 * 32 * sizeof(float);
    std::memcpy(h_input_pinned, h_input, input_size);
    CUDA_CHECK(cudaMemcpy(d_input, h_input_pinned, input_size, cudaMemcpyHostToDevice));

    forward_device(d_input, batch_size);

    CUDA_CHECK(cudaMemcpy(h_output, d_act5, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU1Autoencoder::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    gpu_mse_loss_gradient(d_act5, d_tgt, d_dL_dact5, output_size);

    gpu_conv2d_backward(d_up2, d_w5, d_dL_dact5, d_dL_dup2, d_dw5, d_db5,
                         batch_size, 256, 3, 32, 32);

    gpu_upsample2d_backward(d_dL_dup2, d_dL_dact4, batch_size, 256, 16, 16);

    gpu_relu_backward(d_act4, d_dL_dact4, d_dL_dact4, batch_size * 256 * 16 * 16);

    gpu_conv2d_backward(d_up1, d_w4, d_dL_dact4, d_dL_dup1, d_dw4, d_db4,
                         batch_size, 128, 256, 16, 16);

    gpu_upsample2d_backward(d_dL_dup1, d_dL_dconv3, batch_size, 128, 8, 8);

    gpu_relu_backward(d_conv3_out, d_dL_dconv3, d_dL_dconv3, batch_size * 128 * 8 * 8);

    gpu_conv2d_backward(d_act3, d_w3, d_dL_dconv3, d_dL_dact3, d_dw3, d_db3,
                         batch_size, 128, 128, 8, 8);

    gpu_maxpool2d_backward(d_act2, d_act3, d_dL_dact3, d_dL_dact2,
                            batch_size, 128, 16, 16);

    gpu_relu_backward(d_act2, d_dL_dact2, d_dL_dact2, batch_size * 128 * 16 * 16);

    gpu_conv2d_backward(d_pool1, d_w2, d_dL_dact2, d_dL_dpool1, d_dw2, d_db2,
                         batch_size, 256, 128, 16, 16);

    gpu_maxpool2d_backward(d_act1, d_pool1, d_dL_dpool1, d_dL_dact1,
                            batch_size, 256, 32, 32);

    gpu_relu_backward(d_act1, d_dL_dact1, d_dL_dact1, batch_size * 256 * 32 * 32);

    gpu_conv2d_backward(d_in, d_w1, d_dL_dact1, d_dL_dinput, d_dw1, d_db1,
                         batch_size, 3, 256, 32, 32);
}

void GPU1Autoencoder::backward(const float* h_input, const float* h_target, int batch_size) {
    allocate_device_memory(batch_size);

    // Use pinned memory for faster H2D transfer (optimization: Pinned Host Memory)
    int input_size = batch_size * 3 * 32 * 32 * sizeof(float);
    std::memcpy(h_input_pinned, h_input, input_size);
    std::memcpy(h_target_pinned, h_target, input_size);

    CUDA_CHECK(cudaMemcpy(d_input, h_input_pinned, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target_pinned, input_size, cudaMemcpyHostToDevice));

    backward_device(d_input, d_target, batch_size);
}

void GPU1Autoencoder::update_weights(float learning_rate) {
    const float clip = 1.0f;

    gpu_sgd_update(d_w1, d_dw1, learning_rate, clip, W1_SIZE_1);
    gpu_sgd_update(d_b1, d_db1, learning_rate, clip, B1_SIZE_1);
    gpu_sgd_update(d_w2, d_dw2, learning_rate, clip, W2_SIZE_1);
    gpu_sgd_update(d_b2, d_db2, learning_rate, clip, B2_SIZE_1);
    gpu_sgd_update(d_w3, d_dw3, learning_rate, clip, W3_SIZE_1);
    gpu_sgd_update(d_b3, d_db3, learning_rate, clip, B3_SIZE_1);
    gpu_sgd_update(d_w4, d_dw4, learning_rate, clip, W4_SIZE_1);
    gpu_sgd_update(d_b4, d_db4, learning_rate, clip, B4_SIZE_1);
    gpu_sgd_update(d_w5, d_dw5, learning_rate, clip, W5_SIZE_1);
    gpu_sgd_update(d_b5, d_db5, learning_rate, clip, B5_SIZE_1);
}

float GPU1Autoencoder::compute_loss(const float* h_target, int batch_size) {
    allocate_device_memory(batch_size);
    CUDA_CHECK(cudaMemcpy(d_target, h_target, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    int size = batch_size * 3 * 32 * 32;
    return gpu_mse_loss(d_act5, d_target, size);
}

void GPU1Autoencoder::extract_features(const float* h_input, float* h_features, int batch_size) {
    allocate_device_memory(batch_size);
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Encoder only
    gpu1_conv2d_relu_forward(d_input, d_w1, d_b1, d_act1, batch_size, 3, 256, 32, 32);

    gpu_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    gpu1_conv2d_relu_forward(d_pool1, d_w2, d_b2, d_act2, batch_size, 256, 128, 16, 16);

    gpu_maxpool2d_forward(d_act2, d_act3, batch_size, 128, 16, 16);

    CUDA_CHECK(cudaMemcpy(h_features, d_act3, batch_size * 128 * 8 * 8 * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU1Autoencoder::save_weights(const std::string& filepath) {
    copy_weights_to_host();

    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return;
    }

    fwrite(h_w1, sizeof(float), W1_SIZE_1, f);
    fwrite(h_b1, sizeof(float), B1_SIZE_1, f);
    fwrite(h_w2, sizeof(float), W2_SIZE_1, f);
    fwrite(h_b2, sizeof(float), B2_SIZE_1, f);
    fwrite(h_w3, sizeof(float), W3_SIZE_1, f);
    fwrite(h_b3, sizeof(float), B3_SIZE_1, f);
    fwrite(h_w4, sizeof(float), W4_SIZE_1, f);
    fwrite(h_b4, sizeof(float), B4_SIZE_1, f);
    fwrite(h_w5, sizeof(float), W5_SIZE_1, f);
    fwrite(h_b5, sizeof(float), B5_SIZE_1, f);

    fclose(f);
    printf("GPU1 model weights saved to: %s\n", filepath.c_str());
}

void GPU1Autoencoder::load_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filepath.c_str());
        return;
    }

    if (!h_w1) {
        allocate_host_memory();
    }

    size_t got = 0;
    got = fread(h_w1, sizeof(float), W1_SIZE_1, f);
    if (got != (size_t)W1_SIZE_1) { fprintf(stderr, "Failed reading W1\n"); fclose(f); return; }
    got = fread(h_b1, sizeof(float), B1_SIZE_1, f);
    if (got != (size_t)B1_SIZE_1) { fprintf(stderr, "Failed reading B1\n"); fclose(f); return; }
    got = fread(h_w2, sizeof(float), W2_SIZE_1, f);
    if (got != (size_t)W2_SIZE_1) { fprintf(stderr, "Failed reading W2\n"); fclose(f); return; }
    got = fread(h_b2, sizeof(float), B2_SIZE_1, f);
    if (got != (size_t)B2_SIZE_1) { fprintf(stderr, "Failed reading B2\n"); fclose(f); return; }
    got = fread(h_w3, sizeof(float), W3_SIZE_1, f);
    if (got != (size_t)W3_SIZE_1) { fprintf(stderr, "Failed reading W3\n"); fclose(f); return; }
    got = fread(h_b3, sizeof(float), B3_SIZE_1, f);
    if (got != (size_t)B3_SIZE_1) { fprintf(stderr, "Failed reading B3\n"); fclose(f); return; }
    got = fread(h_w4, sizeof(float), W4_SIZE_1, f);
    if (got != (size_t)W4_SIZE_1) { fprintf(stderr, "Failed reading W4\n"); fclose(f); return; }
    got = fread(h_b4, sizeof(float), B4_SIZE_1, f);
    if (got != (size_t)B4_SIZE_1) { fprintf(stderr, "Failed reading B4\n"); fclose(f); return; }
    got = fread(h_w5, sizeof(float), W5_SIZE_1, f);
    if (got != (size_t)W5_SIZE_1) { fprintf(stderr, "Failed reading W5\n"); fclose(f); return; }
    got = fread(h_b5, sizeof(float), B5_SIZE_1, f);
    if (got != (size_t)B5_SIZE_1) { fprintf(stderr, "Failed reading B5\n"); fclose(f); return; }

    fclose(f);

    if (!memory_allocated) {
        allocate_device_memory(max_batch_size);
    }
    copy_weights_to_device();

    printf("GPU1 model weights loaded from: %s\n", filepath.c_str());
}
