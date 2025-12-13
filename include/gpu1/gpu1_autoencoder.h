#ifndef GPU1_AUTOENCODER_H
#define GPU1_AUTOENCODER_H

#include <string>

// GPU Autoencoder v1
// Optimizations (distinct from GPU v2):
//  - Kernel fusion: Conv2D + ReLU in a single kernel for conv1-4 (loop-unrolled for efficiency)
//  - Loop Unrolling: 3x3 kernel manually unrolled to reduce loop overhead & enable compiler optimization
//  - Pinned Host Memory: Input/Target buffers use cudaMallocHost for faster H2D/D2H transfers
//  - Reduced global memory traffic (no separate ReLU kernel reads/writes)
//  - Note: no const memory, no multi-stream (those are in GPU v2)
//
// Architecture is identical to CPU/GPU/GPU2 versions.

class GPU1Autoencoder {
public:
    GPU1Autoencoder();
    ~GPU1Autoencoder();

    void initialize();

    // Forward: host input -> host output
    void forward(const float* h_input, float* h_output, int batch_size);

    // Backward: compute gradients given host input/target
    void backward(const float* h_input, const float* h_target, int batch_size);

    // SGD update with gradient clipping
    void update_weights(float learning_rate);

    // Compute MSE loss between current output and target
    float compute_loss(const float* h_target, int batch_size);

    // Extract latent features (encoder only), 8192-dim per image
    void extract_features(const float* h_input, float* h_features, int batch_size);

    // Save/Load weights
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    // Host-side weights/biases
    float *h_w1, *h_b1;
    float *h_w2, *h_b2;
    float *h_w3, *h_b3;
    float *h_w4, *h_b4;
    float *h_w5, *h_b5;

    // Pinned host memory for input/target (faster H2D/D2H transfers)
    float *h_input_pinned;
    float *h_target_pinned;

    // Device-side weights/biases
    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    float *d_w3, *d_b3;
    float *d_w4, *d_b4;
    float *d_w5, *d_b5;

    // Device-side gradients
    float *d_dw1, *d_db1;
    float *d_dw2, *d_db2;
    float *d_dw3, *d_db3;
    float *d_dw4, *d_db4;
    float *d_dw5, *d_db5;

    // Activations
    float *d_input;
    float *d_target;
    float *d_act1;
    float *d_pool1;
    float *d_act2;
    float *d_act3;
    float *d_conv3_out;
    float *d_up1;
    float *d_act4;
    float *d_up2;
    float *d_act5;

    // Backward buffers
    float *d_dL_dact5;
    float *d_dL_dup2;
    float *d_dL_dact4;
    float *d_dL_dup1;
    float *d_dL_dconv3;
    float *d_dL_dact3;
    float *d_dL_dact2;
    float *d_dL_dpool1;
    float *d_dL_dact1;
    float *d_dL_dinput;

    int current_batch_size;
    int max_batch_size;
    bool memory_allocated;

    void allocate_host_memory();
    void free_host_memory();
    void allocate_device_memory(int batch_size);
    void free_device_memory();
    void copy_weights_to_device();
    void copy_weights_to_host();

    void forward_device(const float* d_in, int batch_size);
    void backward_device(const float* d_in, const float* d_tgt, int batch_size);
};

#endif // GPU1_AUTOENCODER_H
