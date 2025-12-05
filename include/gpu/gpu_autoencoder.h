#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_H

#include <string>

// ============================================================================
// GPU-based 5-layer Convolutional Autoencoder
// ============================================================================
// Architecture:
//   Input:  3 x 32 x 32
//   Conv1:  3 -> 256, 32x32 + ReLU
//   Pool1:  32x32 -> 16x16
//   Conv2:  256 -> 128, 16x16 + ReLU  
//   Pool2:  16x16 -> 8x8  (Latent: 128 x 8 x 8 = 8192 features)
//   Conv3:  128 -> 128, 8x8 + ReLU
//   Up1:    8x8 -> 16x16
//   Conv4:  128 -> 256, 16x16 + ReLU
//   Up2:    16x16 -> 32x32
//   Conv5:  256 -> 3, 32x32
//   Output: 3 x 32 x 32
//
// Total trainable parameters: 751,875
// ============================================================================

class GPUAutoencoder {
public:
    GPUAutoencoder();
    ~GPUAutoencoder();

    // Initialize weights with Xavier initialization
    void initialize();

    // Forward pass: input (host) -> output (host)
    void forward(const float* h_input, float* h_output, int batch_size);

    // Backward pass: compute gradients for all weights
    void backward(const float* h_input, const float* h_target, int batch_size);

    // Update weights using SGD with gradient clipping
    void update_weights(float learning_rate);

    // Compute MSE loss between output and target
    float compute_loss(const float* h_target, int batch_size);

    // Extract latent features (encoder only): returns 8192-dim features
    void extract_features(const float* h_input, float* h_features, int batch_size);

    // Save/Load model weights to/from file
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    // =========================================================================
    // Host-side weights (for initialization and save/load)
    // =========================================================================
    float *h_w1, *h_b1;  // Conv1: 3->256
    float *h_w2, *h_b2;  // Conv2: 256->128
    float *h_w3, *h_b3;  // Conv3: 128->128
    float *h_w4, *h_b4;  // Conv4: 128->256
    float *h_w5, *h_b5;  // Conv5: 256->3

    // =========================================================================
    // Device-side weights and biases
    // =========================================================================
    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    float *d_w3, *d_b3;
    float *d_w4, *d_b4;
    float *d_w5, *d_b5;

    // =========================================================================
    // Device-side gradients
    // =========================================================================
    float *d_dw1, *d_db1;
    float *d_dw2, *d_db2;
    float *d_dw3, *d_db3;
    float *d_dw4, *d_db4;
    float *d_dw5, *d_db5;

    // =========================================================================
    // Device-side activation buffers
    // =========================================================================
    float *d_input;      // Input: batch x 3 x 32 x 32
    float *d_target;     // Target: batch x 3 x 32 x 32
    float *d_act1;       // After Conv1+ReLU: batch x 256 x 32 x 32
    float *d_pool1;      // After MaxPool1: batch x 256 x 16 x 16
    float *d_act2;       // After Conv2+ReLU: batch x 128 x 16 x 16
    float *d_act3;       // After MaxPool2 (latent): batch x 128 x 8 x 8
    float *d_conv3_out;  // After Conv3+ReLU: batch x 128 x 8 x 8
    float *d_up1;        // After Upsample1: batch x 128 x 16 x 16
    float *d_act4;       // After Conv4+ReLU: batch x 256 x 16 x 16
    float *d_up2;        // After Upsample2: batch x 256 x 32 x 32
    float *d_act5;       // After Conv5 (output): batch x 3 x 32 x 32

    // =========================================================================
    // Device-side gradient buffers for backward pass
    // =========================================================================
    float *d_dL_dact5, *d_dL_dup2, *d_dL_dact4, *d_dL_dup1;
    float *d_dL_dconv3, *d_dL_dact3, *d_dL_dact2, *d_dL_dpool1;
    float *d_dL_dact1, *d_dL_dinput;

    // State
    int current_batch_size;
    int max_batch_size;
    bool memory_allocated;

    // =========================================================================
    // Internal methods
    // =========================================================================
    void allocate_host_memory();
    void free_host_memory();
    void allocate_device_memory(int batch_size);
    void free_device_memory();
    void copy_weights_to_device();
    void copy_weights_to_host();

    // Device-side operations (internal use)
    void forward_device(const float* d_in, int batch_size);
    void backward_device(const float* d_in, const float* d_tgt, int batch_size);
    void extract_features_device(const float* d_in, float* d_features, int batch_size);
};

#endif  // GPU_AUTOENCODER_H
