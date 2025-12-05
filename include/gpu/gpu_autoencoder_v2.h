#ifndef GPU_AUTOENCODER_V2_H
#define GPU_AUTOENCODER_V2_H

#include <string>

// ============================================================================
// GPU Autoencoder V2 - Optimized with Kernel Fusion, Loop Unrolling, float4
// ============================================================================
// Optimizations applied:
//   1. Kernel Fusion (Conv + Bias + ReLU) - Single kernel per layer
//   2. Loop Unrolling (3x3 kernel) - Reduces loop overhead  
//   3. Vectorized Memory (float4) - Better memory bandwidth
// ============================================================================
//
// Architecture (same as baseline):
//   Input:  3 x 32 x 32
//   Conv1:  3 -> 256, 32x32 + ReLU
//   Pool1:  32x32 -> 16x16
//   Conv2:  256 -> 128, 16x16 + ReLU  
//   Pool2:  16x16 -> 8x8  (Latent: 128 x 8 x 8 = 8192 features)
//   Conv3:  128 -> 128, 8x8 + ReLU
//   Up1:    8x8 -> 16x16
//   Conv4:  128 -> 256, 16x16 + ReLU
//   Up2:    16x16 -> 32x32
//   Conv5:  256 -> 3, 32x32 (no activation)
//   Output: 3 x 32 x 32
//
// API: Same as GPU baseline - uses HOST pointers for input/output
// ============================================================================

class GPUAutoencoderV2 {
   public:
    GPUAutoencoderV2();
    ~GPUAutoencoderV2();

    // Forward pass: device input -> device output (matches baseline API)
    void forward(const float* d_input, float* d_output, int batch_size);

    // Backward pass: computes gradients (device pointers)
    void backward(const float* d_input, const float* d_target, int batch_size);

    // Update weights using SGD with vectorized operations
    void update_weights(float learning_rate);

    // Get latent features: device input -> device features (8192-dim)
    void get_features(const float* d_input, float* d_features, int batch_size);

    // Compute MSE loss (device pointers) - returns scalar on host
    float compute_loss(const float* d_output, const float* d_target, int batch_size);
    
    // Compute MSE loss (device pointers)
    float compute_loss_gpu(const float* d_output, const float* d_target, int batch_size);

    // Save/Load weights
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);

    // Get total parameters
    int get_total_params() const { return TOTAL_PARAMS; }

   private:
    // Layer dimensions
    static const int INPUT_C = 3, INPUT_H = 32, INPUT_W = 32;
    static const int CONV1_OUT = 256, CONV1_H = 32, CONV1_W = 32;
    static const int POOL1_H = 16, POOL1_W = 16;
    static const int CONV2_OUT = 128, CONV2_H = 16, CONV2_W = 16;
    static const int POOL2_H = 8, POOL2_W = 8;
    static const int CONV3_OUT = 128, CONV3_H = 8, CONV3_W = 8;
    static const int UP1_H = 16, UP1_W = 16;
    static const int CONV4_OUT = 256, CONV4_H = 16, CONV4_W = 16;
    static const int UP2_H = 32, UP2_W = 32;
    static const int CONV5_OUT = 3, CONV5_H = 32, CONV5_W = 32;

    // Latent space dimensions
    static const int LATENT_C = 128, LATENT_H = 8, LATENT_W = 8;
    static const int LATENT_SIZE = LATENT_C * LATENT_H * LATENT_W;  // 8192

    // Weight sizes
    static const int W1_SIZE = CONV1_OUT * INPUT_C * 3 * 3;    // 256*3*9 = 6912
    static const int B1_SIZE = CONV1_OUT;                       // 256
    static const int W2_SIZE = CONV2_OUT * CONV1_OUT * 3 * 3;  // 128*256*9 = 294912
    static const int B2_SIZE = CONV2_OUT;                       // 128
    static const int W3_SIZE = CONV3_OUT * CONV2_OUT * 3 * 3;  // 128*128*9 = 147456
    static const int B3_SIZE = CONV3_OUT;                       // 128
    static const int W4_SIZE = CONV4_OUT * CONV3_OUT * 3 * 3;  // 256*128*9 = 294912
    static const int B4_SIZE = CONV4_OUT;                       // 256
    static const int W5_SIZE = CONV5_OUT * CONV4_OUT * 3 * 3;  // 3*256*9 = 6912
    static const int B5_SIZE = CONV5_OUT;                       // 3

    static const int TOTAL_PARAMS = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + 
                                    W3_SIZE + B3_SIZE + W4_SIZE + B4_SIZE + 
                                    W5_SIZE + B5_SIZE;  // 751,875

    // Device pointers for weights
    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    float *d_w3, *d_b3;
    float *d_w4, *d_b4;
    float *d_w5, *d_b5;

    // Device pointers for weight gradients
    float *d_grad_w1, *d_grad_b1;
    float *d_grad_w2, *d_grad_b2;
    float *d_grad_w3, *d_grad_b3;
    float *d_grad_w4, *d_grad_b4;
    float *d_grad_w5, *d_grad_b5;

    // Device pointers for activations (forward pass)
    float* d_conv1_out;   // After fused Conv1+Bias+ReLU
    float* d_pool1_out;
    float* d_conv2_out;   // After fused Conv2+Bias+ReLU
    float* d_pool2_out;   // Latent representation
    float* d_conv3_out;   // After fused Conv3+Bias+ReLU
    float* d_up1_out;
    float* d_conv4_out;   // After fused Conv4+Bias+ReLU
    float* d_up2_out;
    float* d_conv5_out;   // Final output (no ReLU)

    // Device pointers for gradients (backward pass)
    float* d_grad_conv5;
    float* d_grad_up2;
    float* d_grad_conv4;
    float* d_grad_up1;
    float* d_grad_conv3;
    float* d_grad_pool2;
    float* d_grad_conv2;
    float* d_grad_pool1;
    float* d_grad_conv1;

    // Maximum batch size for pre-allocated buffers
    int max_batch_size;
    int current_batch_size;

    // Memory management
    void allocate_weights();
    void allocate_activations(int batch_size);
    void allocate_gradients(int batch_size);
    void init_weights();
    void free_memory();
};

#endif  // GPU_AUTOENCODER_V2_H
