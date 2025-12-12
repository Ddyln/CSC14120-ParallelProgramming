#ifndef GPU_CONSTANTS_H
#define GPU_CONSTANTS_H

// ============================================================================
// GPU Autoencoder Shared Constants
// ============================================================================
// This file contains all the architecture constants shared between
// GPU baseline and optimized versions.
// ============================================================================

namespace gpu_constants {

// ============================================================================
// Network Architecture Dimensions
// ============================================================================

// Input
constexpr int INPUT_C = 3;
constexpr int INPUT_H = 32;
constexpr int INPUT_W = 32;
constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W;  // 3072

// Encoder Layer 1: Conv1 + ReLU + MaxPool
constexpr int CONV1_OUT = 256;
constexpr int CONV1_H = 32;
constexpr int CONV1_W = 32;
constexpr int POOL1_H = 16;
constexpr int POOL1_W = 16;

// Encoder Layer 2: Conv2 + ReLU + MaxPool
constexpr int CONV2_OUT = 128;
constexpr int CONV2_H = 16;
constexpr int CONV2_W = 16;
constexpr int POOL2_H = 8;
constexpr int POOL2_W = 8;

// Latent Space (Bottleneck)
constexpr int LATENT_C = 128;
constexpr int LATENT_H = 8;
constexpr int LATENT_W = 8;
constexpr int LATENT_SIZE = LATENT_C * LATENT_H * LATENT_W;  // 8192

// Decoder Layer 3: Conv3 + ReLU
constexpr int CONV3_OUT = 128;
constexpr int CONV3_H = 8;
constexpr int CONV3_W = 8;
constexpr int UP1_H = 16;
constexpr int UP1_W = 16;

// Decoder Layer 4: Conv4 + ReLU
constexpr int CONV4_OUT = 256;
constexpr int CONV4_H = 16;
constexpr int CONV4_W = 16;
constexpr int UP2_H = 32;
constexpr int UP2_W = 32;

// Decoder Layer 5: Conv5 (output, no activation)
constexpr int CONV5_OUT = 3;
constexpr int CONV5_H = 32;
constexpr int CONV5_W = 32;
constexpr int OUTPUT_SIZE = CONV5_OUT * CONV5_H * CONV5_W;  // 3072

// ============================================================================
// Weight Dimensions
// ============================================================================

constexpr int KERNEL_SIZE = 3;
constexpr int K2 = KERNEL_SIZE * KERNEL_SIZE;  // 9

// Layer 1: Input(3) -> Conv1(256)
constexpr int W1_SIZE = CONV1_OUT * INPUT_C * K2;     // 256 * 3 * 9 = 6,912
constexpr int B1_SIZE = CONV1_OUT;                     // 256

// Layer 2: Conv1(256) -> Conv2(128)
constexpr int W2_SIZE = CONV2_OUT * CONV1_OUT * K2;   // 128 * 256 * 9 = 294,912
constexpr int B2_SIZE = CONV2_OUT;                     // 128

// Layer 3: Conv2(128) -> Conv3(128)
constexpr int W3_SIZE = CONV3_OUT * CONV2_OUT * K2;   // 128 * 128 * 9 = 147,456
constexpr int B3_SIZE = CONV3_OUT;                     // 128

// Layer 4: Conv3(128) -> Conv4(256)
constexpr int W4_SIZE = CONV4_OUT * CONV3_OUT * K2;   // 256 * 128 * 9 = 294,912
constexpr int B4_SIZE = CONV4_OUT;                     // 256

// Layer 5: Conv4(256) -> Conv5(3)
constexpr int W5_SIZE = CONV5_OUT * CONV4_OUT * K2;   // 3 * 256 * 9 = 6,912
constexpr int B5_SIZE = CONV5_OUT;                     // 3

// Total trainable parameters
constexpr int TOTAL_WEIGHTS = W1_SIZE + W2_SIZE + W3_SIZE + W4_SIZE + W5_SIZE;
constexpr int TOTAL_BIASES = B1_SIZE + B2_SIZE + B3_SIZE + B4_SIZE + B5_SIZE;
constexpr int TOTAL_PARAMS = TOTAL_WEIGHTS + TOTAL_BIASES;  // 751,875

// ============================================================================
// Training Hyperparameters (defaults)
// ============================================================================

constexpr int DEFAULT_BATCH_SIZE = 64;
constexpr int DEFAULT_EPOCHS = 20;
constexpr float DEFAULT_LEARNING_RATE = 0.001f;
constexpr float DEFAULT_GRAD_CLIP = 1.0f;

}  // namespace gpu_constants

#endif  // GPU_CONSTANTS_H
