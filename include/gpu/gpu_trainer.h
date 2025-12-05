#ifndef GPU_TRAINER_H
#define GPU_TRAINER_H

#include "common/cifar10_dataset.h"
#include "gpu/gpu_autoencoder.h"

// GPU Training configuration
struct GPUTrainConfig {
    int batch_size = 64;       // Larger batch size for GPU
    int epochs = 20;
    float learning_rate = 0.001f;
    bool verbose = true;       // Print per-batch info
};

// Train the GPU autoencoder
void train_gpu_autoencoder(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    const GPUTrainConfig& config,
    const char* output_folder
);

// Extract features using GPU autoencoder
void extract_and_save_features_gpu(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif  // GPU_TRAINER_H
