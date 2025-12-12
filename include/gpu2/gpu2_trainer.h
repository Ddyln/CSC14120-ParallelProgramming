#ifndef GPU2_TRAINER_H
#define GPU2_TRAINER_H

#include "common/cifar10_dataset.h"
#include "gpu2/gpu2_autoencoder.h"

// GPU2 Training configuration
struct GPU2TrainConfig {
    int batch_size = 64;       // Batch size for GPU
    int epochs = 20;
    float learning_rate = 0.001f;
    bool verbose = true;       // Print per-batch info
    int num_streams = 3;       // Number of CUDA streams for pipelining
};

// Train the GPU2 autoencoder with optimized pipeline
void train_gpu2_autoencoder(
    GPU2Autoencoder& model,
    CIFAR10Dataset& dataset,
    const GPU2TrainConfig& config,
    const char* output_folder
);

// Extract features using GPU2 autoencoder
void extract_and_save_features_gpu2(
    GPU2Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif  // GPU2_TRAINER_H
