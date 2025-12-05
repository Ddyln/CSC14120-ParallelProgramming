#ifndef GPU_TRAINER_V2_H
#define GPU_TRAINER_V2_H

#include "common/cifar10_dataset.h"
#include "gpu/gpu_autoencoder_v2.h"

// GPU Training configuration V2
struct GPUTrainConfigV2 {
    int batch_size = 64;       // Larger batch size for GPU
    int epochs = 20;
    float learning_rate = 0.001f;
    bool verbose = true;
};

// Train the GPU autoencoder V2 (with optimizations)
void train_gpu_autoencoder_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const GPUTrainConfigV2& config,
    const char* output_folder
);

// Extract features using GPU autoencoder V2
void extract_and_save_features_gpu_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif  // GPU_TRAINER_V2_H
