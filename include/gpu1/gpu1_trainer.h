#ifndef GPU1_TRAINER_H
#define GPU1_TRAINER_H

#include "common/cifar10_dataset.h"
#include "gpu1/gpu1_autoencoder.h"

struct GPU1TrainConfig {
    int batch_size = 64;
    int epochs = 20;
    float learning_rate = 0.001f;
    bool verbose = true;
};

void train_gpu1_autoencoder(
    GPU1Autoencoder& model,
    CIFAR10Dataset& dataset,
    const GPU1TrainConfig& config,
    const char* output_folder
);

void extract_and_save_features_gpu1(
    GPU1Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif // GPU1_TRAINER_H
