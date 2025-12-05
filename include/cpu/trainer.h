#ifndef TRAINER_H
#define TRAINER_H

#include "autoencoder.h"
#include "cifar10_dataset.h"

// Simple training configuration
struct TrainConfig {
    int batch_size = 32;
    int epochs = 20;
    float learning_rate = 0.001f;
};

// Train the autoencoder
void train_autoencoder(
    Autoencoder& model,
    CIFAR10Dataset& dataset,
    const TrainConfig& config,
    const char* output_folder
);

#endif  // TRAINER_H
