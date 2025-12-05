#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "autoencoder.h"
#include "cifar10_dataset.h"

// Extract features from dataset and save to files
void extract_and_save_features(
    Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif  // FEATURE_EXTRACTOR_H
