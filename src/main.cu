
#include <stdio.h>

#include "autoencoder.h"
#include "cifar10_dataset.h"
#include "feature_extractor.h"
#include "gpu_info.h"
#include "trainer.h"

int main(int argc, char** argv) {
    gpu_info::print();

    const char* input_folder = argv[1];
    const char* output_folder = argv[2];

    printf("Input folder: %s\n", input_folder);
    printf("Output folder: %s\n", output_folder);

    // load CIFAR-10 dataset
    CIFAR10Dataset dataset;
    printf("\nLoading CIFAR-10 training data...\n");
    if (!dataset.load_train(input_folder)) {
        fprintf(stderr, "Failed to load training data from %s\n", input_folder);
        return 1;
    }
    printf("Loaded %zu training images.\n", dataset.train_size());

    printf("\nLoading CIFAR-10 test data...\n");
    if (!dataset.load_test(input_folder)) {
        fprintf(stderr, "Failed to load test data from %s\n", input_folder);
        return 1;
    }
    printf("Loaded %zu test images.\n", dataset.test_size());

    // create and train autoencoder
    Autoencoder model;
    model.initialize();

    TrainConfig config;

    train_autoencoder(model, dataset, config, output_folder);
    extract_and_save_features(model, dataset, output_folder);

    return 0;
}