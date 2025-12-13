#include "cpu/feature_extractor.h"

#include <cstdio>
#include <chrono>

void extract_and_save_features(
    Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
) {
    // Allocate memory for features (latent space: 8x8x128 = 8192)
    const int feature_size = 8 * 8 * 128;
    float* train_features = new float[dataset.train_size() * feature_size];
    float* test_features = new float[dataset.test_size() * feature_size];

    auto total_start = std::chrono::high_resolution_clock::now();

    // Extract training features
    printf("Extracting training features...\n");
    for (size_t i = 0; i < dataset.train_size(); i++) {
        float* img = dataset.get_train_image(i);
        model.extract_features(img, train_features + i * feature_size, 1);
        if ((i + 1) % 10000 == 0) {
            printf("  Processed %zu/%zu images\n", i + 1, dataset.train_size());
        }
    }

    // Extract test features
    printf("Extracting test features...\n");
    for (size_t i = 0; i < dataset.test_size(); i++) {
        float* img = dataset.get_test_image(i);
        model.extract_features(img, test_features + i * feature_size, 1);
        if ((i + 1) % 2000 == 0) {
            printf("  Processed %zu/%zu images\n", i + 1, dataset.test_size());
        }
    }

    // Save features to binary files
    char train_path[512], test_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train_features.bin", output_folder);
    snprintf(test_path, sizeof(test_path), "%s/test_features.bin", output_folder);

    FILE* f_train = fopen(train_path, "wb");
    if (f_train) {
        fwrite(
            train_features,
            sizeof(float),
            dataset.train_size() * feature_size,
            f_train
        );
        fclose(f_train);
        printf("Saved training features to: %s\n", train_path);
    }

    FILE* f_test = fopen(test_path, "wb");
    if (f_test) {
        fwrite(
            test_features,
            sizeof(float),
            dataset.test_size() * feature_size,
            f_test
        );
        fclose(f_test);
        printf("Saved test features to: %s\n", test_path);
    }

    delete[] train_features;
    delete[] test_features;

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        total_end - total_start
    );

    printf("\n========================================\n");
    printf("Feature Extraction Summary (CPU)\n");
    printf("========================================\n");
    printf("Total feature extraction time: %ld seconds (%.1f min)\n", 
           total_duration.count(),
           total_duration.count() / 60.0f);
    printf("========================================\n\n");
    printf("Feature extraction complete!\n");
}
