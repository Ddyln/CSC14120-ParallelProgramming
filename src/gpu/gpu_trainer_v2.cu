// ============================================================================
// GPU Trainer V2 - Training with Optimized Autoencoder
// ============================================================================

#include "gpu/gpu_trainer_v2.h"
#include "gpu/gpu_layers_v2.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <string>

// ============================================================================
// TRAINING FUNCTION V2
// ============================================================================

void train_gpu_autoencoder_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const GPUTrainConfigV2& config,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU Autoencoder Training V2 (Optimized)\n");
    printf("========================================\n");
    printf("Optimizations: Kernel Fusion, Loop Unrolling, float4\n");
    printf("Batch size: %d\n", config.batch_size);
    printf("Epochs: %d\n", config.epochs);
    printf("Learning rate: %.6f\n", config.learning_rate);
    printf("Training samples: %d\n", dataset.train_size);
    printf("========================================\n\n");

    int num_batches = dataset.train_size / config.batch_size;
    int input_size = 3 * 32 * 32;
    int batch_bytes = config.batch_size * input_size * sizeof(float);

    // Allocate device memory for input batch
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch_bytes);
    cudaMalloc(&d_output, batch_bytes);

    // Host buffer for batch
    float* h_batch = new float[config.batch_size * input_size];

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;

        // Shuffle at start of each epoch
        dataset.shuffle_train();

        for (int batch = 0; batch < num_batches; ++batch) {
            // Load batch data
            for (int i = 0; i < config.batch_size; ++i) {
                int idx = batch * config.batch_size + i;
                memcpy(h_batch + i * input_size,
                       dataset.train_images + idx * input_size,
                       input_size * sizeof(float));
            }

            // Copy to device
            cudaMemcpy(d_input, h_batch, batch_bytes, cudaMemcpyHostToDevice);

            // Forward pass (using fused kernels)
            model.forward(d_input, d_output, config.batch_size);

            // Compute loss (using optimized kernel)
            float batch_loss = model.compute_loss(d_output, d_input, config.batch_size);
            epoch_loss += batch_loss;

            // Backward pass
            model.backward(d_input, d_input, config.batch_size);

            // Update weights (using vectorized SGD)
            model.update_weights(config.learning_rate);

            if (config.verbose && (batch + 1) % 100 == 0) {
                printf("  Epoch %d/%d, Batch %d/%d, Loss: %.6f\n",
                       epoch + 1, config.epochs, batch + 1, num_batches, batch_loss);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start).count();

        float avg_loss = epoch_loss / num_batches;
        printf("Epoch %d/%d completed in %.2f seconds, Avg Loss: %.6f\n",
               epoch + 1, config.epochs, epoch_duration / 1000.0f, avg_loss);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        total_end - total_start).count();

    printf("\n========================================\n");
    printf("Training completed in %ld seconds\n", total_duration);
    printf("========================================\n");

    // Save weights
    std::string weights_path = std::string(output_folder) + "/autoencoder_weights_v2.bin";
    model.save_weights(weights_path);

    // Cleanup
    delete[] h_batch;
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// FEATURE EXTRACTION V2
// ============================================================================

void extract_and_save_features_gpu_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("Feature Extraction V2 (Optimized)\n");
    printf("========================================\n");

    int input_size = 3 * 32 * 32;
    int feature_size = 128 * 8 * 8;  // 8192

    int batch_size = 64;
    int input_bytes = batch_size * input_size * sizeof(float);
    int feature_bytes = batch_size * feature_size * sizeof(float);

    // Allocate device memory
    float* d_input;
    float* d_features;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_features, feature_bytes);

    // Host buffers
    float* h_batch = new float[batch_size * input_size];
    float* h_features = new float[batch_size * feature_size];

    auto start = std::chrono::high_resolution_clock::now();

    // Extract training features
    printf("Extracting training features (%d images)...\n", dataset.train_size);
    
    float* train_features = new float[dataset.train_size * feature_size];
    int num_train_batches = (dataset.train_size + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_train_batches; ++batch) {
        int start_idx = batch * batch_size;
        int current_batch_size = std::min(batch_size, dataset.train_size - start_idx);

        // Load batch
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(h_batch + i * input_size,
                   dataset.train_images + (start_idx + i) * input_size,
                   input_size * sizeof(float));
        }

        cudaMemcpy(d_input, h_batch, current_batch_size * input_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Extract features using optimized encoder
        model.get_features(d_input, d_features, current_batch_size);

        cudaMemcpy(h_features, d_features, current_batch_size * feature_size * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Copy to output array
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(train_features + (start_idx + i) * feature_size,
                   h_features + i * feature_size,
                   feature_size * sizeof(float));
        }

        if ((batch + 1) % 100 == 0) {
            printf("  Processed %d/%d batches\n", batch + 1, num_train_batches);
        }
    }

    // Extract test features
    printf("Extracting test features (%d images)...\n", dataset.test_size);
    
    float* test_features = new float[dataset.test_size * feature_size];
    int num_test_batches = (dataset.test_size + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_test_batches; ++batch) {
        int start_idx = batch * batch_size;
        int current_batch_size = std::min(batch_size, dataset.test_size - start_idx);

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(h_batch + i * input_size,
                   dataset.test_images + (start_idx + i) * input_size,
                   input_size * sizeof(float));
        }

        cudaMemcpy(d_input, h_batch, current_batch_size * input_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        model.get_features(d_input, d_features, current_batch_size);

        cudaMemcpy(h_features, d_features, current_batch_size * feature_size * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(test_features + (start_idx + i) * feature_size,
                   h_features + i * feature_size,
                   feature_size * sizeof(float));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Feature extraction completed in %.2f seconds\n", duration / 1000.0f);

    // Save features in LIBSVM format
    printf("Saving features in LIBSVM format...\n");

    std::string train_path = std::string(output_folder) + "/train_features_v2.txt";
    std::string test_path = std::string(output_folder) + "/test_features_v2.txt";

    FILE* f_train = fopen(train_path.c_str(), "w");
    if (f_train) {
        for (int i = 0; i < dataset.train_size; ++i) {
            fprintf(f_train, "%d", dataset.train_labels[i]);
            for (int j = 0; j < feature_size; ++j) {
                float val = train_features[i * feature_size + j];
                if (val != 0.0f) {
                    fprintf(f_train, " %d:%.6f", j + 1, val);
                }
            }
            fprintf(f_train, "\n");
        }
        fclose(f_train);
        printf("Saved training features to %s\n", train_path.c_str());
    }

    FILE* f_test = fopen(test_path.c_str(), "w");
    if (f_test) {
        for (int i = 0; i < dataset.test_size; ++i) {
            fprintf(f_test, "%d", dataset.test_labels[i]);
            for (int j = 0; j < feature_size; ++j) {
                float val = test_features[i * feature_size + j];
                if (val != 0.0f) {
                    fprintf(f_test, " %d:%.6f", j + 1, val);
                }
            }
            fprintf(f_test, "\n");
        }
        fclose(f_test);
        printf("Saved test features to %s\n", test_path.c_str());
    }

    printf("\n========================================\n");
    printf("Feature extraction V2 completed!\n");
    printf("========================================\n");

    // Cleanup
    delete[] h_batch;
    delete[] h_features;
    delete[] train_features;
    delete[] test_features;
    cudaFree(d_input);
    cudaFree(d_features);
}
