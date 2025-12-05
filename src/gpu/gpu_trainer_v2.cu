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
    printf("Training samples: %zu\n", dataset.train_size());
    printf("========================================\n\n");

    int num_batches = dataset.train_size() / config.batch_size;
    int input_size = 3 * 32 * 32;
    int batch_bytes = config.batch_size * input_size * sizeof(float);

    // Host buffers for batch
    float* h_batch = new float[config.batch_size * input_size];
    
    // Device buffers for batch (Optimized: Allocate once, reuse)
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch_bytes);
    cudaMalloc(&d_output, batch_bytes);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;

        // Shuffle at start of each epoch
        dataset.shuffle_train();

        for (int batch = 0; batch < num_batches; ++batch) {
            // Load batch data
            const float* train_data = dataset.train_images().data();
            for (int i = 0; i < config.batch_size; ++i) {
                int idx = batch * config.batch_size + i;
                // Copy batch data (host to host)
                memcpy(h_batch + i * input_size,
                       train_data + idx * input_size,
                       input_size * sizeof(float));
            }
            
            // Copy batch to device ONCE per iteration
            cudaMemcpy(d_input, h_batch, batch_bytes, cudaMemcpyHostToDevice);

            // Forward pass (Device API - computes activations and output)
            model.forward(d_input, d_output, config.batch_size);

            // Compute loss (Device API - No extra copies)
            float batch_loss = model.compute_loss(d_output, d_input, config.batch_size);
            epoch_loss += batch_loss;

            // Backward pass (Device API - uses activations from forward)
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

    // Host buffers
    float* h_batch = new float[batch_size * input_size];
    float* h_features = new float[batch_size * feature_size];
    
    // Device buffers (allocated once, reused)
    float* d_batch;
    float* d_features;
    cudaMalloc(&d_batch, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_features, batch_size * feature_size * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    // Extract training features
    int train_count = dataset.train_size();
    printf("Extracting training features (%d images)...\n", train_count);
    
    float* train_features = new float[train_count * feature_size];
    int num_train_batches = (train_count + batch_size - 1) / batch_size;

    const float* train_data = dataset.train_images().data();
    for (int batch = 0; batch < num_train_batches; ++batch) {
        int start_idx = batch * batch_size;
        int current_batch_size = std::min(batch_size, train_count - start_idx);

        // Load batch
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(h_batch + i * input_size,
                   train_data + (start_idx + i) * input_size,
                   input_size * sizeof(float));
        }

        // Copy to device
        cudaMemcpy(d_batch, h_batch, current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Extract features using optimized encoder (device API)
        model.get_features(d_batch, d_features, current_batch_size);
        
        // Copy features back to host
        cudaMemcpy(h_features, d_features, current_batch_size * feature_size * sizeof(float), cudaMemcpyDeviceToHost);

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
    int test_count = dataset.test_size();
    printf("Extracting test features (%d images)...\n", test_count);
    
    float* test_features = new float[test_count * feature_size];
    int num_test_batches = (test_count + batch_size - 1) / batch_size;

    const float* test_data = dataset.test_images().data();
    for (int batch = 0; batch < num_test_batches; ++batch) {
        int start_idx = batch * batch_size;
        int current_batch_size = std::min(batch_size, test_count - start_idx);

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(h_batch + i * input_size,
                   test_data + (start_idx + i) * input_size,
                   input_size * sizeof(float));
        }

        // Copy to device
        cudaMemcpy(d_batch, h_batch, current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Extract features (device API)
        model.get_features(d_batch, d_features, current_batch_size);
        
        // Copy features back to host
        cudaMemcpy(h_features, d_features, current_batch_size * feature_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(test_features + (start_idx + i) * feature_size,
                   h_features + i * feature_size,
                   feature_size * sizeof(float));
        }
    }
    
    // Free device buffers
    cudaFree(d_batch);
    cudaFree(d_features);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Feature extraction completed in %.2f seconds\n", duration / 1000.0f);

    // Save features in LIBSVM format
    printf("Saving features in LIBSVM format...\n");

    std::string train_path = std::string(output_folder) + "/train_features_v2.txt";
    std::string test_path = std::string(output_folder) + "/test_features_v2.txt";

    const uint8_t* train_labels = dataset.train_labels().data();
    FILE* f_train = fopen(train_path.c_str(), "w");
    if (f_train) {
        for (int i = 0; i < train_count; ++i) {
            fprintf(f_train, "%d", train_labels[i]);
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

    const uint8_t* test_labels = dataset.test_labels().data();
    FILE* f_test = fopen(test_path.c_str(), "w");
    if (f_test) {
        for (int i = 0; i < test_count; ++i) {
            fprintf(f_test, "%d", test_labels[i]);
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

    // Cleanup (host buffers only - V2 manages device memory internally)
    delete[] h_batch;
    delete[] h_features;
    delete[] train_features;
    delete[] test_features;
}
