#include "gpu2/gpu2_trainer.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <climits>
#include <cfloat>

#include "common/cifar10_dataset.h"

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

void train_gpu2_autoencoder(
    GPU2Autoencoder& model,
    CIFAR10Dataset& dataset,
    const GPU2TrainConfig& config,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU2 Autoencoder Training (Optimized v2)\n");
    printf("Optimizations:\n");
    printf("- Constant Memory for Biases\n");
    printf("- Pinned Host Memory\n");
    printf("- Multi-Stream Pipeline (%d streams)\n", config.num_streams);
    printf("========================================\n");
    printf("Batch size: %d\n", config.batch_size);
    printf("Epochs: %d\n", config.epochs);
    printf("Learning rate: %.4f\n", config.learning_rate);
    printf("Training samples: %zu\n", dataset.train_size());
    printf("========================================\n\n");

    const size_t num_batches = dataset.train_size() / config.batch_size;
    const size_t output_size = config.batch_size * 3 * 32 * 32;

    // Get initial GPU memory
    size_t free_mem_before, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_before, &total_mem));
    size_t used_mem_before = total_mem - free_mem_before;

    // Allocate host output buffer (pinned memory)
    float* h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, output_size * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float best_loss = FLT_MAX;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        printf("Epoch %d/%d...\n", epoch + 1, config.epochs);
        fflush(stdout);
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Shuffle training data at the start of each epoch
        dataset.shuffle_train();
        dataset.reset_cursor();

        float epoch_loss = 0.0f;
        float epoch_forward_time = 0.0f;
        float epoch_backward_time = 0.0f;
        float epoch_update_time = 0.0f;
        float epoch_best_loss = FLT_MAX;

        for (size_t batch = 0; batch < num_batches; batch++) {
            // Get next batch from dataset
            auto batch_data = dataset.next_train_batch(config.batch_size);
            const float* batch_images = batch_data.first.data();

            // ================================================================
            // Forward Pass (GPU with Async Transfers)
            // ================================================================
            cudaEventRecord(start);
            
            model.forward(batch_images, h_output, config.batch_size);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float forward_ms = 0;
            cudaEventElapsedTime(&forward_ms, start, stop);
            epoch_forward_time += forward_ms;

            // ================================================================
            // Compute Loss
            // ================================================================
            float batch_loss = model.compute_loss(batch_images, config.batch_size);
            epoch_loss += batch_loss;
            epoch_best_loss = (batch_loss < epoch_best_loss) ? batch_loss : epoch_best_loss;

            // ================================================================
            // Backward Pass (GPU with Async Transfers)
            // ================================================================
            cudaEventRecord(start);
            
            model.backward(batch_images, batch_images, config.batch_size);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float backward_ms = 0;
            cudaEventElapsedTime(&backward_ms, start, stop);
            epoch_backward_time += backward_ms;

            // ================================================================
            // Weight Update (GPU)
            // ================================================================
            cudaEventRecord(start);
            
            model.update_weights(config.learning_rate);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float update_ms = 0;
            cudaEventElapsedTime(&update_ms, start, stop);
            epoch_update_time += update_ms;

            // Print per-batch info if verbose
            if (config.verbose && ((batch + 1) % 100 == 0 || batch == 0)) {
                printf("  Epoch %d/%d - Batch %zu/%zu - Loss: %.6f - "
                       "Forward: %.1f ms, Backward: %.1f ms, Update: %.1f ms\n",
                       epoch + 1, config.epochs,
                       batch + 1, num_batches,
                       batch_loss,
                       forward_ms, backward_ms, update_ms);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start
        );

        float avg_loss = epoch_loss / num_batches;
        printf("Epoch %d/%d Complete - Avg Loss: %.6f (Best: %.6f) - Time: %ld ms (Forward: %.0f ms, Backward: %.0f ms, Update: %.0f ms)\n",
               epoch + 1, config.epochs,
               avg_loss,
               epoch_best_loss,
               epoch_duration.count(),
               epoch_forward_time, epoch_backward_time, epoch_update_time);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        total_end - total_start
    );

    // Get final GPU memory
    size_t free_mem_after;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_after, &total_mem));
    size_t used_mem_after = total_mem - free_mem_after;
    size_t peak_mem_used = used_mem_before > used_mem_after ? used_mem_before : used_mem_after;

    printf("\n========================================\n");
    printf("Training Summary (GPU2 Optimized)\n");
    printf("========================================\n");
    printf("Best Loss: %.6f\n", best_loss);
    printf("Total Training Time: %ld seconds (%.1f min)\n", 
           total_duration.count(),
           total_duration.count() / 60.0f);
    printf("GPU Memory (Before): %.1f MB / %.1f MB\n", 
           used_mem_before / (1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f));
    printf("GPU Memory (After):  %.1f MB / %.1f MB\n",
           used_mem_after / (1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f));
    printf("Peak Memory Used:    %.1f MB\n",
           peak_mem_used / (1024.0f * 1024.0f));
    printf("========================================\n\n");

    printf("Saving GPU2 model weights...\n");
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/gpu2_autoencoder_weights.bin", output_folder);
    model.save_weights(model_path);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_output));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void extract_and_save_features_gpu2(
    GPU2Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU2 Feature Extraction\n");
    printf("========================================\n");

    const int feature_size = 8 * 8 * 128;  // 8192
    const int batch_size = 64;

    // Allocate feature buffers (pinned memory for faster transfers)
    float* train_features;
    float* test_features;
    float* batch_features;
    
    CUDA_CHECK(cudaMallocHost(&train_features, dataset.train_size() * feature_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&test_features, dataset.test_size() * feature_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&batch_features, batch_size * feature_size * sizeof(float)));

    auto start = std::chrono::high_resolution_clock::now();

    // Extract training features
    printf("Extracting training features (%zu images)...\n", dataset.train_size());
    size_t num_train_batches = dataset.train_size() / batch_size;
    size_t train_remaining = dataset.train_size() % batch_size;

    dataset.reset_cursor();
    for (size_t i = 0; i < num_train_batches; i++) {
        auto batch_data = dataset.next_train_batch(batch_size);
        const float* batch_images = batch_data.first.data();
        
        model.extract_features(batch_images, batch_features, batch_size);
        
        // Copy to output buffer
        memcpy(train_features + i * batch_size * feature_size,
               batch_features,
               batch_size * feature_size * sizeof(float));

        if ((i + 1) % 10 == 0) {
            printf("  Extracted %zu/%zu training batches\n", i + 1, num_train_batches);
        }
    }

    // Handle remaining training samples
    if (train_remaining > 0) {
        auto batch_data = dataset.next_train_batch(train_remaining);
        const float* batch_images = batch_data.first.data();
        
        model.extract_features(batch_images, batch_features, train_remaining);
        
        memcpy(train_features + num_train_batches * batch_size * feature_size,
               batch_features,
               train_remaining * feature_size * sizeof(float));
        printf("  Extracted remaining %zu training samples\n", train_remaining);
    }

    // Extract test features
    printf("Extracting test features (%zu images)...\n", dataset.test_size());
    const std::vector<float>& test_images = dataset.test_images();
    size_t num_test_batches = dataset.test_size() / batch_size;
    size_t test_remaining = dataset.test_size() % batch_size;

    for (size_t i = 0; i < num_test_batches; i++) {
        const float* batch_images = &test_images[i * batch_size * CIFAR10_IMAGE_SIZE];
        
        model.extract_features(batch_images, batch_features, batch_size);
        
        // Copy to output buffer
        memcpy(test_features + i * batch_size * feature_size,
               batch_features,
               batch_size * feature_size * sizeof(float));

        if ((i + 1) % 10 == 0) {
            printf("  Extracted %zu/%zu test batches\n", i + 1, num_test_batches);
        }
    }

    // Handle remaining test samples
    if (test_remaining > 0) {
        const float* batch_images = &test_images[num_test_batches * batch_size * CIFAR10_IMAGE_SIZE];
        
        model.extract_features(batch_images, batch_features, test_remaining);
        
        memcpy(test_features + num_test_batches * batch_size * feature_size,
               batch_features,
               test_remaining * feature_size * sizeof(float));
        printf("  Extracted remaining %zu test samples\n", test_remaining);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Save features
    printf("\nSaving features to binary files...\n");
    
    FILE* train_file = fopen("train_features_gpu2.bin", "wb");
    if (train_file) {
        fwrite(train_features, sizeof(float), dataset.train_size() * feature_size, train_file);
        fclose(train_file);
        printf("Training features saved to: train_features_gpu2.bin\n");
    }

    FILE* test_file = fopen("test_features_gpu2.bin", "wb");
    if (test_file) {
        fwrite(test_features, sizeof(float), dataset.test_size() * feature_size, test_file);
        fclose(test_file);
        printf("Test features saved to: test_features_gpu2.bin\n");
    }

    printf("\n========================================\n");
    printf("Feature Extraction Summary (GPU2)\n");
    printf("========================================\n");
    printf("Total feature extraction time: %ld ms (%.1f sec)\n", 
           duration.count(),
           duration.count() / 1000.0f);
    printf("========================================\n");
    printf("Total extraction time: %ld ms\n", duration.count());
    printf("========================================\n");

    // Cleanup
    CUDA_CHECK(cudaFreeHost(train_features));
    CUDA_CHECK(cudaFreeHost(test_features));
    CUDA_CHECK(cudaFreeHost(batch_features));
}
