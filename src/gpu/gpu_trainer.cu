#include "gpu/gpu_trainer.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <climits>
#include <cfloat>

#include "gpu/gpu_layers.cuh"
// 
// // Helper function to create directories
// static void create_directory(const char* path) {
//     #ifdef _WIN32
//         mkdir(path);
//     #else
//         mkdir(path, 0755);
//     #endif
// }

void train_gpu_autoencoder(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    const GPUTrainConfig& config,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU Autoencoder Training (Baseline)\n");
    printf("========================================\n");
    printf("Batch size: %d\n", config.batch_size);
    printf("Epochs: %d\n", config.epochs);
    printf("Learning rate: %.4f\n", config.learning_rate);
    printf("Training samples: %zu\n", dataset.train_size());
    printf("========================================\n\n");

    const size_t num_batches = dataset.train_size() / config.batch_size;
    const size_t output_size = config.batch_size * 3 * 32 * 32;

    // Allocate host output buffer
    float* h_output = new float[output_size];

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Không cần lấy bộ nhớ GPU ban đầu nữa vì model đã được khởi tạo trước đó
    // // Get initial GPU memory (after model initialization)
    // size_t free_mem_before, total_mem;
    // CUDA_CHECK(cudaMemGetInfo(&free_mem_before, &total_mem));
    // size_t used_mem_before = total_mem - free_mem_before;

    float best_loss = FLT_MAX;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
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
            // Forward Pass (GPU)
            // ================================================================
            cudaEventRecord(start);
            
            model.forward(batch_images, h_output, config.batch_size);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float forward_ms = 0;
            cudaEventElapsedTime(&forward_ms, start, stop);
            epoch_forward_time += forward_ms;

            // ================================================================
            // Compute Loss (target = input for autoencoder)
            // ================================================================
            float batch_loss = model.compute_loss(batch_images, config.batch_size);
            epoch_loss += batch_loss;
            epoch_best_loss = (batch_loss < epoch_best_loss) ? batch_loss : epoch_best_loss;

            // ================================================================
            // Backward Pass (GPU)
            // ================================================================
            cudaEventRecord(start);
            
            // For autoencoder, target = input
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
        if (avg_loss < best_loss) best_loss = avg_loss;
        
        printf("Epoch %d/%d Complete - Avg Loss: %.6f (Best: %.6f) - Time: %ld ms "
               "(Forward: %.0f ms, Backward: %.0f ms, Update: %.0f ms)\n",
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
    size_t free_mem_after, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_after, &total_mem));
    size_t used_mem_after = total_mem - free_mem_after;

    printf("\n========================================\n");
    printf("Training Summary (GPU Baseline)\n");
    printf("========================================\n");
    printf("Best Loss: %.6f\n", best_loss);
    printf("Total Training Time: %ld seconds (%.1f min)\n", 
           total_duration.count(),
           total_duration.count() / 60.0f);
    printf("GPU Memory Usage:  %.1f MB / %.1f MB\n",
           used_mem_after / (1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f));
    printf("========================================\n\n");

    // Save model weights
    printf("Saving GPU model weights...\n");
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/gpu_autoencoder_weights.bin", output_folder);
    model.save_weights(model_path);

    // Cleanup
    delete[] h_output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void extract_and_save_features_gpu(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU Feature Extraction\n");
    printf("========================================\n");
    //
    // // Create output directory if it doesn't exist
    // create_directory(output_folder);

    const int feature_size = 8 * 8 * 128;  // 8192
    const int batch_size = 64;

    // Allocate feature buffers
    float* train_features = new float[dataset.train_size() * feature_size];
    float* test_features = new float[dataset.test_size() * feature_size];
    float* batch_features = new float[batch_size * feature_size];

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

        if ((i + 1) % 100 == 0) {
            printf("  Processed %zu/%zu batches\n", i + 1, num_train_batches);
        }
    }

    // Handle remaining training images
    if (train_remaining > 0) {
        for (size_t i = 0; i < train_remaining; i++) {
            size_t idx = num_train_batches * batch_size + i;
            float* img = dataset.get_train_image(idx);
            float single_feature[8192];
            model.extract_features(img, single_feature, 1);
            memcpy(train_features + idx * feature_size,
                   single_feature,
                   feature_size * sizeof(float));
        }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        train_end - start
    );
    printf("Training feature extraction: %ld ms\n", train_duration.count());

    // Extract test features
    printf("Extracting test features (%zu images)...\n", dataset.test_size());
    size_t num_test_batches = dataset.test_size() / batch_size;
    size_t test_remaining = dataset.test_size() % batch_size;

    for (size_t i = 0; i < num_test_batches; i++) {
        // Get batch of test images
        float* batch_images = new float[batch_size * 3 * 32 * 32];
        for (int j = 0; j < batch_size; j++) {
            memcpy(batch_images + j * 3 * 32 * 32,
                   dataset.get_test_image(i * batch_size + j),
                   3 * 32 * 32 * sizeof(float));
        }
        
        model.extract_features(batch_images, batch_features, batch_size);
        
        memcpy(test_features + i * batch_size * feature_size,
               batch_features,
               batch_size * feature_size * sizeof(float));

        delete[] batch_images;

        if ((i + 1) % 25 == 0) {
            printf("  Processed %zu/%zu batches\n", i + 1, num_test_batches);
        }
    }

    // Handle remaining test images
    if (test_remaining > 0) {
        for (size_t i = 0; i < test_remaining; i++) {
            size_t idx = num_test_batches * batch_size + i;
            float* img = dataset.get_test_image(idx);
            float single_feature[8192];
            model.extract_features(img, single_feature, 1);
            memcpy(test_features + idx * feature_size,
                   single_feature,
                   feature_size * sizeof(float));
        }
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        test_end - train_end
    );
    printf("Test feature extraction: %ld ms\n", test_duration.count());

    // Save features
    char train_path[512], test_path[512];
    char train_labels_path[512], test_labels_path[512];
    
    snprintf(train_path, sizeof(train_path), "%s/gpu_train_features.bin", output_folder);
    snprintf(test_path, sizeof(test_path), "%s/gpu_test_features.bin", output_folder);
    snprintf(train_labels_path, sizeof(train_labels_path), "%s/train_labels.bin", output_folder);
    snprintf(test_labels_path, sizeof(test_labels_path), "%s/test_labels.bin", output_folder);

    // Save train features
    FILE* f_train = fopen(train_path, "wb");
    if (f_train) {
        fwrite(train_features, sizeof(float), dataset.train_size() * feature_size, f_train);
        fclose(f_train);
        printf("Saved training features to: %s\n", train_path);
    }

    // Save test features
    FILE* f_test = fopen(test_path, "wb");
    if (f_test) {
        fwrite(test_features, sizeof(float), dataset.test_size() * feature_size, f_test);
        fclose(f_test);
        printf("Saved test features to: %s\n", test_path);
    }

    // Save labels
    FILE* f_train_labels = fopen(train_labels_path, "wb");
    if (f_train_labels) {
        fwrite(dataset.train_labels().data(), sizeof(uint8_t), dataset.train_size(), f_train_labels);
        fclose(f_train_labels);
        printf("Saved training labels to: %s\n", train_labels_path);
    }

    FILE* f_test_labels = fopen(test_labels_path, "wb");
    if (f_test_labels) {
        fwrite(dataset.test_labels().data(), sizeof(uint8_t), dataset.test_size(), f_test_labels);
        fclose(f_test_labels);
        printf("Saved test labels to: %s\n", test_labels_path);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - start
    );
    printf("\n========================================\n");
    printf("Feature Extraction Summary (GPU)\n");
    printf("========================================\n");
    printf("Total feature extraction time: %ld ms (%.1f sec)\n", 
           total_duration.count(),
           total_duration.count() / 1000.0f);
    printf("========================================\n");

    delete[] train_features;
    delete[] test_features;
    delete[] batch_features;
}
