#include "gpu1/gpu1_trainer.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <cstring>

void train_gpu1_autoencoder(
    GPU1Autoencoder& model,
    CIFAR10Dataset& dataset,
    const GPU1TrainConfig& config,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU Autoencoder Training (v1 - fused conv + shared memory)\n");
    printf("========================================\n");
    printf("Batch size: %d\n", config.batch_size);
    printf("Epochs: %d\n", config.epochs);
    printf("Learning rate: %.4f\n", config.learning_rate);
    printf("Training samples: %zu\n", dataset.train_size());
    printf("========================================\n\n");

    const size_t num_batches = dataset.train_size() / config.batch_size;
    const size_t output_size = config.batch_size * 3 * 32 * 32;

    float* h_output = new float[output_size];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle_train();
        dataset.reset_cursor();

        float epoch_loss = 0.0f;
        float epoch_forward_time = 0.0f;
        float epoch_backward_time = 0.0f;
        float epoch_update_time = 0.0f;

        for (size_t batch = 0; batch < num_batches; batch++) {
            auto batch_data = dataset.next_train_batch(config.batch_size);
            const float* batch_images = batch_data.first.data();

            cudaEventRecord(start);
            model.forward(batch_images, h_output, config.batch_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float forward_ms = 0.0f;
            cudaEventElapsedTime(&forward_ms, start, stop);
            epoch_forward_time += forward_ms;

            float batch_loss = model.compute_loss(batch_images, config.batch_size);
            epoch_loss += batch_loss;

            cudaEventRecord(start);
            model.backward(batch_images, batch_images, config.batch_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float backward_ms = 0.0f;
            cudaEventElapsedTime(&backward_ms, start, stop);
            epoch_backward_time += backward_ms;

            cudaEventRecord(start);
            model.update_weights(config.learning_rate);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float update_ms = 0.0f;
            cudaEventElapsedTime(&update_ms, start, stop);
            epoch_update_time += update_ms;

            if (config.verbose && ((batch + 1) % 100 == 0 || batch == 0)) {
                printf("  Epoch %d/%d - Batch %zu/%zu - Loss: %.6f - Forward: %.1f ms, Backward: %.1f ms, Update: %.1f ms\n",
                       epoch + 1, config.epochs,
                       batch + 1, num_batches,
                       batch_loss,
                       forward_ms, backward_ms, update_ms);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

        float avg_loss = epoch_loss / num_batches;
        printf("Epoch %d/%d Complete - Avg Loss: %.6f - Time: %ld ms (Forward: %.0f ms, Backward: %.0f ms, Update: %.0f ms)\n",
               epoch + 1, config.epochs,
               avg_loss,
               epoch_duration.count(),
               epoch_forward_time, epoch_backward_time, epoch_update_time);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);

    printf("\n========================================\n");
    printf("GPU v1 Training Complete!\n");
    printf("Total training time: %ld seconds\n", total_duration.count());
    printf("========================================\n");

    printf("\nSaving GPU1 model weights...\n");
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/gpu1_autoencoder_weights.bin", output_folder);
    model.save_weights(model_path);

    delete[] h_output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void extract_and_save_features_gpu1(
    GPU1Autoencoder& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
) {
    printf("\n========================================\n");
    printf("GPU v1 Feature Extraction\n");
    printf("========================================\n");

    const int feature_size = 8 * 8 * 128;
    const int batch_size = 64;

    float* train_features = new float[dataset.train_size() * feature_size];
    float* test_features  = new float[dataset.test_size()  * feature_size];
    float* batch_features = new float[batch_size * feature_size];

    auto start = std::chrono::high_resolution_clock::now();

    printf("Extracting training features (%zu images)...\n", dataset.train_size());
    size_t num_train_batches = dataset.train_size() / batch_size;
    size_t train_remaining   = dataset.train_size() % batch_size;

    dataset.reset_cursor();
    for (size_t i = 0; i < num_train_batches; i++) {
        auto batch_data = dataset.next_train_batch(batch_size);
        const float* batch_images = batch_data.first.data();

        model.extract_features(batch_images, batch_features, batch_size);

        std::memcpy(train_features + i * batch_size * feature_size,
                    batch_features,
                    batch_size * feature_size * sizeof(float));

        if ((i + 1) % 100 == 0) {
            printf("  Processed %zu/%zu training batches\n", i + 1, num_train_batches);
        }
    }

    if (train_remaining > 0) {
        auto batch_data = dataset.next_train_batch(train_remaining);
        const float* batch_images = batch_data.first.data();

        model.extract_features(batch_images, batch_features, train_remaining);

        std::memcpy(train_features + num_train_batches * batch_size * feature_size,
                    batch_features,
                    train_remaining * feature_size * sizeof(float));
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - start);
    printf("Training feature extraction: %ld ms\n", train_duration.count());

    printf("Extracting test features (%zu images)...\n", dataset.test_size());
    const std::vector<float>& test_images = dataset.test_images();
    size_t num_test_batches = dataset.test_size() / batch_size;
    size_t test_remaining   = dataset.test_size() % batch_size;

    for (size_t i = 0; i < num_test_batches; i++) {
        const float* batch_images = &test_images[i * batch_size * CIFAR10_IMAGE_SIZE];

        model.extract_features(batch_images, batch_features, batch_size);

        std::memcpy(test_features + i * batch_size * feature_size,
                    batch_features,
                    batch_size * feature_size * sizeof(float));

        if ((i + 1) % 25 == 0) {
            printf("  Processed %zu/%zu test batches\n", i + 1, num_test_batches);
        }
    }

    if (test_remaining > 0) {
        const float* batch_images = &test_images[num_test_batches * batch_size * CIFAR10_IMAGE_SIZE];

        model.extract_features(batch_images, batch_features, test_remaining);

        std::memcpy(test_features + num_test_batches * batch_size * feature_size,
                    batch_features,
                    test_remaining * feature_size * sizeof(float));
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - train_end);
    printf("Test feature extraction: %ld ms\n", test_duration.count());

    char train_path[512], test_path[512];
    snprintf(train_path, sizeof(train_path), "%s/gpu1_train_features.bin", output_folder);
    snprintf(test_path, sizeof(test_path),  "%s/gpu1_test_features.bin",  output_folder);

    FILE* f_train = fopen(train_path, "wb");
    if (f_train) {
        fwrite(train_features, sizeof(float), dataset.train_size() * feature_size, f_train);
        fclose(f_train);
        printf("Saved training features to: %s\n", train_path);
    }

    FILE* f_test = fopen(test_path, "wb");
    if (f_test) {
        fwrite(test_features, sizeof(float), dataset.test_size() * feature_size, f_test);
        fclose(f_test);
        printf("Saved test features to: %s\n", test_path);
    }

    // also save labels (reuse same filenames as gpu baseline for convenience)
    char train_labels_path[512], test_labels_path[512];
    snprintf(train_labels_path, sizeof(train_labels_path), "%s/train_labels.bin", output_folder);
    snprintf(test_labels_path,  sizeof(test_labels_path),  "%s/test_labels.bin",  output_folder);

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

    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("\nTotal feature extraction time (GPU v1): %ld ms\n", total_duration.count());
    printf("========================================\n");

    delete[] train_features;
    delete[] test_features;
    delete[] batch_features;
}
