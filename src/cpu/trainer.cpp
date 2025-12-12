#include "cpu/trainer.h"

#include <chrono>
#include <cstdio>

#include "cpu/layers.h"

void train_autoencoder(
    Autoencoder& model,
    CIFAR10Dataset& dataset,
    const TrainConfig& config,
    const char* output_folder
) {
    printf("\nTraining Autoencoder Config:\n");
    printf("\tBatch size: %d\n", config.batch_size);
    printf("\tEpochs: %d\n", config.epochs);
    printf("\tLearning rate: %.4f\n", config.learning_rate);

    const size_t num_batches = dataset.train_size() / config.batch_size;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        dataset.shuffle_train();

        float epoch_loss = 0.0f;

        for (size_t batch = 0; batch < num_batches; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            // load batch
            auto batch_data = dataset.next_train_batch(config.batch_size);
            const float* batch_images = batch_data.first.data();

            const size_t output_size = config.batch_size * 3 * 32 * 32;
            float* output = new float[output_size];

            // forward pass
            auto t_forward_start = std::chrono::high_resolution_clock::now();
            model.forward(batch_images, output, config.batch_size);
            auto t_forward_end = std::chrono::high_resolution_clock::now();
            auto forward_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_forward_end - t_forward_start
            );

            // compute loss
            float batch_loss = mse_loss(output, batch_images, output_size);

            // backward pass
            auto t_backward_start = std::chrono::high_resolution_clock::now();
            model.backward(batch_images, batch_images, config.batch_size);
            auto t_backward_end = std::chrono::high_resolution_clock::now();
            auto backward_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_backward_end - t_backward_start
            );

            // update weights
            model.update_weights(config.learning_rate);

            epoch_loss += batch_loss;

            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                batch_end - batch_start
            );

            printf(
                "  Epoch %d/%d - Batch %zu/%zu - Loss: %.6f - Forward: %ld ms - "
                "Backward: %ld ms - Total: %ld ms\n",
                epoch + 1,
                config.epochs,
                batch + 1,
                num_batches,
                batch_loss,
                forward_time.count(),
                backward_time.count(),
                batch_total.count()
            );

            delete[] output;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration =
            std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

        printf(
            "Epoch %d/%d - Loss: %.6f - Time: %ld seconds\n",
            epoch + 1,
            config.epochs,
            epoch_loss / num_batches,
            epoch_duration.count()
        );
    }

    printf("\nSaving model weights...\n");
    char model_path[512];
    snprintf(
        model_path,
        sizeof(model_path),
        "%s/autoencoder_weights.bin",
        output_folder
    );
    model.save_weights(model_path);
    printf("Model saved to: %s\n", model_path);

    printf("\nTraining complete!\n");
}
