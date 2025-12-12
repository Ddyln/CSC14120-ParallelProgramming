#include <stdio.h>
#include <cstring>

// Common
#include "common/cifar10_dataset.h"
#include "common/gpu_info.h"

// CPU implementation
#include "cpu/autoencoder.h"
#include "cpu/feature_extractor.h"
#include "cpu/trainer.h"

// GPU implementation
#include "gpu/gpu_autoencoder.h"
#include "gpu/gpu_trainer.h"

// GPU2 implementation (optimized)
#include "gpu2/gpu2_autoencoder.h"
#include "gpu2/gpu2_trainer.h"

void print_usage(const char* program_name) {
    printf("Usage: %s <input_folder> <output_folder> [--cpu | --gpu | --gpu_v2]\n", program_name);
    printf("\nOptions:\n");
    printf("  --cpu     Use CPU implementation (default)\n");
    printf("  --gpu     Use GPU baseline implementation\n");
    printf("  --gpu_v2  Use GPU optimized v2 (4 techniques: constant memory, pinned memory, multi-stream, memory coalescing)\n");
    printf("\nExample:\n");
    printf("  %s ./data ./output --gpu     # GPU baseline\n", program_name);
    printf("  %s ./data ./output --gpu_v2  # GPU v2 optimized\n", program_name);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    gpu_info::print();

    const char* input_folder = argv[1];
    const char* output_folder = argv[2];
    
    // Parse mode argument
    // 0: CPU (default), 1: GPU Baseline, 2: GPU v2 Optimized
    int mode = 0;
    if (argc >= 4) {
        if (strcmp(argv[3], "--gpu") == 0) {
            mode = 1;
        } else if (strcmp(argv[3], "--gpu_v2") == 0) {
            mode = 2;
        } else if (strcmp(argv[3], "--cpu") == 0) {
            mode = 0;
        } else {
            printf("Unknown option: %s\n", argv[3]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("\n========================================\n");
    printf("CIFAR-10 Autoencoder Feature Extraction\n");
    printf("========================================\n");
    printf("Input folder: %s\n", input_folder);
    printf("Output folder: %s\n", output_folder);
    const char* mode_str = (mode == 0) ? "CPU" : (mode == 1) ? "GPU Baseline" : "GPU v2 Optimized";
    printf("Mode: %s\n", mode_str);
    printf("========================================\n");

    // Load CIFAR-10 dataset
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

    if (mode == 1) {
        // ================================================================
        // GPU Baseline Implementation
        // ================================================================
        printf("\n>>> Using GPU Baseline Implementation <<<\n");
        
        GPUAutoencoder gpu_model;
        gpu_model.initialize();

        GPUTrainConfig gpu_config;
        gpu_config.batch_size = 256;
        gpu_config.epochs = 20;
        gpu_config.learning_rate = 0.001f;
        gpu_config.verbose = true;

        train_gpu_autoencoder(gpu_model, dataset, gpu_config, output_folder);
        extract_and_save_features_gpu(gpu_model, dataset, output_folder);

    } else if (mode == 2) {
        // ================================================================
        // GPU v2 Optimized Implementation
        // ================================================================
        printf("\n>>> Using GPU v2 Optimized Implementation <<<\n");
        printf(">>> Optimizations: Constant Memory + Pinned Memory + Multi-Stream Pipeline + Shared Memory Tiling + Memory Coalescing <<<\n");
        
        GPU2Autoencoder gpu2_model;
        gpu2_model.initialize();

        GPU2TrainConfig gpu2_config;
        gpu2_config.batch_size = 256;
        gpu2_config.epochs = 20;
        gpu2_config.learning_rate = 0.001f;
        gpu2_config.verbose = true;
        gpu2_config.num_streams = 3;

        train_gpu2_autoencoder(gpu2_model, dataset, gpu2_config, output_folder);
        extract_and_save_features_gpu2(gpu2_model, dataset, output_folder);

    } else {
        // ================================================================
        // CPU Implementation
        // ================================================================
        printf("\n>>> Using CPU Implementation <<<\n");
        
        Autoencoder model;
        model.initialize();

        TrainConfig config;
        config.batch_size = 32;
        config.epochs = 20;
        config.learning_rate = 0.001f;

        train_autoencoder(model, dataset, config, output_folder);
        extract_and_save_features(model, dataset, output_folder);
    }

    printf("\n========================================\n");
    printf("All tasks completed successfully!\n");
    printf("========================================\n");

    return 0;
}