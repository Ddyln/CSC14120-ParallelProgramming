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

// GPU V2 implementation (optimized)
#include "gpu/gpu_autoencoder_v2.h"
#include "gpu/gpu_trainer_v2.h"

void print_usage(const char* program_name) {
    printf("Usage: %s <input_folder> <output_folder> [--cpu | --gpu | --gpu-v2]\n", program_name);
    printf("\nOptions:\n");
    printf("  --cpu     Use CPU implementation (default)\n");
    printf("  --gpu     Use GPU baseline implementation\n");
    printf("  --gpu-v2  Use GPU optimized V2 (Kernel Fusion, Loop Unrolling, float4)\n");
    printf("\nExample:\n");
    printf("  %s ./data ./output --gpu-v2\n", program_name);
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
    enum Mode { CPU_MODE, GPU_BASELINE, GPU_V2 };
    Mode mode = CPU_MODE;
    
    if (argc >= 4) {
        if (strcmp(argv[3], "--gpu") == 0) {
            mode = GPU_BASELINE;
        } else if (strcmp(argv[3], "--gpu-v2") == 0) {
            mode = GPU_V2;
        } else if (strcmp(argv[3], "--cpu") == 0) {
            mode = CPU_MODE;
        } else {
            printf("Unknown option: %s\n", argv[3]);
            print_usage(argv[0]);
            return 1;
        }
    }

    const char* mode_str = (mode == GPU_V2) ? "GPU Optimized V2" : 
                           (mode == GPU_BASELINE) ? "GPU Baseline" : "CPU";

    printf("\n========================================\n");
    printf("CIFAR-10 Autoencoder Feature Extraction\n");
    printf("========================================\n");
    printf("Input folder: %s\n", input_folder);
    printf("Output folder: %s\n", output_folder);
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

    if (mode == GPU_V2) {
        // ================================================================
        // GPU Optimized V2 Implementation
        // ================================================================
        printf("\n>>> Using GPU Optimized V2 Implementation <<<\n");
        printf("Optimizations: Kernel Fusion, Loop Unrolling, float4\n");
        
        GPUAutoencoderV2 gpu_model_v2;

        GPUTrainConfigV2 gpu_config_v2;
        gpu_config_v2.batch_size = 64;
        gpu_config_v2.epochs = 20;
        gpu_config_v2.learning_rate = 0.001f;
        gpu_config_v2.verbose = true;

        train_gpu_autoencoder_v2(gpu_model_v2, dataset, gpu_config_v2, output_folder);
        extract_features_gpu_v2(gpu_model_v2, dataset, output_folder);

    } else if (mode == GPU_BASELINE) {
        // ================================================================
        // GPU Baseline Implementation
        // ================================================================
        printf("\n>>> Using GPU Baseline Implementation <<<\n");
        
        GPUAutoencoder gpu_model;
        gpu_model.initialize();

        GPUTrainConfig gpu_config;
        gpu_config.batch_size = 64;
        gpu_config.epochs = 20;
        gpu_config.learning_rate = 0.001f;
        gpu_config.verbose = true;

        train_gpu_autoencoder(gpu_model, dataset, gpu_config, output_folder);
        extract_and_save_features_gpu(gpu_model, dataset, output_folder);

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