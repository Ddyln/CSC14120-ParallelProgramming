#include "common/cifar10_dataset.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

// Internal helper to read a single CIFAR-10 batch file
bool CIFAR10Dataset::load_batch_file(const std::string& file_path, bool train) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        std::cerr << "[CIFAR10Dataset] Failed to open file: " << file_path << std::endl;
        return false;
    }

    // Each record: 1 byte label + 3072 bytes image
    const size_t record_size = 1 + CIFAR10_IMAGE_SIZE;

    // Determine number of records by file size
    in.seekg(0, std::ios::end);
    std::streampos file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size % record_size != 0) {
        std::cerr << "[CIFAR10Dataset] File size not aligned to record size in: "
                  << file_path << std::endl;
        return false;
    }
    size_t num_records = static_cast<size_t>(file_size) / record_size;

    std::vector<char> buffer(record_size);
    for (size_t i = 0; i < num_records; ++i) {
        if (!in.read(buffer.data(), record_size)) {
            std::cerr << "[CIFAR10Dataset] Early EOF reading: " << file_path
                      << std::endl;
            return false;
        }
        uint8_t label = static_cast<uint8_t>(buffer[0]);
        // Normalize pixels to [0,1]
        for (int p = 0; p < CIFAR10_IMAGE_SIZE; ++p) {
            float v = static_cast<unsigned char>(buffer[1 + p]) / 255.0f;
            if (train) {
                images_train_.push_back(v);
            } else {
                images_test_.push_back(v);
            }
        }
        if (train) {
            labels_train_.push_back(label);
        } else {
            labels_test_.push_back(label);
        }
    }

    return true;
}

bool CIFAR10Dataset::load_train(const std::string& data_dir) {
    images_train_.clear();
    labels_train_.clear();
    train_cursor_ = 0;

    bool ok = true;
    for (int i = 1; i <= CIFAR10_TRAIN_BATCHES; ++i) {
        std::string path = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        ok &= load_batch_file(path, true);
    }
    size_t loaded = train_size();
    if (loaded != CIFAR10_TRAIN_IMAGES) {
        std::cerr << "[CIFAR10Dataset] Warning: expected " << CIFAR10_TRAIN_IMAGES
                  << " train images, loaded " << loaded << std::endl;
    }
    return ok;
}

bool CIFAR10Dataset::load_test(const std::string& data_dir) {
    images_test_.clear();
    labels_test_.clear();
    std::string path = data_dir + "/test_batch.bin";
    bool ok = load_batch_file(path, false);
    size_t loaded = test_size();
    if (loaded != CIFAR10_TEST_IMAGES) {
        std::cerr << "[CIFAR10Dataset] Warning: expected " << CIFAR10_TEST_IMAGES
                  << " test images, loaded " << loaded << std::endl;
    }
    return ok;
}

void CIFAR10Dataset::shuffle_train() {
    size_t n = train_size();
    if (n == 0) return;
    // Create index permutation
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(idx.begin(), idx.end(), gen);

    // Apply permutation to images and labels
    std::vector<float> new_images(images_train_.size());
    std::vector<uint8_t> new_labels(labels_train_.size());
    for (size_t i = 0; i < n; ++i) {
        size_t src = idx[i];
        // copy one image (3072 floats)
        std::copy(
            images_train_.begin() + src * CIFAR10_IMAGE_SIZE,
            images_train_.begin() + (src + 1) * CIFAR10_IMAGE_SIZE,
            new_images.begin() + i * CIFAR10_IMAGE_SIZE
        );
        new_labels[i] = labels_train_[src];
    }
    images_train_.swap(new_images);
    labels_train_.swap(new_labels);
    train_cursor_ = 0;
}

std::pair<std::vector<float>, std::vector<uint8_t>> CIFAR10Dataset::next_train_batch(
    size_t batch_size
) {
    if (batch_size == 0) return {{}, {}};
    size_t total = train_size();
    if (total == 0) return {{}, {}};

    // If batch extends beyond end, start new epoch
    if (train_cursor_ + batch_size > total) {
        train_cursor_ = 0;
    }

    std::vector<float> batch_images(batch_size * CIFAR10_IMAGE_SIZE);
    std::vector<uint8_t> batch_labels(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = train_cursor_ + i;
        std::copy(
            images_train_.begin() + idx * CIFAR10_IMAGE_SIZE,
            images_train_.begin() + (idx + 1) * CIFAR10_IMAGE_SIZE,
            batch_images.begin() + i * CIFAR10_IMAGE_SIZE
        );
        batch_labels[i] = labels_train_[idx];
    }

    train_cursor_ += batch_size;
    return {std::move(batch_images), std::move(batch_labels)};
}
