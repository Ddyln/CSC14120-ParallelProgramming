#ifndef CIFAR10_DATASET_H
#define CIFAR10_DATASET_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

constexpr int CIFAR10_IMAGE_WIDTH = 32;
constexpr int CIFAR10_IMAGE_HEIGHT = 32;
constexpr int CIFAR10_IMAGE_CHANNELS = 3;
constexpr int CIFAR10_IMAGE_SIZE =
    CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNELS;  // 3072
constexpr int CIFAR10_TRAIN_IMAGES = 50000;
constexpr int CIFAR10_TEST_IMAGES = 10000;
constexpr int CIFAR10_TRAIN_BATCHES = 5;

class CIFAR10Dataset {
   private:
    bool load_batch_file(const std::string& file_path, bool train);

    std::vector<float> images_train_;  // normalized pixel values [0,1]
    std::vector<uint8_t> labels_train_;
    std::vector<float> images_test_;
    std::vector<uint8_t> labels_test_;

    size_t train_cursor_ = 0;  // index of next image for batching
   public:
    bool load_train(const std::string& data_dir);
    bool load_test(const std::string& data_dir);
    void shuffle_train();

    // Get next training batch (advances internal cursor). Resets epoch if exceeds.
    // Returns pair of (images, labels) where images is a flattened float vector of size
    // batch_size * CIFAR10_IMAGE_SIZE
    std::pair<std::vector<float>, std::vector<uint8_t>> next_train_batch(
        size_t batch_size
    );

    size_t train_size() const { return images_train_.size() / CIFAR10_IMAGE_SIZE; }
    size_t test_size() const { return images_test_.size() / CIFAR10_IMAGE_SIZE; }

    // Access full data
    const std::vector<float>& train_images() const { return images_train_; }
    const std::vector<uint8_t>& train_labels() const { return labels_train_; }
    const std::vector<float>& test_images() const { return images_test_; }
    const std::vector<uint8_t>& test_labels() const { return labels_test_; }

    // Access individual images (returns pointer to image data)
    float* get_train_image(size_t index) {
        return &images_train_[index * CIFAR10_IMAGE_SIZE];
    }
    float* get_test_image(size_t index) {
        return &images_test_[index * CIFAR10_IMAGE_SIZE];
    }

    // Reset training cursor to beginning of epoch
    void reset_cursor() { train_cursor_ = 0; }
};

#endif  // CIFAR10_DATASET_H
