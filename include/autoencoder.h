#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <string>

// 5-layer convolutional autoencoder
// Architecture:
// Input: 3x32x32
// Conv1: 3->256, Pool: 32x32->16x16
// Conv2: 256->128, Pool: 16x16->8x8 (latent: 8x8x128)
// Conv3: 128->128, Upsample: 8x8->16x16
// Conv4: 128->256, Upsample: 16x16->32x32
// Conv5: 256->3 (output: 3x32x32)
//
// Total params: 751,875

class Autoencoder {
   public:
    Autoencoder();
    ~Autoencoder();

    // Initialize weights
    void initialize();

    // Forward pass: returns reconstruction
    void forward(const float *input, float *output, int batch_size);

    // Backward pass: compute gradients
    void backward(const float *input, const float *target, int batch_size);

    // Update weights using gradients
    void update_weights(float learning_rate);

    // Extract features (encoder only)
    void extract_features(const float *input, float *features, int batch_size);

    // Save/load model
    void save_weights(const std::string &filepath);
    void load_weights(const std::string &filepath);

   private:
    // Weights and biases for 5 conv layers
    float *w1, *b1;  // Conv1: 3->256 (3x3 kernel)
    float *w2, *b2;  // Conv2: 256->128 (3x3 kernel)
    float *w3, *b3;  // Conv3: 128->128 (3x3 kernel)
    float *w4, *b4;  // Conv4: 128->256 (3x3 kernel)
    float *w5, *b5;  // Conv5: 256->3 (3x3 kernel)

    // Gradients
    float *dw1, *db1;
    float *dw2, *db2;
    float *dw3, *db3;
    float *dw4, *db4;
    float *dw5, *db5;

    // Intermediate activations (for batch_size=32)
    float *act1, *act2, *act3, *act4, *act5;

    // Intermediate storage for backward pass
    float *pool1, *conv3_out, *up1, *up2;

    // Current batch size
    int current_batch_size;

    void allocate_memory();
    void free_memory();
};

#endif  // AUTOENCODER_H
