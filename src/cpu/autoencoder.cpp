#include "cpu/autoencoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "cpu/layers.h"

Autoencoder::Autoencoder() {
    w1 = b1 = w2 = b2 = w3 = b3 = w4 = b4 = w5 = b5 = nullptr;
    dw1 = db1 = dw2 = db2 = dw3 = db3 = dw4 = db4 = dw5 = db5 = nullptr;
    act1 = act2 = act3 = act4 = act5 = nullptr;
    pool1 = conv3_out = up1 = up2 = nullptr;
    current_batch_size = 32;
}

Autoencoder::~Autoencoder() { free_memory(); }

void Autoencoder::allocate_memory() {
    // Allocate weights and biases
    // w1: 256*3*3*3 = 6912, b1: 256
    w1 = new float[256 * 3 * 3 * 3];
    b1 = new float[256];
    dw1 = new float[256 * 3 * 3 * 3];
    db1 = new float[256];

    // w2: 128*256*3*3 = 294912, b2: 128
    w2 = new float[128 * 256 * 3 * 3];
    b2 = new float[128];
    dw2 = new float[128 * 256 * 3 * 3];
    db2 = new float[128];

    // w3: 128*128*3*3 = 147456, b3: 128
    w3 = new float[128 * 128 * 3 * 3];
    b3 = new float[128];
    dw3 = new float[128 * 128 * 3 * 3];
    db3 = new float[128];

    // w4: 256*128*3*3 = 294912, b4: 256
    w4 = new float[256 * 128 * 3 * 3];
    b4 = new float[256];
    dw4 = new float[256 * 128 * 3 * 3];
    db4 = new float[256];

    // w5: 3*256*3*3 = 6912, b5: 3
    w5 = new float[3 * 256 * 3 * 3];
    b5 = new float[3];
    dw5 = new float[3 * 256 * 3 * 3];
    db5 = new float[3];

    // Allocate activations (for batch_size=32)
    act1 = new float[32 * 256 * 32 * 32];
    act2 = new float[32 * 128 * 16 * 16];
    act3 = new float[32 * 128 * 8 * 8];
    act4 = new float[32 * 256 * 16 * 16];
    act5 = new float[32 * 3 * 32 * 32];

    // Allocate intermediate storage for backward pass
    pool1 = new float[32 * 256 * 16 * 16];
    conv3_out = new float[32 * 128 * 8 * 8];
    up1 = new float[32 * 128 * 16 * 16];
    up2 = new float[32 * 256 * 32 * 32];
}

void Autoencoder::free_memory() {
    delete[] w1;
    delete[] b1;
    delete[] dw1;
    delete[] db1;
    delete[] w2;
    delete[] b2;
    delete[] dw2;
    delete[] db2;
    delete[] w3;
    delete[] b3;
    delete[] dw3;
    delete[] db3;
    delete[] w4;
    delete[] b4;
    delete[] dw4;
    delete[] db4;
    delete[] w5;
    delete[] b5;
    delete[] dw5;
    delete[] db5;
    delete[] act1;
    delete[] act2;
    delete[] act3;
    delete[] act4;
    delete[] act5;
    delete[] pool1;
    delete[] conv3_out;
    delete[] up1;
    delete[] up2;
}

void Autoencoder::initialize() {
    allocate_memory();

    // Seed random number generator for backward pass
    srand(time(NULL));

    // Initialize weights using Xavier initialization
    init_weights_xavier(w1, 3, 256);
    init_weights_xavier(w2, 256, 128);
    init_weights_xavier(w3, 128, 128);
    init_weights_xavier(w4, 128, 256);
    init_weights_xavier(w5, 256, 3);

    // Initialize biases to zero
    memset(b1, 0, 256 * sizeof(float));
    memset(b2, 0, 128 * sizeof(float));
    memset(b3, 0, 128 * sizeof(float));
    memset(b4, 0, 256 * sizeof(float));
    memset(b5, 0, 3 * sizeof(float));
}

void Autoencoder::forward(const float* input, float* output, int batch_size) {
    current_batch_size = batch_size;

    // Encoder
    // Conv1: 3->256, 32x32
    conv2d_forward(input, w1, b1, act1, batch_size, 3, 256, 32, 32);
    relu_forward(act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    maxpool2d_forward(act1, pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16
    conv2d_forward(pool1, w2, b2, act2, batch_size, 256, 128, 16, 16);
    relu_forward(act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    maxpool2d_forward(act2, act3, batch_size, 128, 16, 16);

    // Decoder
    // Conv3: 128->128, 8x8
    conv2d_forward(act3, w3, b3, conv3_out, batch_size, 128, 128, 8, 8);
    relu_forward(conv3_out, batch_size * 128 * 8 * 8);

    // Upsample1: 8x8->16x16
    upsample2d_forward(conv3_out, up1, batch_size, 128, 8, 8);

    // Conv4: 128->256, 16x16
    conv2d_forward(up1, w4, b4, act4, batch_size, 128, 256, 16, 16);
    relu_forward(act4, batch_size * 256 * 16 * 16);

    // Upsample2: 16x16->32x32
    upsample2d_forward(act4, up2, batch_size, 256, 16, 16);

    // Conv5: 256->3, 32x32 (output)
    conv2d_forward(up2, w5, b5, act5, batch_size, 256, 3, 32, 32);

    // Copy to output
    memcpy(output, act5, batch_size * 3 * 32 * 32 * sizeof(float));
}

void Autoencoder::backward(const float* input, const float* target, int batch_size) {
    memset(dw1, 0, 256 * 3 * 3 * 3 * sizeof(float));
    memset(db1, 0, 256 * sizeof(float));
    memset(dw2, 0, 128 * 256 * 3 * 3 * sizeof(float));
    memset(db2, 0, 128 * sizeof(float));
    memset(dw3, 0, 128 * 128 * 3 * 3 * sizeof(float));
    memset(db3, 0, 128 * sizeof(float));
    memset(dw4, 0, 256 * 128 * 3 * 3 * sizeof(float));
    memset(db4, 0, 256 * sizeof(float));
    memset(dw5, 0, 3 * 256 * 3 * 3 * sizeof(float));
    memset(db5, 0, 3 * sizeof(float));

    // Allocate gradient buffers
    float* dL_dact5 = new float[batch_size * 3 * 32 * 32];
    float* dL_dup2 = new float[batch_size * 256 * 32 * 32];
    float* dL_dact4 = new float[batch_size * 256 * 16 * 16];
    float* dL_dup1 = new float[batch_size * 128 * 16 * 16];
    float* dL_dconv3 = new float[batch_size * 128 * 8 * 8];
    float* dL_dact3 = new float[batch_size * 128 * 8 * 8];
    float* dL_dact2 = new float[batch_size * 128 * 16 * 16];
    float* dL_dpool1 = new float[batch_size * 256 * 16 * 16];
    float* dL_dact1 = new float[batch_size * 256 * 32 * 32];

    // 1. Gradient at output: dL/dact5 = 2(act5 - target) / N
    const size_t output_size = batch_size * 3 * 32 * 32;
    for (size_t i = 0; i < output_size; i++) {
        dL_dact5[i] = 2.0f * (act5[i] - target[i]) / output_size;
    }

    // 2. Backward through Conv5: 256->3, 32x32
    conv2d_backward(up2, w5, dL_dact5, dL_dup2, dw5, db5, batch_size, 256, 3, 32, 32);

    // 3. Backward through Upsample2: 16x16->32x32
    upsample2d_backward(dL_dup2, dL_dact4, batch_size, 256, 16, 16);

    // 4. Backward through ReLU4
    relu_backward(act4, dL_dact4, dL_dact4, batch_size * 256 * 16 * 16);

    // 5. Backward through Conv4: 128->256, 16x16
    conv2d_backward(up1, w4, dL_dact4, dL_dup1, dw4, db4, batch_size, 128, 256, 16, 16);

    // 6. Backward through Upsample1: 8x8->16x16
    upsample2d_backward(dL_dup1, dL_dconv3, batch_size, 128, 8, 8);

    // 7. Backward through ReLU3
    relu_backward(conv3_out, dL_dconv3, dL_dconv3, batch_size * 128 * 8 * 8);

    // 8. Backward through Conv3: 128->128, 8x8
    conv2d_backward(
        act3,
        w3,
        dL_dconv3,
        dL_dact3,
        dw3,
        db3,
        batch_size,
        128,
        128,
        8,
        8
    );

    // 9. Backward through MaxPool2: 16x16->8x8
    maxpool2d_backward(act2, act3, dL_dact3, dL_dact2, batch_size, 128, 16, 16);

    // 10. Backward through ReLU2
    relu_backward(act2, dL_dact2, dL_dact2, batch_size * 128 * 16 * 16);

    // 11. Backward through Conv2: 256->128, 16x16
    conv2d_backward(
        pool1,
        w2,
        dL_dact2,
        dL_dpool1,
        dw2,
        db2,
        batch_size,
        256,
        128,
        16,
        16
    );

    // 12. Backward through MaxPool1: 32x32->16x16
    maxpool2d_backward(act1, pool1, dL_dpool1, dL_dact1, batch_size, 256, 32, 32);

    // 13. Backward through ReLU1
    relu_backward(act1, dL_dact1, dL_dact1, batch_size * 256 * 32 * 32);

    // 14. Backward through Conv1: 3->256, 32x32
    float* dL_dinput = new float[batch_size * 3 * 32 * 32];
    conv2d_backward(
        input,
        w1,
        dL_dact1,
        dL_dinput,
        dw1,
        db1,
        batch_size,
        3,
        256,
        32,
        32
    );

    // Cleanup
    delete[] dL_dact5;
    delete[] dL_dup2;
    delete[] dL_dact4;
    delete[] dL_dup1;
    delete[] dL_dconv3;
    delete[] dL_dact3;
    delete[] dL_dact2;
    delete[] dL_dpool1;
    delete[] dL_dact1;
    delete[] dL_dinput;
}

void Autoencoder::update_weights(float learning_rate) {
    // Update weights: w = w - lr * dw
    const float clip_value = 1.0f;

    int sizes[] = {
        256 * 3 * 3 * 3,
        256,
        128 * 256 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        256 * 128 * 3 * 3,
        256,
        3 * 256 * 3 * 3,
        3};

    float* weights[] = {w1, b1, w2, b2, w3, b3, w4, b4, w5, b5};
    float* gradients[] = {dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5};

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            // Clip gradients to prevent explosion
            float grad = gradients[i][j];
            if (grad > clip_value) grad = clip_value;
            if (grad < -clip_value) grad = -clip_value;

            weights[i][j] -= learning_rate * grad;
        }
    }
}

void Autoencoder::extract_features(
    const float* input,
    float* features,
    int batch_size
) {
    // Run encoder only
    // Conv1: 3->256, 32x32
    conv2d_forward(input, w1, b1, act1, batch_size, 3, 256, 32, 32);
    relu_forward(act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    float* pool1 = new float[batch_size * 256 * 16 * 16];
    maxpool2d_forward(act1, pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16
    conv2d_forward(pool1, w2, b2, act2, batch_size, 256, 128, 16, 16);
    relu_forward(act2, batch_size * 128 * 16 * 16);
    delete[] pool1;

    // MaxPool2 (encoded layer): 16x16->8x8
    maxpool2d_forward(act2, features, batch_size, 128, 16, 16);
}

void Autoencoder::save_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return;

    // Write weights and biases
    fwrite(w1, sizeof(float), 256 * 3 * 3 * 3, f);
    fwrite(b1, sizeof(float), 256, f);
    fwrite(w2, sizeof(float), 128 * 256 * 3 * 3, f);
    fwrite(b2, sizeof(float), 128, f);
    fwrite(w3, sizeof(float), 128 * 128 * 3 * 3, f);
    fwrite(b3, sizeof(float), 128, f);
    fwrite(w4, sizeof(float), 256 * 128 * 3 * 3, f);
    fwrite(b4, sizeof(float), 256, f);
    fwrite(w5, sizeof(float), 3 * 256 * 3 * 3, f);
    fwrite(b5, sizeof(float), 3, f);

    fclose(f);
}

void Autoencoder::load_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return;

    // Read weights and biases with proper error checking
    size_t read_count = 0;
    read_count += fread(w1, sizeof(float), 256 * 3 * 3 * 3, f);
    read_count += fread(b1, sizeof(float), 256, f);
    read_count += fread(w2, sizeof(float), 128 * 256 * 3 * 3, f);
    read_count += fread(b2, sizeof(float), 128, f);
    read_count += fread(w3, sizeof(float), 128 * 128 * 3 * 3, f);
    read_count += fread(b3, sizeof(float), 128, f);
    read_count += fread(w4, sizeof(float), 256 * 128 * 3 * 3, f);
    read_count += fread(b4, sizeof(float), 256, f);
    read_count += fread(w5, sizeof(float), 3 * 256 * 3 * 3, f);
    read_count += fread(b5, sizeof(float), 3, f);
    
    if (read_count == 0) {
        fprintf(stderr, "Warning: No data read from weights file\n");
    }

    fclose(f);
}
