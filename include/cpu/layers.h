#ifndef LAYERS_H
#define LAYERS_H

#include <cstddef>

// Simple layer operations for CPU implementation

// Convolution 3x3 with padding=1, stride=1
void conv2d_forward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

void conv2d_backward(
    const float* input,
    const float* weights,
    const float* dL_doutput,
    float* dL_dinput,
    float* dL_dweights,
    float* dL_dbias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

// ReLU activation: max(0, x)
void relu_forward(float* data, size_t size);

void relu_backward(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    size_t size
);

// Max pooling 2x2, stride=2
void maxpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void maxpool2d_backward(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Nearest neighbor upsampling (scale=2)
void upsample2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void upsample2d_backward(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Mean squared error loss
float mse_loss(const float* output, const float* target, size_t size);

// Xavier weight initialization
void init_weights_xavier(float* weights, int in_channels, int out_channels);

#endif  // LAYERS_H
