#include "cpu/layers.h"

#include <cmath>
#include <cstring>
#include <random>

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
) {
    // 3x3 convolution with padding=1, stride=1
    const int kernel_size = 3;
    const int pad = 1;

    // #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float sum = bias[oc];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = h + kh - pad;
                                int iw = w + kw - pad;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = b * (in_channels * height * width) +
                                                    ic * (height * width) + ih * width +
                                                    iw;
                                    int weight_idx =
                                        oc * (in_channels * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }

                    int output_idx = b * (out_channels * height * width) +
                                     oc * (height * width) + h * width + w;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

void relu_forward(float* data, size_t size) {
    // #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }
}

void maxpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    // 2x2 max pooling with stride=2
    const int pool_size = 2;
    const int out_height = in_height / 2;
    const int out_width = in_width / 2;

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    float max_val = -INFINITY;

                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = h * pool_size + ph;
                            int iw = w * pool_size + pw;
                            int input_idx = b * (channels * in_height * in_width) +
                                            c * (in_height * in_width) + ih * in_width +
                                            iw;
                            max_val = (input[input_idx] > max_val) ? input[input_idx]
                                                                   : max_val;
                        }
                    }

                    int output_idx = b * (channels * out_height * out_width) +
                                     c * (out_height * out_width) + h * out_width + w;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}

void upsample2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    // Nearest neighbor upsampling (scale=2)
    const int out_height = in_height * 2;
    const int out_width = in_width * 2;

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int ih = h / 2;
                    int iw = w / 2;
                    int input_idx = b * (channels * in_height * in_width) +
                                    c * (in_height * in_width) + ih * in_width + iw;
                    int output_idx = b * (channels * out_height * out_width) +
                                     c * (out_height * out_width) + h * out_width + w;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

float mse_loss(const float* output, const float* target, size_t size) {
    float loss = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

void init_weights_xavier(float* weights, int in_channels, int out_channels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (in_channels + out_channels));
    std::uniform_real_distribution<float> dis(-limit, limit);

    int kernel_size = 3 * 3;  // 3x3 kernel
    int total_weights = out_channels * in_channels * kernel_size;

    for (int i = 0; i < total_weights; i++) {
        weights[i] = dis(gen);
    }
}

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
) {
    const int kernel_size = 3;
    const int pad = 1;

    // Initialize gradients to zero
    memset(dL_dinput, 0, batch_size * in_channels * height * width * sizeof(float));
    memset(
        dL_dweights,
        0,
        out_channels * in_channels * kernel_size * kernel_size * sizeof(float)
    );
    memset(dL_dbias, 0, out_channels * sizeof(float));

    // Compute bias gradients
    for (int oc = 0; oc < out_channels; oc++) {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = b * (out_channels * height * width) +
                              oc * (height * width) + h * width + w;
                    grad_sum += dL_doutput[idx];
                }
            }
        }
        dL_dbias[oc] = grad_sum;
    }

    // Compute weight and input gradients
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int out_idx = b * (out_channels * height * width) +
                                  oc * (height * width) + h * width + w;
                    float grad = dL_doutput[out_idx];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = h + kh - pad;
                                int iw = w + kw - pad;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = b * (in_channels * height * width) +
                                                    ic * (height * width) + ih * width +
                                                    iw;
                                    int weight_idx =
                                        oc * (in_channels * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;

                                    // Weight gradient
                                    dL_dweights[weight_idx] += grad * input[input_idx];

                                    // Input gradient
                                    dL_dinput[input_idx] += grad * weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void relu_backward(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    size_t size
) {
    for (size_t i = 0; i < size; i++) {
        // Gradient is 1 if output > 0, else 0
        dL_dinput[i] = (output[i] > 0.0f) ? dL_doutput[i] : 0.0f;
    }
}

void maxpool2d_backward(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int pool_size = 2;
    const int out_height = in_height / 2;
    const int out_width = in_width / 2;

    // Initialize input gradient to zero
    memset(dL_dinput, 0, batch_size * channels * in_height * in_width * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    // Find which input position had the max value
                    float max_val = -INFINITY;
                    int max_ih = 0, max_iw = 0;

                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = h * pool_size + ph;
                            int iw = w * pool_size + pw;
                            int input_idx = b * (channels * in_height * in_width) +
                                            c * (in_height * in_width) + ih * in_width +
                                            iw;
                            if (input[input_idx] > max_val) {
                                max_val = input[input_idx];
                                max_ih = ih;
                                max_iw = iw;
                            }
                        }
                    }

                    // Pass gradient to the max input position
                    int out_idx = b * (channels * out_height * out_width) +
                                  c * (out_height * out_width) + h * out_width + w;
                    int max_input_idx = b * (channels * in_height * in_width) +
                                        c * (in_height * in_width) + max_ih * in_width +
                                        max_iw;
                    dL_dinput[max_input_idx] = dL_doutput[out_idx];
                }
            }
        }
    }
}

void upsample2d_backward(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int out_height = in_height * 2;
    const int out_width = in_width * 2;

    // Initialize input gradient to zero
    memset(dL_dinput, 0, batch_size * channels * in_height * in_width * sizeof(float));

    // Sum gradients from all output positions that came from each input
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int ih = h / 2;
                    int iw = w / 2;
                    int input_idx = b * (channels * in_height * in_width) +
                                    c * (in_height * in_width) + ih * in_width + iw;
                    int output_idx = b * (channels * out_height * out_width) +
                                     c * (out_height * out_width) + h * out_width + w;
                    dL_dinput[input_idx] += dL_doutput[output_idx];
                }
            }
        }
    }
}