#include <iostream>

template<
    int IN_CH,
    int OUT_CH,
    int IN_H,
    int IN_W
>
void conv_2d_impl(
    const float *input,
    const float *weights,
    float *output
);

int main()
{
    const int INPUT_SIZE =
        3 * 32 * 32;

    const int WEIGHT_SIZE =
        64 * 3 * 3 * 3;

    const int OUTPUT_SIZE =
        64 * 16 * 16;

    static float input[INPUT_SIZE];
    static float weights[WEIGHT_SIZE];
    static float output[OUTPUT_SIZE];

    // Fill input with ones
    for (int i = 0; i < INPUT_SIZE / 2; i++) {
        input[i] = 1.0f;
    }
    for (int j = INPUT_SIZE / 2; j < INPUT_SIZE; j++) {
        input[j] = 2.0f;
    }

    // Fill weights with ones
    for (int i = 0; i < WEIGHT_SIZE / 2; i++) {
        weights[i] = 1.0f;
    }
    for (int j = WEIGHT_SIZE / 2; j < WEIGHT_SIZE; j++) {
        weights[j] = 2.0f;
    }

    conv_2d_impl<3, 64, 32, 32>(input, weights, output);

std::cout << "Top-left: " << output[0] << std::endl;

int interior_idx = 0 * 16 * 16 + 1 * 16 + 1;

std::cout << "Interior: "
          << output[interior_idx]
          << std::endl;
}