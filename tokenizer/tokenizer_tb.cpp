#include <cmath>
#include <iostream>

void tokenizer(
    const float *in,
    const float *conv_1_weights,
    const float *conv_2_weights,
    float *out
);

int main()
{
    const int INPUT_SIZE = 3 * 32 * 32;
    const int CONV1_WEIGHT_SIZE = 64 * 3 * 3 * 3;
    const int CONV2_WEIGHT_SIZE = 256 * 64 * 3 * 3;
    const int OUTPUT_SIZE = 4 * 256;

    static float input[INPUT_SIZE];
    static float conv1_weights[CONV1_WEIGHT_SIZE];
    static float conv2_weights[CONV2_WEIGHT_SIZE];
    static float output[OUTPUT_SIZE];

    // Input = 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1.0f;
    }

    // Conv1 weights = 1
    for (int i = 0; i < CONV1_WEIGHT_SIZE; i++) {
        conv1_weights[i] = 1.0f;
    }

    // Conv2 weights = 1
    for (int i = 0; i < CONV2_WEIGHT_SIZE; i++) {
        conv2_weights[i] = 1.0f;
    }

    tokenizer(
        input,
        conv1_weights,
        conv2_weights,
        output
    );

    std::cout << "First 20 outputs:\n";

    for (int i = 0; i < 20; i++) {
        std::cout << output[i] << "\n";
    }

    // Check for NaN / infinity
    bool valid = true;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (!std::isfinite(output[i])) {
            valid = false;
            break;
        }
    }

    if (valid) {
        std::cout << "TEST PASSED: all outputs are finite\n";
    }
    else {
        std::cout << "TEST FAILED: invalid output detected\n";
    }

    return 0;
}