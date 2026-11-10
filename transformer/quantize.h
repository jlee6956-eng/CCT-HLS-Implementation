#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <cstdint>
#include <cmath>

// CPU side only: run once before training.
// Converts FP32 weights to INT8. Returns the scale so that w ≈ scale * q
template<int SIZE>
float quantize_int8(const float *w, int8_t *q) {
    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; i++)
        if (fabsf(w[i]) > max_abs) max_abs = fabsf(w[i]);

    float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;

    for (int i = 0; i < SIZE; i++)
        q[i] = (int8_t)lroundf(w[i] / scale);

    return scale;
}

#endif