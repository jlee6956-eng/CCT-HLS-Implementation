#include <cmath>
#include <ap_int.h>

template<int ROWS, int COLS>
void layer_norm_impl(const float *in, float *out) {
    const int N = ROWS * COLS;
    const float EPS = 1e-5f;

    float sum = 0.0f;
    float mean = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += in[i];
    }

    mean = sum / (float)N;

    for (int i = 0; i < N; i++) {
        float diff = in[i] - mean;
        variance += diff * diff;
    }

    variance /= (float)N;

    float denom = std::sqrt(variance + EPS);

    for (int i = 0; i < N; i++) {
        out[i] = (in[i] - mean) / denom;
    }
}

extern "C" {
void layer_norm(const float *in, float *out) {
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave

#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    layer_norm_impl<8, 8>(in, out);
}
}