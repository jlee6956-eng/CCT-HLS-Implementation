#include <cmath>
#include <ap_int.h>

// Normalizes each row (token) independently across its COLS
// features, matching standard transformer LayerNorm.
template<int ROWS, int COLS>
void layer_norm_impl(const float *in, float *out) {
    const float EPS = 1e-5f;

    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            sum += in[r * COLS + c];
        }
        float mean = sum / (float)COLS;

        float variance = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float diff = in[r * COLS + c] - mean;
            variance += diff * diff;
        }
        variance /= (float)COLS;

        float denom = std::sqrt(variance + EPS);

        for (int c = 0; c < COLS; c++) {
            int idx = r * COLS + c;
            out[idx] = (in[idx] - mean) / denom;
        }
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