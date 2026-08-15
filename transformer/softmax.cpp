#include <ap_int.h>
#include <cmath>

template<int N>
void softmax_impl(const float *in, float *out) {

    // 1. Find maximum
    float max_val = in[0];

    for (int i = 1; i < N; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }

    // 2. Exponentiate shifted values and compute sum
    float exp_vals[N];
    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        exp_vals[i] = std::exp(in[i] - max_val);
        sum += exp_vals[i];
    }

    // 3. Normalize
    for (int i = 0; i < N; i++) {
        out[i] = exp_vals[i] / sum;
    }
}


extern "C" {
void softmax(const float *in, float *out) {

#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave

#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    softmax_impl<16>(in, out);
}
}