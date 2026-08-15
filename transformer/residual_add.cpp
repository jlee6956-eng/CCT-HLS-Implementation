#include "cct_modules.h"

template<int TOKENS, int DIM>
void residual_add_impl(const float *X, const float *MX, float *out) {
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            int index = i * DIM + j;
            out[index] = X[index] + MX[index];
        }
    }
}

extern "C" {
    void residual_add(const float *X, const float *MX, float *out) {
        #pragma HLS INTERFACE m_axi port=X   bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=MX bundle=gmem1 offset=slave
        #pragma HLS INTERFACE m_axi port=out   bundle=gmem2 offset=slave
        #pragma HLS INTERFACE s_axilite port=X   bundle=control
        #pragma HLS INTERFACE s_axilite port=MX bundle=control
        #pragma HLS INTERFACE s_axilite port=out   bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        residual_add_impl<8, 8>(X, MX, out);
    }
}