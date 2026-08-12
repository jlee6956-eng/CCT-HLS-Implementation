#include <iostream>
#include "cct_modules.h"

template<int TOKENS, int DIM>
void attention_value_impl(const float *in, const float *V, float *out) {
    gemm<TOKENS, TOKENS, DIM>(in, V, out);
}

extern "C" {
    void attention_value(const float *attention_weights, const float *V, float *out) {
        #pragma HLS INTERFACE m_axi port=attention_weights bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=V bundle=gmemm2 offset=slave
        #pragma HLS INTEFACE s_axilite port=attention_weights bundle=control
        #pragma HLS INTEFACE s_axilite port=out bundle=control
        #pragma HLS INTEFACE s_axilite port=V bundle=control
        #pragma HLS INTEFACE s_axilite port=return bundle=control

        attention_value_impl<8, 8, 8>(attention_weights, V, out);
    }
}