#include "tokenizer_modules.h"

void relu_2_impl(const float *in, float *out) {

    relu_impl<256, 4, 4>(in, out);

}

extern "C" {

    void relu_2(const float *in, float *out) {

        #pragma HLS INTERFACE m_axi port=in bundle=gmem1 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem0 offset=slave

        #pragma HLS INTERFACE s_axilite port=in bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        relu_2_impl(in, out);
    }

}