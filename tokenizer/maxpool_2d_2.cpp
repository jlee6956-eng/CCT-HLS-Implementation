#include "tokenizer_modules.h"

void maxpool_2d_2_impl(const float *in, const float *out) {
    maxpool_2d_impl<256, 4, 4>(in, out);
}

extern "C" {
    void maxpool_2d_2(const float *in, float *out) {
        #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave

        #pragma HLS INTERFACE s_axilite port=in bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        maxpool_2d_2_impl(in, out);
    }
}