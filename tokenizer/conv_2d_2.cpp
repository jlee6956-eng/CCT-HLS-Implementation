#include "tokenizer_modules.h" 

void conv_2d_2_impl(const float *input,
    const float *weights,
    float *output) {
        conv_2d_impl<64,256,8,8>(input, weights, output);
    }

extern "C" {
    void conv_2d(const float *input, const float *weights, float *output) {
        #pragma HLS INTERFACE m_axi port=input bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=weights bundle=gmem1 offset=slave
        #pragma HLS INTERFACE m_axi port=output bundle=gmem2 offset=slave
        #pragma HLS INTERFACE s_axilite port=input bundle=control
        #pragma HLS INTERFACE s_axilite port=weights bundle=control
        #pragma HLS INTERFACE s_axilite port=output bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        conv_2d_2_impl(input, weights, output);
    }
}