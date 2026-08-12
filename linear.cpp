#include "cct_modules.h" 


template<int TOKENS, int DIM, int OUT_DIM>
void linear_impl(const float *attention_weights, const float *W, float *next_layer) {
    gemm_impl<TOKENS,TOKENS,DIM>(attention_weights, W, next_layer);
}

extern "C" {
    void linear(const float *attention_weights, const float *W, float *next_layer) {
        #pragma HLS INTERFACE m_axi port=attention_weights bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=W bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=next_layer bundle=gmemm2 offset=slave
        #pragma HLS INTEFACE s_axilite port=attention_weights bundle=control
        #pragma HLS INTEFACE s_axilite port=W bundle=control
        #pragma HLS INTEFACE s_axilite port=next_layer bundle=control
        #pragma HLS INTEFACE s_axilite port=return bundle=control

        linear_impl<8, 8, 8>(attention_weights, W, next_layer); 

    }
}