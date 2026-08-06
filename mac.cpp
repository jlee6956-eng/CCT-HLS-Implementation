#include <ap_int.h>

extern "C" {
void mac(const unsigned int *in1,
         const unsigned int *in2,
         unsigned long long *out,
         int size) {
#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave
#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=in2 bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ARRAY_PARTITION variable=in1
#pragma HLS ARRAY_PARTITION variable=in2

unsigned long long sum = 0;
    for (int i = 0; i < size; i++) {
        #pragma HLS PIEPLINE II=1
        sum += (unsigned long long)in1[i] * (unsigned long long)in2[i];
    }

    *out = sum;
}
}