#include <ap_int.h>

template<int SIZE>
void mac_impl(const unsigned int *in1,
              const unsigned int *in2,
              unsigned long long *out) {
    unsigned long long sum = 0;

    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE II=1
        sum += (unsigned long long)in1[i] * (unsigned long long)in2[i];
    }

    *out = sum;
}

extern "C" {
void mac(const unsigned int *in1,
         const unsigned int *in2,
         unsigned long long *out) {
#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave

#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=in2 bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    mac_impl<64>(in1, in2, out);
}
}