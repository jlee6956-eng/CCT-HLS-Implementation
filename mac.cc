#include <ap_int.h>

extern "C" {
    void accumulate (unsigned int *in1, unsigned int *in2, unsigned int *out2) {
        #pragma HLS INTERFACE m_axi port=in1 bundle=aximm1
        #pragma HLS INTERFACE m_axi 
    }
}