#include <io_stream>

template<int ROWS, int COLS>
void gemm_impl(const float *in1, const float *in2, const float *out) {
    
}

extern "C" {
    void gemm(const float *in1, const float *in2, const float *out)
    #pragma HLS INTERFACE port=in1 bundle=gmemm0 offset=slave
    #pragma HLS INTERFACE port=in2 bundle=gmemm1 offset=slave
    #pragma HLS INTERFACE port=out bundle=gmemm2 offset=slave
    #pragma HLS INTERFACE port=in1 bundle=control
    #pragma HLS INTERFACE port=in2 bundle=control
    #pragma HLS INTERFACE port=out bundle=control
    #pragma HLS INTERFACE port=return bundle=control

    
}
