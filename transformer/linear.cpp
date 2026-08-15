#include "cct_modules.h"

template<int TOKENS, int IN_DIM, int OUT_DIM>
void linear_impl(const float *in,
                 const float *W,
                 float *out)
{
    gemm_impl<TOKENS, IN_DIM, OUT_DIM>(
        in,
        W,
        out
    );
}

extern "C" {

void linear(const float *in,
            const float *W,
            float *out)
{
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=W   bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave

#pragma HLS INTERFACE s_axilite port=in  bundle=control
#pragma HLS INTERFACE s_axilite port=W   bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    linear_impl<8, 8, 8>(in, W, out);
}

}