#include "cct_modules.h"

template<int TOKENS, int DIM>
void gelu_impl(const float *X, float *out)
{
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {

            int index = i * DIM + j;

            out[index] = 0.5f * X[index] *
                         (1.0f + std::erf(
                             X[index] / std::sqrt(2.0f)
                         ));
        }
    }
}

extern "C" {

void gelu(const float *X, float *out)
{
#pragma HLS INTERFACE m_axi port=X   bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave

#pragma HLS INTERFACE s_axilite port=X   bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    gelu_impl<8, 8>(X, out);
}

}