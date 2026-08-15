#include "cct_modules.h" 

template<int TOKENS, int ORIG_DIM, int INNER_DIM>
template<int TOKENS, int ORIG_DIM, int INNER_DIM>
void mlp_impl(
    const float *X,
    const float *W1,
    const float *W2,
    float *out)
{
    float inner_layer[TOKENS * INNER_DIM];
    float inner_layer_gelu[TOKENS * INNER_DIM];
    linear_impl<TOKENS, ORIG_DIM, INNER_DIM>(
        X,
        W1,
        inner_layer
    );
    gelu_impl<TOKENS, INNER_DIM>(
        inner_layer,
        inner_layer_gelu
    );
    linear_impl<TOKENS, INNER_DIM, ORIG_DIM>(
        inner_layer_gelu,
        W2,
        out
    );
}

extern "C" {
    void mlp(const float *X, const float *W1, const float W2, float *out) {
        #pragma HLS INTERFACE m_axi port=X   bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave
        #pragma HLS INTERFACE m_axi port=W1   bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=W2 bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=X   bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=W1   bundle=control
        #pragma HLS INTERFACE s_axilite port=W2 bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        mlp_impl<8, 8, 12>(X, W1, W2, out);
    }
}