#include "cct_modules.h"

template<int TOKENS, int DIM, int RANK>
void lora_layer_impl(
    const float *in,
    const float *weight,
    const float *lora_a,
    const float *lora_b,
    float alpha,
    float *out
) {
    float base[TOKENS * DIM];
    float XA[TOKENS * RANK];
    float lora_update[TOKENS * DIM];

    // Original layer: XW
    gemm_impl<TOKENS, DIM, DIM>(
        in,
        weight,
        base
    );

    // Compress: XA
    gemm_impl<TOKENS, DIM, RANK>(
        in,
        lora_a,
        XA
    );

    // Expand: (XA)B
    gemm_impl<TOKENS, RANK, DIM>(
        XA,
        lora_b,
        lora_update
    );

    const float scale = alpha / (float)RANK;

    // Final result: XW + (alpha / rank)(XA)B
    for (int i = 0; i < TOKENS * DIM; i++) {
#pragma HLS PIPELINE II=1
        out[i] = base[i] + scale * lora_update[i];
    }
}

extern "C" {

void lora_layer(
    const float *in,
    const float *weight,
    const float *lora_a,
    const float *lora_b,
    float alpha,
    float *out
) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=lora_a bundle=gmem2 offset=slave
#pragma HLS INTERFACE m_axi port=lora_b bundle=gmem3 offset=slave
#pragma HLS INTERFACE m_axi port=out    bundle=gmem4 offset=slave

#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=lora_a bundle=control
#pragma HLS INTERFACE s_axilite port=lora_b bundle=control
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Replace these with your actual dimensions.
    const int TOKENS = 4;
    const int DIM = 256;
    const int RANK = 4;

    lora_layer_impl<TOKENS, DIM, RANK>(
        in,
        weight,
        lora_a,
        lora_b,
        alpha,
        out
    );
}

}