#include "cct_modules.h"

template<
    int TOKENS,
    int EMBED_DIM,
    int MLP_DIM,
    int NUM_HEADS,
    int LORA_RANK
>
void transformer_block_impl(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    const float *lora_AQ,
    const float *lora_BQ,
    const float *lora_AV,
    const float *lora_BV,
    float lora_alpha,
    const float *W1,
    const float *W2,
    float *out)
{
    static_assert(NUM_HEADS > 0, "NUM_HEADS must be positive");
    static_assert(LORA_RANK > 0, "LORA_RANK must be positive");
    static_assert(EMBED_DIM % NUM_HEADS == 0,
                  "EMBED_DIM must be divisible by NUM_HEADS");

    float layer1[TOKENS * EMBED_DIM];
    float layer2[TOKENS * EMBED_DIM];
    float layer3[TOKENS * EMBED_DIM];
    float layer4[TOKENS * EMBED_DIM];
    float layer5[TOKENS * EMBED_DIM];

    // Pre-normalization for attention.
    layer_norm_impl<TOKENS, EMBED_DIM>(X, layer1);

    // Multi-head self-attention with LoRA on Q and V.
    multi_head_lora_impl<
        TOKENS,
        EMBED_DIM,
        NUM_HEADS,
        LORA_RANK
    >(
        layer1,
        WQ,
        WK,
        WV,
        WO,
        lora_AQ,
        lora_BQ,
        lora_AV,
        lora_BV,
        lora_alpha,
        layer2
    );

    // First residual connection.
    residual_add_impl<TOKENS, EMBED_DIM>(X, layer2, layer3);

    // Pre-normalization for the MLP.
    layer_norm_impl<TOKENS, EMBED_DIM>(layer3, layer4);

    // Feed-forward network.
    mlp_impl<TOKENS, EMBED_DIM, MLP_DIM>(layer4, W1, W2, layer5);

    // Second residual connection.
    residual_add_impl<TOKENS, EMBED_DIM>(layer3, layer5, out);
}

extern "C" {

void transformer_block_lora(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    const float *lora_AQ,
    const float *lora_BQ,
    const float *lora_AV,
    const float *lora_BV,
    float lora_alpha,
    const float *W1,
    const float *W2,
    float *out)
{
#pragma HLS INTERFACE m_axi port=X       offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=WQ      offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=WK      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=WV      offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=WO      offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=lora_AQ offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=lora_BQ offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=lora_AV offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=lora_BV offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=W1      offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=W2      offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=out     offset=slave bundle=gmem11

#pragma HLS INTERFACE s_axilite port=X          bundle=control
#pragma HLS INTERFACE s_axilite port=WQ         bundle=control
#pragma HLS INTERFACE s_axilite port=WK         bundle=control
#pragma HLS INTERFACE s_axilite port=WV         bundle=control
#pragma HLS INTERFACE s_axilite port=WO         bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AQ    bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BQ    bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AV    bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BV    bundle=control
#pragma HLS INTERFACE s_axilite port=lora_alpha bundle=control
#pragma HLS INTERFACE s_axilite port=W1         bundle=control
#pragma HLS INTERFACE s_axilite port=W2         bundle=control
#pragma HLS INTERFACE s_axilite port=out        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    const int TOKENS = 4;
    const int EMBED_DIM = 256;
    const int MLP_DIM = 512;
    const int NUM_HEADS = 4;
    const int LORA_RANK = 4;

    transformer_block_impl<
        TOKENS,
        EMBED_DIM,
        MLP_DIM,
        NUM_HEADS,
        LORA_RANK
    >(
        X,
        WQ,
        WK,
        WV,
        WO,
        lora_AQ,
        lora_BQ,
        lora_AV,
        lora_BV,
        lora_alpha,
        W1,
        W2,
        out
    );
}

}
