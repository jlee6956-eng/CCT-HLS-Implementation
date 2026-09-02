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

    // LoRA matrices for Q
    const float *lora_AQ,
    const float *lora_BQ,

    // LoRA matrices for V
    const float *lora_AV,
    const float *lora_BV,

    float lora_alpha,

    const float *W1,
    const float *W2,

    float *out
) {
    float layer1[TOKENS * EMBED_DIM];
    float layer2[TOKENS * EMBED_DIM];
    float layer3[TOKENS * EMBED_DIM];
    float layer4[TOKENS * EMBED_DIM];
    float layer5[TOKENS * EMBED_DIM];

    // 1. LayerNorm
    layer_norm_impl<TOKENS, EMBED_DIM>(
        X,
        layer1
    );

    // 2. Multi-head self-attention with LoRA on Q and V
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

    // 3. First residual connection
    residual_add_impl<TOKENS, EMBED_DIM>(
        X,
        layer2,
        layer3
    );

    // 4. Second LayerNorm
    layer_norm_impl<TOKENS, EMBED_DIM>(
        layer3,
        layer4
    );

    // 5. MLP
    mlp_impl<TOKENS, EMBED_DIM, MLP_DIM>(
        layer4,
        W1,
        W2,
        layer5
    );

    // 6. Second residual connection
    residual_add_impl<TOKENS, EMBED_DIM>(
        layer3,
        layer5,
        out
    );
}