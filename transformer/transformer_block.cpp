#include "cct_modules.h"

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS>
void transformer_block_impl(
    const float *X,

    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,

    const float *W1,
    const float *W2,

    float *out)
{
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

    // 2. MHSA
    multi_head_impl<TOKENS, EMBED_DIM, NUM_HEADS>(
        layer1,
        WQ,
        WK,
        WV,
        WO,
        layer2
    );

    // 3. Residual
    residual_add_impl<TOKENS, EMBED_DIM>(
        X,
        layer2,
        layer3
    );

    // 4. LayerNorm
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

    // 6. Residual
    residual_add_impl<TOKENS, EMBED_DIM>(
        layer3,
        layer5,
        out
    );
}