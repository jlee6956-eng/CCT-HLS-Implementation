#include "cct_modules_backward.h"

// ============================================================
// Training-mode forward: identical math to transformer_block, but
// also emits the intermediate activations the backward kernel
// needs (LayerNorm inputs/outputs, per-head Q/K/V, softmax
// probabilities, the pre-WO concatenated head output, and the
// pre-GELU MLP activation).
//
// Normal (non-LoRA) backprop: WQ/WK/WV/WO/W1/W2 are all trainable
// and all receive real weight gradients in the backward kernel
// below -- there are no frozen base weights and no LoRA adapters.
// ============================================================

extern "C" {

void transformer_block_train(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    const float *W1,
    const float *W2,
    float *out,
    float *layer1_cache,
    float *Q_cache,
    float *K_cache,
    float *V_cache,
    float *probs_cache,
    float *concatenated_cache,
    float *layer3_cache,
    float *layer4_cache,
    float *inner_cache)
{
#pragma HLS INTERFACE m_axi port=X                  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=WQ                 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=WK                 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=WV                 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=WO                 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=W1                 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=W2                 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=out                offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=layer1_cache       offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=Q_cache            offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=K_cache            offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=V_cache            offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=probs_cache        offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=concatenated_cache offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=layer3_cache       offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=layer4_cache       offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=inner_cache        offset=slave bundle=gmem16

#pragma HLS INTERFACE s_axilite port=X                  bundle=control
#pragma HLS INTERFACE s_axilite port=WQ                 bundle=control
#pragma HLS INTERFACE s_axilite port=WK                 bundle=control
#pragma HLS INTERFACE s_axilite port=WV                 bundle=control
#pragma HLS INTERFACE s_axilite port=WO                 bundle=control
#pragma HLS INTERFACE s_axilite port=W1                 bundle=control
#pragma HLS INTERFACE s_axilite port=W2                 bundle=control
#pragma HLS INTERFACE s_axilite port=out                bundle=control
#pragma HLS INTERFACE s_axilite port=layer1_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=Q_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=K_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=V_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=probs_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=concatenated_cache bundle=control
#pragma HLS INTERFACE s_axilite port=layer3_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=layer4_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=inner_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    const int TOKENS = 4;
    const int EMBED_DIM = 256;
    const int MLP_DIM = 512;
    const int NUM_HEADS = 4;

    transformer_block_forward_train_impl<
        TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS
    >(
        X, WQ, WK, WV, WO,
        W1, W2,
        out,
        layer1_cache, Q_cache, K_cache, V_cache, probs_cache,
        concatenated_cache, layer3_cache, layer4_cache, inner_cache
    );
}

// ============================================================
// Backward: consumes dOut (gradient wrt the block's output) and
// the caches written by transformer_block_train, and produces dX
// (to hand to the previous block) plus real weight gradients for
// WQ/WK/WV/WO/W1/W2 -- standard backprop through every weight
// matrix in the block, no LoRA involved.
// ============================================================

void transformer_block_backward(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    const float *W1,
    const float *W2,
    const float *layer1_cache,
    const float *Q_cache,
    const float *K_cache,
    const float *V_cache,
    const float *probs_cache,
    const float *concatenated_cache,
    const float *layer3_cache,
    const float *layer4_cache,
    const float *inner_cache,
    const float *dOut,
    float *dX,
    float *dWQ,
    float *dWK,
    float *dWV,
    float *dWO,
    float *dW1,
    float *dW2)
{
#pragma HLS INTERFACE m_axi port=X                  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=WQ                 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=WK                 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=WV                 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=WO                 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=W1                 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=W2                 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=layer1_cache       offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=Q_cache            offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=K_cache            offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=V_cache            offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=probs_cache        offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=concatenated_cache offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=layer3_cache       offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=layer4_cache       offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=inner_cache        offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=dOut               offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=dX                 offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=dWQ                offset=slave bundle=gmem18
#pragma HLS INTERFACE m_axi port=dWK                offset=slave bundle=gmem19
#pragma HLS INTERFACE m_axi port=dWV                offset=slave bundle=gmem20
#pragma HLS INTERFACE m_axi port=dWO                offset=slave bundle=gmem21
#pragma HLS INTERFACE m_axi port=dW1                offset=slave bundle=gmem22
#pragma HLS INTERFACE m_axi port=dW2                offset=slave bundle=gmem23

#pragma HLS INTERFACE s_axilite port=X                  bundle=control
#pragma HLS INTERFACE s_axilite port=WQ                 bundle=control
#pragma HLS INTERFACE s_axilite port=WK                 bundle=control
#pragma HLS INTERFACE s_axilite port=WV                 bundle=control
#pragma HLS INTERFACE s_axilite port=WO                 bundle=control
#pragma HLS INTERFACE s_axilite port=W1                 bundle=control
#pragma HLS INTERFACE s_axilite port=W2                 bundle=control
#pragma HLS INTERFACE s_axilite port=layer1_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=Q_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=K_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=V_cache            bundle=control
#pragma HLS INTERFACE s_axilite port=probs_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=concatenated_cache bundle=control
#pragma HLS INTERFACE s_axilite port=layer3_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=layer4_cache       bundle=control
#pragma HLS INTERFACE s_axilite port=inner_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=dOut               bundle=control
#pragma HLS INTERFACE s_axilite port=dX                 bundle=control
#pragma HLS INTERFACE s_axilite port=dWQ                bundle=control
#pragma HLS INTERFACE s_axilite port=dWK                bundle=control
#pragma HLS INTERFACE s_axilite port=dWV                bundle=control
#pragma HLS INTERFACE s_axilite port=dWO                bundle=control
#pragma HLS INTERFACE s_axilite port=dW1                bundle=control
#pragma HLS INTERFACE s_axilite port=dW2                bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    const int TOKENS = 4;
    const int EMBED_DIM = 256;
    const int MLP_DIM = 512;
    const int NUM_HEADS = 4;

    transformer_block_backward_impl<
        TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS
    >(
        X, WQ, WK, WV, WO,
        W1, W2,
        layer1_cache, Q_cache, K_cache, V_cache, probs_cache,
        concatenated_cache, layer3_cache, layer4_cache, inner_cache,
        dOut,
        dX, dWQ, dWK, dWV, dWO, dW1, dW2
    );
}

}
