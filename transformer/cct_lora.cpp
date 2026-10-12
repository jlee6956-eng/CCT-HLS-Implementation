#include "cct_modules_multiblock.h"

// ============================================================
// Top-level kernels for NUM_BLOCKS LoRA transformer blocks in
// sequence. Every per-block buffer holds all blocks back to back
// (block b starts at  ptr + b * one block's size).
// ============================================================

#define TOKENS      4
#define EMBED_DIM   256
#define MLP_DIM     512
#define NUM_HEADS   4
#define LORA_RANK   4
#define NUM_BLOCKS  2
#define REFRESH_EVERY 10   // full backward (no freezing) every 10th batch

extern "C" {

// Training-mode forward: runs every block and saves the caches
// the backward kernel needs.
void cct_lora_train(
    const float *X,                                   // [T*D]
    const float *WQ, const float *WK,                 // [B*D*D]
    const float *WV, const float *WO,                 // [B*D*D]
    const float *lora_AQ, const float *lora_BQ,       // [B*D*R], [B*R*D]
    const float *lora_AV, const float *lora_BV,       // [B*D*R], [B*R*D]
    float lora_alpha,
    const float *W1, const float *W2,                 // [B*D*M], [B*M*D]
    float *out,                                       // [T*D]
    float *block_in_cache,                            // [B*T*D]
    float *layer1_cache,                              // [B*T*D]
    float *Q_cache, float *K_cache, float *V_cache,   // [B*T*D]
    float *probs_cache,                               // [B*H*T*T]
    float *layer3_cache, float *layer4_cache,         // [B*T*D]
    float *inner_cache)                               // [B*T*M]
{
#pragma HLS INTERFACE m_axi port=X              offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=WQ             offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=WK             offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=WV             offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=WO             offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=lora_AQ        offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=lora_BQ        offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=lora_AV        offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=lora_BV        offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=W1             offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=W2             offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=out            offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=block_in_cache offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=layer1_cache   offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=Q_cache        offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=K_cache        offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=V_cache        offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=probs_cache    offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=layer3_cache   offset=slave bundle=gmem18
#pragma HLS INTERFACE m_axi port=layer4_cache   offset=slave bundle=gmem19
#pragma HLS INTERFACE m_axi port=inner_cache    offset=slave bundle=gmem20

#pragma HLS INTERFACE s_axilite port=X              bundle=control
#pragma HLS INTERFACE s_axilite port=WQ             bundle=control
#pragma HLS INTERFACE s_axilite port=WK             bundle=control
#pragma HLS INTERFACE s_axilite port=WV             bundle=control
#pragma HLS INTERFACE s_axilite port=WO             bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AQ        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BQ        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AV        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BV        bundle=control
#pragma HLS INTERFACE s_axilite port=W1             bundle=control
#pragma HLS INTERFACE s_axilite port=W2             bundle=control
#pragma HLS INTERFACE s_axilite port=out            bundle=control
#pragma HLS INTERFACE s_axilite port=block_in_cache bundle=control
#pragma HLS INTERFACE s_axilite port=layer1_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=Q_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=K_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=V_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=probs_cache    bundle=control
#pragma HLS INTERFACE s_axilite port=layer3_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=layer4_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=inner_cache    bundle=control
#pragma HLS INTERFACE s_axilite port=lora_alpha     bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    cct_lora_forward_train_impl<
        TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK, NUM_BLOCKS>(
        X, WQ, WK, WV, WO,
        lora_AQ, lora_BQ, lora_AV, lora_BV, lora_alpha,
        W1, W2,
        out,
        block_in_cache, layer1_cache,
        Q_cache, K_cache, V_cache, probs_cache,
        layer3_cache, layer4_cache, inner_cache);
}

// Backward with the freezing decision made in hardware:
// chooses n_freeze from the running averages, skips blocks
// 0 .. n_freeze-1, and updates the averages. One call = one batch.
void cct_lora_backward(
    const float *WQ, const float *WK,
    const float *WV, const float *WO,
    const float *lora_AQ, const float *lora_BQ,
    const float *lora_AV, const float *lora_BV,
    float lora_alpha,
    const float *W1, const float *W2,
    const float *block_in_cache,
    const float *layer1_cache,
    const float *Q_cache, const float *K_cache, const float *V_cache,
    const float *probs_cache,
    const float *layer3_cache, const float *layer4_cache,
    const float *inner_cache,
    const float *dOut,                                // [T*D]
    float *dX,                                        // [T*D]
    float *dAQ, float *dBQ,                           // [B*D*R], [B*R*D]
    float *dAV, float *dBV,                           // [B*D*R], [B*R*D]
    float cutoff,                                     // freeze if avg info < cutoff
    float *avg_info,                                  // [B] state, kept between calls
    int *batch_count,                                 // [1] state, start at 0
    float *info,                                      // [B] this batch's info
    int *n_used)                                      // [1] n_freeze chosen this batch
{
#pragma HLS INTERFACE m_axi port=WQ             offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=WK             offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=WV             offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=WO             offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=lora_AQ        offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=lora_BQ        offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=lora_AV        offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=lora_BV        offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=W1             offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=W2             offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=block_in_cache offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=layer1_cache   offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=Q_cache        offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=K_cache        offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=V_cache        offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=probs_cache    offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=layer3_cache   offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=layer4_cache   offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=inner_cache    offset=slave bundle=gmem18
#pragma HLS INTERFACE m_axi port=dOut           offset=slave bundle=gmem19
#pragma HLS INTERFACE m_axi port=dX             offset=slave bundle=gmem20
#pragma HLS INTERFACE m_axi port=dAQ            offset=slave bundle=gmem21
#pragma HLS INTERFACE m_axi port=dBQ            offset=slave bundle=gmem22
#pragma HLS INTERFACE m_axi port=dAV            offset=slave bundle=gmem23
#pragma HLS INTERFACE m_axi port=dBV            offset=slave bundle=gmem24
#pragma HLS INTERFACE m_axi port=info           offset=slave bundle=gmem25
#pragma HLS INTERFACE m_axi port=avg_info       offset=slave bundle=gmem26
#pragma HLS INTERFACE m_axi port=batch_count    offset=slave bundle=gmem27
#pragma HLS INTERFACE m_axi port=n_used         offset=slave bundle=gmem28

#pragma HLS INTERFACE s_axilite port=WQ             bundle=control
#pragma HLS INTERFACE s_axilite port=WK             bundle=control
#pragma HLS INTERFACE s_axilite port=WV             bundle=control
#pragma HLS INTERFACE s_axilite port=WO             bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AQ        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BQ        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_AV        bundle=control
#pragma HLS INTERFACE s_axilite port=lora_BV        bundle=control
#pragma HLS INTERFACE s_axilite port=W1             bundle=control
#pragma HLS INTERFACE s_axilite port=W2             bundle=control
#pragma HLS INTERFACE s_axilite port=block_in_cache bundle=control
#pragma HLS INTERFACE s_axilite port=layer1_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=Q_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=K_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=V_cache        bundle=control
#pragma HLS INTERFACE s_axilite port=probs_cache    bundle=control
#pragma HLS INTERFACE s_axilite port=layer3_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=layer4_cache   bundle=control
#pragma HLS INTERFACE s_axilite port=inner_cache    bundle=control
#pragma HLS INTERFACE s_axilite port=dOut           bundle=control
#pragma HLS INTERFACE s_axilite port=dX             bundle=control
#pragma HLS INTERFACE s_axilite port=dAQ            bundle=control
#pragma HLS INTERFACE s_axilite port=dBQ            bundle=control
#pragma HLS INTERFACE s_axilite port=dAV            bundle=control
#pragma HLS INTERFACE s_axilite port=dBV            bundle=control
#pragma HLS INTERFACE s_axilite port=info           bundle=control
#pragma HLS INTERFACE s_axilite port=avg_info       bundle=control
#pragma HLS INTERFACE s_axilite port=batch_count    bundle=control
#pragma HLS INTERFACE s_axilite port=n_used         bundle=control
#pragma HLS INTERFACE s_axilite port=cutoff         bundle=control
#pragma HLS INTERFACE s_axilite port=lora_alpha     bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    cct_lora_backward_auto_impl<
        TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK, NUM_BLOCKS,
        REFRESH_EVERY>(
        WQ, WK, WV, WO,
        lora_AQ, lora_BQ, lora_AV, lora_BV, lora_alpha,
        W1, W2,
        block_in_cache, layer1_cache,
        Q_cache, K_cache, V_cache, probs_cache,
        layer3_cache, layer4_cache, inner_cache,
        dOut, dX, dAQ, dBQ, dAV, dBV,
        cutoff, avg_info, batch_count, info, n_used);
}

}