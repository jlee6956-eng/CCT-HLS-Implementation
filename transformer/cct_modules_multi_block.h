#pragma once
#include "cct_modules_backward.h"

// ============================================================
// NUM_BLOCKS LoRA transformer blocks in sequence.
// Every per-block buffer (weights, LoRA matrices, caches,
// gradients) holds all blocks back to back: block b's data
// starts at  ptr + b * (size of one block's buffer).
// ============================================================

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS,
         int LORA_RANK, int NUM_BLOCKS>
void cct_lora_forward_train_impl(
    const float *X,                       // [T*D] network input
    const float *WQ, const float *WK, const float *WV, const float *WO,  // [B*D*D]
    const float *lora_AQ, const float *lora_BQ,   // [B*D*R], [B*R*D]
    const float *lora_AV, const float *lora_BV,
    float lora_alpha,
    const float *W1, const float *W2,     // [B*D*MLP], [B*MLP*D]
    float *out,                           // [T*D] output of the last block
    float *block_in_cache,                // [B*T*D] each block's input
    float *layer1_cache,                  // [B*T*D]
    float *Q_cache, float *K_cache, float *V_cache,  // [B*T*D]
    float *probs_cache,                   // [B*H*T*T]
    float *layer3_cache, float *layer4_cache,        // [B*T*D]
    float *inner_cache)                   // [B*T*MLP]
{
    const int TD  = TOKENS * EMBED_DIM;
    const int DD  = EMBED_DIM * EMBED_DIM;
    const int DR  = EMBED_DIM * LORA_RANK;
    const int DM  = EMBED_DIM * MLP_DIM;
    const int TM  = TOKENS * MLP_DIM;
    const int HTT = NUM_HEADS * TOKENS * TOKENS;

    float x[TD];          // current block's input
    float y[TD];          // current block's output

    for (int i = 0; i < TD; i++) x[i] = X[i];

    for (int b = 0; b < NUM_BLOCKS; b++) {
        // save this block's input: its backward pass needs it (LN1 backward)
        for (int i = 0; i < TD; i++) block_in_cache[b * TD + i] = x[i];

        transformer_block_lora_forward_train_impl<
            TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
            x,
            WQ + b * DD, WK + b * DD, WV + b * DD, WO + b * DD,
            lora_AQ + b * DR, lora_BQ + b * DR,
            lora_AV + b * DR, lora_BV + b * DR,
            lora_alpha,
            W1 + b * DM, W2 + b * DM,
            y,
            layer1_cache + b * TD,
            Q_cache + b * TD, K_cache + b * TD, V_cache + b * TD,
            probs_cache + b * HTT,
            layer3_cache + b * TD, layer4_cache + b * TD,
            inner_cache + b * TM);

        // this block's output is the next block's input
        for (int i = 0; i < TD; i++) x[i] = y[i];
    }

    for (int i = 0; i < TD; i++) out[i] = x[i];
}

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS,
         int LORA_RANK, int NUM_BLOCKS>
void cct_lora_backward_impl(
    const float *WQ, const float *WK, const float *WV, const float *WO,
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
    const float *dOut,                    // [T*D] gradient at the last block's output
    float *dX,                            // [T*D] gradient at block n_freeze's input
    float *dAQ, float *dBQ,               // [B*D*R], [B*R*D]
    float *dAV, float *dBV,
    int n_freeze,                         // freeze blocks 0 .. n_freeze-1 (0 = train all)
    float *info)                          // [B] sum of squared LoRA gradients per block
{
    const int TD  = TOKENS * EMBED_DIM;
    const int DD  = EMBED_DIM * EMBED_DIM;
    const int DR  = EMBED_DIM * LORA_RANK;
    const int DM  = EMBED_DIM * MLP_DIM;
    const int TM  = TOKENS * MLP_DIM;
    const int HTT = NUM_HEADS * TOKENS * TOKENS;

    float g[TD];          // gradient at the current block's output
    float g_in[TD];       // gradient at the current block's input

    // keep n_freeze in range [0, NUM_BLOCKS]
    if (n_freeze < 0) n_freeze = 0;
    if (n_freeze > NUM_BLOCKS) n_freeze = NUM_BLOCKS;

    for (int i = 0; i < TD; i++) g[i] = dOut[i];

    // last block first; stop before the frozen blocks.
    // Frozen blocks (0 .. n_freeze-1) are skipped entirely: their
    // dA/dB and info entries are left untouched.
    for (int b = NUM_BLOCKS - 1; b >= n_freeze; b--) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=NUM_BLOCKS
        transformer_block_lora_backward_impl<
            TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
            block_in_cache + b * TD,
            WQ + b * DD, WK + b * DD, WV + b * DD, WO + b * DD,
            lora_AQ + b * DR, lora_BQ + b * DR,
            lora_AV + b * DR, lora_BV + b * DR,
            lora_alpha,
            W1 + b * DM, W2 + b * DM,
            layer1_cache + b * TD,
            Q_cache + b * TD, K_cache + b * TD, V_cache + b * TD,
            probs_cache + b * HTT,
            layer3_cache + b * TD, layer4_cache + b * TD,
            inner_cache + b * TM,
            g, g_in,
            dAQ + b * DR, dBQ + b * DR,
            dAV + b * DR, dBV + b * DR);

        // info for the freezing decision: sum of squared LoRA gradients
        float sum = 0.0f;
        for (int i = 0; i < DR; i++) {
#pragma HLS PIPELINE
            float aq = dAQ[b * DR + i], bq = dBQ[b * DR + i];
            float av = dAV[b * DR + i], bv = dBV[b * DR + i];
            sum += aq * aq + bq * bq + av * av + bv * bv;
        }
        info[b] = sum;

        // this block's dX is the previous block's dOut
        for (int i = 0; i < TD; i++) g[i] = g_in[i];
    }

    // gradient at the input of the first unfrozen block
    // (the network input when n_freeze = 0)
    for (int i = 0; i < TD; i++) dX[i] = g[i];
}

// ============================================================
// Backward pass with the freezing decision made in hardware.
//
// State kept in memory between calls (one call = one batch):
//   avg_info[b]  : running average of block b's info
//   batch_count  : number of batches seen so far
//
// Each call:
//   1. Choose n_freeze:
//        - every REFRESH_EVERY batches (including the first),
//          n_freeze = 0 so every block's average gets refreshed;
//        - otherwise, freeze blocks from the input side while
//          their running average is below `cutoff`.
//   2. Run the backward pass with that n_freeze.
//   3. Update the running average of every block that ran.
// ============================================================

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS,
         int LORA_RANK, int NUM_BLOCKS, int REFRESH_EVERY>
void cct_lora_backward_auto_impl(
    const float *WQ, const float *WK, const float *WV, const float *WO,
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
    const float *dOut,
    float *dX,
    float *dAQ, float *dBQ,
    float *dAV, float *dBV,
    float cutoff,          // freeze a block if its average info is below this
    float *avg_info,       // [B] state: running average of each block's info
    int *batch_count,      // [1] state: batches seen so far (start at 0)
    float *info,           // [B] this batch's info (blocks that ran)
    int *n_used)           // [1] the n_freeze chosen for this batch
{
    const float DECAY = 0.9f;     // avg = 0.9 * avg + 0.1 * new info

    int count = *batch_count;

    // ---- 1. choose n_freeze ----
    int n = 0;
    bool refresh = (count % REFRESH_EVERY == 0);
    if (!refresh) {
        for (int b = 0; b < NUM_BLOCKS; b++) {
            // stop at the first block that is still learning
            if (avg_info[b] >= cutoff) break;
            n++;
        }
    }

    // ---- 2. backward pass, skipping blocks 0 .. n-1 ----
    cct_lora_backward_impl<
        TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK, NUM_BLOCKS>(
        WQ, WK, WV, WO,
        lora_AQ, lora_BQ, lora_AV, lora_BV, lora_alpha,
        W1, W2,
        block_in_cache, layer1_cache,
        Q_cache, K_cache, V_cache, probs_cache,
        layer3_cache, layer4_cache, inner_cache,
        dOut, dX, dAQ, dBQ, dAV, dBV,
        n, info);

    // ---- 3. update running averages of the blocks that ran ----
    for (int b = n; b < NUM_BLOCKS; b++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=NUM_BLOCKS
        if (count == 0)
            avg_info[b] = info[b];                    // first batch: start the average
        else
            avg_info[b] = DECAY * avg_info[b] + (1.0f - DECAY) * info[b];
    }

    *batch_count = count + 1;
    *n_used = n;
}