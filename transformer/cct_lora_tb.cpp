// Testbench for cct_lora_train / cct_lora_backward (NUM_BLOCKS blocks).
//
// Check 1: running the multi-block kernels must give exactly the same
// results as calling the single-block functions by hand, one block
// after another (block 0 -> block 1 forward, block 1 -> block 0 backward).
// Check 2: with n_freeze = 1, block 1 still matches, and block 0's
// gradients are left untouched (its backward pass was skipped).
// Check 3: the top-level kernel chooses n_freeze itself from the
// running averages; check the n it picks over several batches.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cct_modules_multiblock.h"

#define TOKENS      4
#define EMBED_DIM   256
#define MLP_DIM     512
#define NUM_HEADS   4
#define LORA_RANK   4
#define NUM_BLOCKS  2
#define REFRESH_EVERY 10

extern "C" {
void cct_lora_train(
    const float *X, const float *WQ, const float *WK, const float *WV, const float *WO,
    const float *lora_AQ, const float *lora_BQ, const float *lora_AV, const float *lora_BV,
    float lora_alpha, const float *W1, const float *W2, float *out,
    float *block_in_cache, float *layer1_cache,
    float *Q_cache, float *K_cache, float *V_cache, float *probs_cache,
    float *layer3_cache, float *layer4_cache, float *inner_cache);

void cct_lora_backward(
    const float *WQ, const float *WK, const float *WV, const float *WO,
    const float *lora_AQ, const float *lora_BQ, const float *lora_AV, const float *lora_BV,
    float lora_alpha, const float *W1, const float *W2,
    const float *block_in_cache, const float *layer1_cache,
    const float *Q_cache, const float *K_cache, const float *V_cache, const float *probs_cache,
    const float *layer3_cache, const float *layer4_cache, const float *inner_cache,
    const float *dOut, float *dX, float *dAQ, float *dBQ, float *dAV, float *dBV,
    float cutoff, float *avg_info, int *batch_count, float *info, int *n_used);
}

// the multi-block backward with a fixed n_freeze (tests 1 and 2)
void backward_fixed_n(int n_freeze, float *info);

// sizes of one block's piece of each buffer
const int TD  = TOKENS * EMBED_DIM;
const int DD  = EMBED_DIM * EMBED_DIM;
const int DR  = EMBED_DIM * LORA_RANK;
const int DM  = EMBED_DIM * MLP_DIM;
const int TM  = TOKENS * MLP_DIM;
const int HTT = NUM_HEADS * TOKENS * TOKENS;
const int B   = NUM_BLOCKS;

// static so the large arrays don't overflow the stack
static float X[TD], dOut[TD];
static float WQ[B*DD], WK[B*DD], WV[B*DD], WO[B*DD];
static float AQ[B*DR], BQ[B*DR], AV[B*DR], BV[B*DR];
static float W1[B*DM], W2[B*DM];

// outputs of the multi-block kernels
static float out[TD], dX[TD];
static float block_in[B*TD], l1[B*TD], Qc[B*TD], Kc[B*TD], Vc[B*TD];
static float probs[B*HTT], l3[B*TD], l4[B*TD], inner[B*TM];
static float dAQ[B*DR], dBQ[B*DR], dAV[B*DR], dBV[B*DR];
static float info[B];

// reference results (single block, called by hand)
static float r_mid[TD], r_out[TD], r_g1[TD], r_dX[TD];
static float r_l1[B*TD], r_Qc[B*TD], r_Kc[B*TD], r_Vc[B*TD];
static float r_probs[B*HTT], r_l3[B*TD], r_l4[B*TD], r_inner[B*TM];
static float r_dAQ[B*DR], r_dBQ[B*DR], r_dAV[B*DR], r_dBV[B*DR];

void fill(float *a, int n, float scale) {
    for (int i = 0; i < n; i++)
        a[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

// largest difference between two arrays
float max_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int check(const char *name, const float *a, const float *b, int n) {
    float d = max_diff(a, b, n);
    printf("  %-8s max diff = %g  %s\n", name, d, d == 0.0f ? "PASS" : "FAIL");
    return d == 0.0f ? 0 : 1;
}

const float ALPHA = 8.0f;

void backward_fixed_n(int n_freeze, float *info_out) {
    cct_lora_backward_impl<TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK, NUM_BLOCKS>(
        WQ, WK, WV, WO, AQ, BQ, AV, BV, ALPHA, W1, W2,
        block_in, l1, Qc, Kc, Vc, probs, l3, l4, inner,
        dOut, dX, dAQ, dBQ, dAV, dBV, n_freeze, info_out);
}

// Test 3 helper: run the hardware-decision kernel for several batches
// and check the n it chooses each batch.
int run_auto(const char *name, float cutoff, const int *expected_n, int batches) {
    float avg_info[B];
    float info_b[B];
    int batch_count = 0, n_used = -1;
    int errors = 0;
    printf("Auto freezing, %s (cutoff = %g):\n  n chosen per batch:", name, cutoff);
    for (int t = 0; t < batches; t++) {
        cct_lora_backward(WQ, WK, WV, WO, AQ, BQ, AV, BV, ALPHA, W1, W2,
                          block_in, l1, Qc, Kc, Vc, probs, l3, l4, inner,
                          dOut, dX, dAQ, dBQ, dAV, dBV,
                          cutoff, avg_info, &batch_count, info_b, &n_used);
        printf(" %d", n_used);
        if (n_used != expected_n[t]) errors++;
    }
    printf("\n  avg_info = [%g, %g], batch_count = %d  %s\n",
           avg_info[0], avg_info[1], batch_count, errors == 0 ? "PASS" : "FAIL");
    return errors;
}

int main() {
    srand(42);
    const float alpha = ALPHA;

    // ---- random inputs and weights ----
    fill(X, TD, 1.0f);
    fill(dOut, TD, 1.0f);
    fill(WQ, B*DD, 0.06f); fill(WK, B*DD, 0.06f);
    fill(WV, B*DD, 0.06f); fill(WO, B*DD, 0.06f);
    fill(AQ, B*DR, 0.06f); fill(BQ, B*DR, 0.06f);
    fill(AV, B*DR, 0.06f); fill(BV, B*DR, 0.06f);
    fill(W1, B*DM, 0.06f); fill(W2, B*DM, 0.04f);

    // ---- run the multi-block kernels ----
    cct_lora_train(X, WQ, WK, WV, WO, AQ, BQ, AV, BV, alpha, W1, W2, out,
                   block_in, l1, Qc, Kc, Vc, probs, l3, l4, inner);

    // ---- reference: forward, block 0 then block 1 ----
    // block 0: X -> r_mid
    transformer_block_lora_forward_train_impl<TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
        X, WQ, WK, WV, WO, AQ, BQ, AV, BV, alpha, W1, W2, r_mid,
        r_l1, r_Qc, r_Kc, r_Vc, r_probs, r_l3, r_l4, r_inner);
    // block 1: r_mid -> r_out  (block 1's data starts at offset 1 * size)
    transformer_block_lora_forward_train_impl<TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
        r_mid, WQ + DD, WK + DD, WV + DD, WO + DD, AQ + DR, BQ + DR, AV + DR, BV + DR,
        alpha, W1 + DM, W2 + DM, r_out,
        r_l1 + TD, r_Qc + TD, r_Kc + TD, r_Vc + TD, r_probs + HTT,
        r_l3 + TD, r_l4 + TD, r_inner + TM);

    // ---- reference: backward, block 1 then block 0 ----
    // block 1: dOut -> r_g1  (block 1's input was r_mid)
    transformer_block_lora_backward_impl<TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
        r_mid, WQ + DD, WK + DD, WV + DD, WO + DD, AQ + DR, BQ + DR, AV + DR, BV + DR,
        alpha, W1 + DM, W2 + DM,
        r_l1 + TD, r_Qc + TD, r_Kc + TD, r_Vc + TD, r_probs + HTT,
        r_l3 + TD, r_l4 + TD, r_inner + TM,
        dOut, r_g1, r_dAQ + DR, r_dBQ + DR, r_dAV + DR, r_dBV + DR);
    // block 0: r_g1 -> r_dX  (block 0's input was X)
    transformer_block_lora_backward_impl<TOKENS, EMBED_DIM, MLP_DIM, NUM_HEADS, LORA_RANK>(
        X, WQ, WK, WV, WO, AQ, BQ, AV, BV, alpha, W1, W2,
        r_l1, r_Qc, r_Kc, r_Vc, r_probs, r_l3, r_l4, r_inner,
        r_g1, r_dX, r_dAQ, r_dBQ, r_dAV, r_dBV);

    // reference info: sum of squared LoRA gradients for block b
    float r_info[B];
    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        for (int i = 0; i < DR; i++)
            sum += r_dAQ[b*DR+i]*r_dAQ[b*DR+i] + r_dBQ[b*DR+i]*r_dBQ[b*DR+i]
                 + r_dAV[b*DR+i]*r_dAV[b*DR+i] + r_dBV[b*DR+i]*r_dBV[b*DR+i];
        r_info[b] = sum;
    }

    int errors = 0;
    printf("Forward:\n");
    errors += check("out", out, r_out, TD);

    // ---- Test 1: n_freeze = 0, every block trains ----
    backward_fixed_n(0, info);
    printf("Backward, n_freeze = 0 (train all blocks):\n");
    errors += check("dX",   dX,   r_dX,   TD);
    errors += check("dAQ",  dAQ,  r_dAQ,  B*DR);
    errors += check("dBQ",  dBQ,  r_dBQ,  B*DR);
    errors += check("dAV",  dAV,  r_dAV,  B*DR);
    errors += check("dBV",  dBV,  r_dBV,  B*DR);
    errors += check("info", info, r_info, B);
    for (int b = 0; b < B; b++) printf("    info[%d] = %g\n", b, info[b]);

    // ---- Test 2: n_freeze = 1, block 0 frozen ----
    // Fill outputs with a marker value so we can see block 0 was skipped.
    const float MARK = 12345.0f;
    for (int i = 0; i < B*DR; i++) dAQ[i] = dBQ[i] = dAV[i] = dBV[i] = MARK;
    for (int b = 0; b < B; b++) info[b] = MARK;

    backward_fixed_n(1, info);
    printf("Backward, n_freeze = 1 (block 0 frozen):\n");
    // block 1 trained: must match the reference
    errors += check("dAQ[1]", dAQ + DR, r_dAQ + DR, DR);
    errors += check("dBQ[1]", dBQ + DR, r_dBQ + DR, DR);
    errors += check("dAV[1]", dAV + DR, r_dAV + DR, DR);
    errors += check("dBV[1]", dBV + DR, r_dBV + DR, DR);
    errors += check("info[1]", info + 1, r_info + 1, 1);
    // dX is now the gradient at block 1's input
    errors += check("dX", dX, r_g1, TD);
    // block 0 skipped: its outputs must still hold the marker
    int untouched = (info[0] == MARK);
    for (int i = 0; i < DR; i++)
        untouched &= (dAQ[i] == MARK && dBQ[i] == MARK && dAV[i] == MARK && dBV[i] == MARK);
    printf("  block 0 untouched: %s\n", untouched ? "PASS" : "FAIL");
    errors += untouched ? 0 : 1;

    // ---- Test 3: freezing decided in hardware, over 12 batches ----
    // Same data every batch, so block 0's info (~300) and block 1's (~441)
    // stay constant. Batches 0 and 10 are refresh batches (n = 0).
    const int T3 = 12;
    int exp_mid[T3]  = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};  // freeze block 0 only
    int exp_none[T3] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // never freeze
    int exp_all[T3]  = {0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2};  // freeze both blocks
    errors += run_auto("cutoff between blocks", 370.0f, exp_mid,  T3);
    errors += run_auto("cutoff below both",       1.0f, exp_none, T3);
    errors += run_auto("cutoff above both",     1.0e9f, exp_all,  T3);

    printf(errors == 0 ? "\nTEST PASSED\n" : "\nTEST FAILED\n");
    return errors;   // 0 = pass (Vitis C simulation checks this)
}