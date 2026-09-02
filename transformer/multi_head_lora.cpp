#include "cct_modules.h"

// Computes XW + (alpha / RANK)(XA)B.
template<int TOKENS, int DIM, int RANK>
void lora_project_impl(
    const float *X,
    const float *W,
    const float *A,
    const float *B,
    float alpha,
    float *out)
{
    float base[TOKENS * DIM];
    float XA[TOKENS * RANK];
    float update[TOKENS * DIM];

    gemm_impl<TOKENS, DIM, DIM>(X, W, base);
    gemm_impl<TOKENS, DIM, RANK>(X, A, XA);
    gemm_impl<TOKENS, RANK, DIM>(XA, B, update);

    const float scale = alpha / (float)RANK;

    for (int i = 0; i < TOKENS * DIM; i++) {
#pragma HLS PIPELINE II=1
        out[i] = base[i] + scale * update[i];
    }
}

// Multi-head self-attention with LoRA applied to Q and V.
template<int TOKENS, int EMBED_DIM, int NUM_HEADS, int LORA_RANK>
void multi_head_lora_impl(
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
    float *out)
{
    static_assert(NUM_HEADS > 0, "NUM_HEADS must be positive");
    static_assert(LORA_RANK > 0, "LORA_RANK must be positive");
    static_assert(EMBED_DIM % NUM_HEADS == 0,
                  "EMBED_DIM must be divisible by NUM_HEADS");

    const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

    float Q[TOKENS * EMBED_DIM];
    float K[TOKENS * EMBED_DIM];
    float V[TOKENS * EMBED_DIM];
    float concatenated[TOKENS * EMBED_DIM];

    float Q_head[TOKENS * HEAD_DIM];
    float K_head[TOKENS * HEAD_DIM];
    float V_head[TOKENS * HEAD_DIM];

    float scores[TOKENS * TOKENS];
    float probabilities[TOKENS * TOKENS];
    float head_output[TOKENS * HEAD_DIM];

    // Q = XWQ + (alpha / rank)(X AQ)BQ
    lora_project_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WQ, lora_AQ, lora_BQ, lora_alpha, Q);

    // K is unchanged by LoRA.
    gemm_impl<TOKENS, EMBED_DIM, EMBED_DIM>(X, WK, K);

    // V = XWV + (alpha / rank)(X AV)BV
    lora_project_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WV, lora_AV, lora_BV, lora_alpha, V);

    for (int head = 0; head < NUM_HEADS; head++) {
        // Extract this head's Q, K, and V values.
        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                const int full_index =
                    token * EMBED_DIM + head * HEAD_DIM + d;
                const int head_index = token * HEAD_DIM + d;

                Q_head[head_index] = Q[full_index];
                K_head[head_index] = K[full_index];
                V_head[head_index] = V[full_index];
            }
        }

        // scores = QK^T / sqrt(HEAD_DIM)
        attention_scores_impl<TOKENS, HEAD_DIM>(
            Q_head, K_head, scores);

        // probabilities = softmax(scores)
        attention_softmax_impl<TOKENS>(
            scores, probabilities);

        // head_output = probabilities * V
        attention_value_impl<TOKENS, HEAD_DIM>(
            probabilities, V_head, head_output);

        // Put this head into the concatenated output.
        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                const int head_index = token * HEAD_DIM + d;
                const int full_index =
                    token * EMBED_DIM + head * HEAD_DIM + d;

                concatenated[full_index] = head_output[head_index];
            }
        }
    }

    // Final attention output projection.
    linear_impl<TOKENS, EMBED_DIM, EMBED_DIM>(
        concatenated, WO, out);
}

extern "C" {

void multi_head_lora(
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
#pragma HLS INTERFACE m_axi port=out     offset=slave bundle=gmem9

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
#pragma HLS INTERFACE s_axilite port=out        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    const int TOKENS = 4;
    const int EMBED_DIM = 256;
    const int NUM_HEADS = 4;
    const int LORA_RANK = 4;

    multi_head_lora_impl<
        TOKENS,
        EMBED_DIM,
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
        out
    );
}

}
