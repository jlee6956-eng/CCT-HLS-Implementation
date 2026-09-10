#pragma once
#include "cct_modules.h"

// ============================================================
// Backward primitives for LoRA fine-tuning.
//
// Convention: WQ/WK/WV/WO/W1/W2 are frozen base weights, so their
// backward paths only ever produce a "dIn" (gradient continuing the
// chain rule back toward X) and never a weight gradient. The LoRA
// matrices (lora_A*/lora_B*) are the only trainable parameters, so
// their backward paths produce both dIn and the weight gradients
// (dA, dB) needed for the optimizer step.
// ============================================================

// dIn1 = dOut @ in2^T   (In1: MxK, In2: KxN, dOut: MxN, dIn1: MxK)
template<int M, int K, int N>
void gemm_backward_dIn1(const float *dOut, const float *in2, float *dIn1) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += dOut[i * N + j] * in2[k * N + j];
            }
            dIn1[i * K + k] = sum;
        }
    }
}

// dIn2 = in1^T @ dOut   (In1: MxK, In2: KxN, dOut: MxN, dIn2: KxN)
template<int M, int K, int N>
void gemm_backward_dIn2(const float *in1, const float *dOut, float *dIn2) {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += in1[i * K + k] * dOut[i * N + j];
            }
            dIn2[k * N + j] = sum;
        }
    }
}

// Backward of the (no-affine) row-wise standardization used by
// layer_norm_impl. `in` is the layer's cached input, `out` its
// cached output; both are needed by the formula below.
template<int ROWS, int COLS>
void layer_norm_backward_impl(
    const float *in,
    const float *out,
    const float *dOut,
    float *dIn)
{
    const float EPS = 1e-5f;

    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            sum += in[r * COLS + c];
        }
        float mean = sum / (float)COLS;

        float var = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float diff = in[r * COLS + c] - mean;
            var += diff * diff;
        }
        var /= (float)COLS;
        float std_dev = hls::sqrtf(var + EPS);

        float mean_dy = 0.0f;
        float mean_dy_y = 0.0f;
        for (int c = 0; c < COLS; c++) {
            int idx = r * COLS + c;
            mean_dy += dOut[idx];
            mean_dy_y += dOut[idx] * out[idx];
        }
        mean_dy /= (float)COLS;
        mean_dy_y /= (float)COLS;

        for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II=1
            int idx = r * COLS + c;
            dIn[idx] = (dOut[idx] - mean_dy - out[idx] * mean_dy_y) / std_dev;
        }
    }
}

// Backward of a single-row softmax. `out` is the cached softmax
// output for that row.
template<int N>
void softmax_backward_impl(const float *out, const float *dOut, float *dIn) {
    float dot = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += dOut[i] * out[i];
    }
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        dIn[i] = out[i] * (dOut[i] - dot);
    }
}

template<int TOKENS>
void attention_softmax_backward_impl(
    const float *probs,
    const float *dProbs,
    float *dScores)
{
    float prob_row[TOKENS];
    float dprob_row[TOKENS];
    float dscore_row[TOKENS];

    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < TOKENS; j++) {
            prob_row[j] = probs[i * TOKENS + j];
            dprob_row[j] = dProbs[i * TOKENS + j];
        }

        softmax_backward_impl<TOKENS>(prob_row, dprob_row, dscore_row);

        for (int k = 0; k < TOKENS; k++) {
            dScores[i * TOKENS + k] = dscore_row[k];
        }
    }
}

// Backward of scores = (Q @ K^T) / sqrt(DIM).
template<int TOKENS, int DIM>
void attention_scores_backward_impl(
    const float *query,
    const float *key,
    const float *dScore,
    float *dQuery,
    float *dKey)
{
    const float scale = hls::sqrtf((float)DIM);

    float key_T[DIM * TOKENS];
    transpose_impl<TOKENS, DIM>(key, key_T);

    float draw[TOKENS * TOKENS];
    for (int i = 0; i < TOKENS * TOKENS; i++) {
#pragma HLS PIPELINE II=1
        draw[i] = dScore[i] / scale;
    }

    // dQuery = draw @ key_T^T = draw @ key
    gemm_backward_dIn1<TOKENS, DIM, TOKENS>(draw, key_T, dQuery);

    // dKeyT = query^T @ draw, then transpose back to key's layout.
    float dKeyT[DIM * TOKENS];
    gemm_backward_dIn2<TOKENS, DIM, TOKENS>(query, draw, dKeyT);
    transpose_impl<DIM, TOKENS>(dKeyT, dKey);
}

// Backward of head_output = probabilities @ V.
template<int TOKENS, int DIM>
void attention_value_backward_impl(
    const float *probs,
    const float *V,
    const float *dOut,
    float *dProbs,
    float *dV)
{
    gemm_backward_dIn1<TOKENS, TOKENS, DIM>(dOut, V, dProbs);
    gemm_backward_dIn2<TOKENS, TOKENS, DIM>(probs, dOut, dV);
}

// Backward of the erf-based GELU. `X` is the cached pre-activation.
template<int TOKENS, int DIM>
void gelu_backward_impl(const float *X, const float *dOut, float *dIn) {
    const float inv_sqrt_2pi = 0.3989422804014327f;

    for (int i = 0; i < TOKENS * DIM; i++) {
#pragma HLS PIPELINE II=1
        float x = X[i];
        float cdf = 0.5f * (1.0f + hls::erf(x / hls::sqrtf(2.0f)));
        float pdf = inv_sqrt_2pi * hls::expf(-0.5f * x * x);
        dIn[i] = dOut[i] * (cdf + x * pdf);
    }
}

// Backward of out = X@W + (alpha/RANK)(X@A)@B, where W is frozen
// and A, B are the trainable LoRA matrices. XA is recomputed here
// (cheap: RANK is small) rather than cached from the forward pass.
template<int TOKENS, int DIM, int RANK>
void lora_project_backward_impl(
    const float *X,
    const float *W,
    const float *A,
    const float *B,
    float alpha,
    const float *dOut,
    float *dX,
    float *dA,
    float *dB)
{
    const float scale = alpha / (float)RANK;

    float XA[TOKENS * RANK];
    gemm_impl<TOKENS, DIM, RANK>(X, A, XA);

    float dUpdate[TOKENS * DIM];
    for (int i = 0; i < TOKENS * DIM; i++) {
#pragma HLS PIPELINE II=1
        dUpdate[i] = scale * dOut[i];
    }

    float dXA[TOKENS * RANK];
    gemm_backward_dIn1<TOKENS, RANK, DIM>(dUpdate, B, dXA);
    gemm_backward_dIn2<TOKENS, RANK, DIM>(XA, dUpdate, dB);

    float dX_fromA[TOKENS * DIM];
    gemm_backward_dIn1<TOKENS, DIM, RANK>(dXA, A, dX_fromA);
    gemm_backward_dIn2<TOKENS, DIM, RANK>(X, dXA, dA);

    float dX_fromW[TOKENS * DIM];
    gemm_backward_dIn1<TOKENS, DIM, DIM>(dOut, W, dX_fromW);

    for (int i = 0; i < TOKENS * DIM; i++) {
#pragma HLS PIPELINE II=1
        dX[i] = dX_fromW[i] + dX_fromA[i];
    }
}

// Backward of mlp_impl (W1, W2 frozen -> no weight gradients).
// `inner` is the cached pre-GELU activation (X @ W1).
template<int TOKENS, int ORIG_DIM, int INNER_DIM>
void mlp_backward_impl(
    const float *W1,
    const float *W2,
    const float *inner,
    const float *dOut,
    float *dX)
{
    float d_inner_gelu[TOKENS * INNER_DIM];
    gemm_backward_dIn1<TOKENS, INNER_DIM, ORIG_DIM>(dOut, W2, d_inner_gelu);

    float d_inner[TOKENS * INNER_DIM];
    gelu_backward_impl<TOKENS, INNER_DIM>(inner, d_inner_gelu, d_inner);

    gemm_backward_dIn1<TOKENS, ORIG_DIM, INNER_DIM>(d_inner, W1, dX);
}

// ============================================================
// Multi-head attention with LoRA on Q and V: training-mode
// forward (caches what the backward pass needs) and backward.
// ============================================================

template<int TOKENS, int EMBED_DIM, int NUM_HEADS, int LORA_RANK>
void multi_head_lora_forward_train_impl(
    const float *X,
    const float *WQ,
    const float *lora_AQ,
    const float *lora_BQ,
    const float *WK,
    const float *WV,
    const float *lora_AV,
    const float *lora_BV,
    const float *WO,
    float lora_alpha,
    float *out,
    float *Q_cache,
    float *K_cache,
    float *V_cache,
    float *probs_cache)   // [NUM_HEADS * TOKENS * TOKENS]
{
    const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

    float concatenated[TOKENS * EMBED_DIM];

    lora_project_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WQ, lora_AQ, lora_BQ, lora_alpha, Q_cache);

    gemm_impl<TOKENS, EMBED_DIM, EMBED_DIM>(X, WK, K_cache);

    lora_project_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WV, lora_AV, lora_BV, lora_alpha, V_cache);

    for (int head = 0; head < NUM_HEADS; head++) {
        float Q_h[TOKENS * HEAD_DIM];
        float K_h[TOKENS * HEAD_DIM];
        float V_h[TOKENS * HEAD_DIM];
        float scores[TOKENS * TOKENS];
        float head_output[TOKENS * HEAD_DIM];

        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                int full = token * EMBED_DIM + head * HEAD_DIM + d;
                int hidx = token * HEAD_DIM + d;
                Q_h[hidx] = Q_cache[full];
                K_h[hidx] = K_cache[full];
                V_h[hidx] = V_cache[full];
            }
        }

        float *probs_h = probs_cache + head * TOKENS * TOKENS;

        attention_scores_impl<TOKENS, HEAD_DIM>(Q_h, K_h, scores);
        attention_softmax_impl<TOKENS>(scores, probs_h);
        attention_value_impl<TOKENS, HEAD_DIM>(probs_h, V_h, head_output);

        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                int hidx = token * HEAD_DIM + d;
                int full = token * EMBED_DIM + head * HEAD_DIM + d;
                concatenated[full] = head_output[hidx];
            }
        }
    }

    linear_impl<TOKENS, EMBED_DIM, EMBED_DIM>(concatenated, WO, out);
}

template<int TOKENS, int EMBED_DIM, int NUM_HEADS, int LORA_RANK>
void multi_head_lora_backward_impl(
    const float *X,
    const float *WQ,
    const float *lora_AQ,
    const float *lora_BQ,
    const float *WK,
    const float *WV,
    const float *lora_AV,
    const float *lora_BV,
    const float *WO,
    float lora_alpha,
    const float *Q_cache,
    const float *K_cache,
    const float *V_cache,
    const float *probs_cache,
    const float *dOut,        // gradient wrt this module's output
    float *dX,
    float *dAQ, float *dBQ,
    float *dAV, float *dBV)
{
    const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

    // WO is frozen: only need the dIn half of its gemm backward.
    float dConcatenated[TOKENS * EMBED_DIM];
    gemm_backward_dIn1<TOKENS, EMBED_DIM, EMBED_DIM>(dOut, WO, dConcatenated);

    float dQ[TOKENS * EMBED_DIM];
    float dK[TOKENS * EMBED_DIM];
    float dV[TOKENS * EMBED_DIM];

    for (int head = 0; head < NUM_HEADS; head++) {
        float Q_h[TOKENS * HEAD_DIM];
        float K_h[TOKENS * HEAD_DIM];
        float V_h[TOKENS * HEAD_DIM];
        float dHeadOut[TOKENS * HEAD_DIM];

        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                int full = token * EMBED_DIM + head * HEAD_DIM + d;
                int hidx = token * HEAD_DIM + d;
                Q_h[hidx] = Q_cache[full];
                K_h[hidx] = K_cache[full];
                V_h[hidx] = V_cache[full];
                dHeadOut[hidx] = dConcatenated[full];
            }
        }

        const float *probs_h = probs_cache + head * TOKENS * TOKENS;

        float dProbs_h[TOKENS * TOKENS];
        float dScores_h[TOKENS * TOKENS];
        float dQ_h[TOKENS * HEAD_DIM];
        float dK_h[TOKENS * HEAD_DIM];
        float dV_h[TOKENS * HEAD_DIM];

        attention_value_backward_impl<TOKENS, HEAD_DIM>(
            probs_h, V_h, dHeadOut, dProbs_h, dV_h);

        attention_softmax_backward_impl<TOKENS>(probs_h, dProbs_h, dScores_h);

        attention_scores_backward_impl<TOKENS, HEAD_DIM>(
            Q_h, K_h, dScores_h, dQ_h, dK_h);

        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
                int full = token * EMBED_DIM + head * HEAD_DIM + d;
                int hidx = token * HEAD_DIM + d;
                dQ[full] = dQ_h[hidx];
                dK[full] = dK_h[hidx];
                dV[full] = dV_h[hidx];
            }
        }
    }

    float dX_q[TOKENS * EMBED_DIM];
    float dX_k[TOKENS * EMBED_DIM];
    float dX_v[TOKENS * EMBED_DIM];

    lora_project_backward_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WQ, lora_AQ, lora_BQ, lora_alpha, dQ, dX_q, dAQ, dBQ);

    // K has no LoRA branch and WK is frozen: dIn half only.
    gemm_backward_dIn1<TOKENS, EMBED_DIM, EMBED_DIM>(dK, WK, dX_k);

    lora_project_backward_impl<TOKENS, EMBED_DIM, LORA_RANK>(
        X, WV, lora_AV, lora_BV, lora_alpha, dV, dX_v, dAV, dBV);

    for (int i = 0; i < TOKENS * EMBED_DIM; i++) {
#pragma HLS PIPELINE II=1
        dX[i] = dX_q[i] + dX_k[i] + dX_v[i];
    }
}

// ============================================================
// Full transformer block (LN -> MHA+LoRA -> residual -> LN ->
// MLP -> residual): training-mode forward and backward.
// ============================================================

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS, int LORA_RANK>
void transformer_block_lora_forward_train_impl(
    const float *X,
    const float *WQ, const float *WK, const float *WV, const float *WO,
    const float *lora_AQ, const float *lora_BQ,
    const float *lora_AV, const float *lora_BV,
    float lora_alpha,
    const float *W1, const float *W2,
    float *out,
    float *layer1_cache,   // LN1(X)
    float *Q_cache, float *K_cache, float *V_cache,
    float *probs_cache,    // [NUM_HEADS * TOKENS * TOKENS]
    float *layer3_cache,   // X + attn(LN1(X))
    float *layer4_cache,   // LN2(layer3)
    float *inner_cache)    // layer4 @ W1, pre-GELU
{
    float layer2[TOKENS * EMBED_DIM];
    float layer5[TOKENS * EMBED_DIM];
    float inner_gelu[TOKENS * MLP_DIM];

    layer_norm_impl<TOKENS, EMBED_DIM>(X, layer1_cache);

    multi_head_lora_forward_train_impl<TOKENS, EMBED_DIM, NUM_HEADS, LORA_RANK>(
        layer1_cache, WQ, lora_AQ, lora_BQ, WK, WV, lora_AV, lora_BV, WO,
        lora_alpha, layer2, Q_cache, K_cache, V_cache, probs_cache);

    residual_add_impl<TOKENS, EMBED_DIM>(X, layer2, layer3_cache);

    layer_norm_impl<TOKENS, EMBED_DIM>(layer3_cache, layer4_cache);

    linear_impl<TOKENS, EMBED_DIM, MLP_DIM>(layer4_cache, W1, inner_cache);
    gelu_impl<TOKENS, MLP_DIM>(inner_cache, inner_gelu);
    linear_impl<TOKENS, MLP_DIM, EMBED_DIM>(inner_gelu, W2, layer5);

    residual_add_impl<TOKENS, EMBED_DIM>(layer3_cache, layer5, out);
}

template<int TOKENS, int EMBED_DIM, int MLP_DIM, int NUM_HEADS, int LORA_RANK>
void transformer_block_lora_backward_impl(
    const float *X,
    const float *WQ, const float *WK, const float *WV, const float *WO,
    const float *lora_AQ, const float *lora_BQ,
    const float *lora_AV, const float *lora_BV,
    float lora_alpha,
    const float *W1, const float *W2,
    const float *layer1_cache,
    const float *Q_cache, const float *K_cache, const float *V_cache,
    const float *probs_cache,
    const float *layer3_cache,
    const float *layer4_cache,
    const float *inner_cache,
    const float *dOut,     // gradient wrt the block's output
    float *dX,             // gradient wrt the block's input (for the previous block)
    float *dAQ, float *dBQ,
    float *dAV, float *dBV)
{
    // out = layer3 + layer5  ->  dLayer3 gets a direct dOut contribution,
    // dLayer5 = dOut.
    float dLayer4[TOKENS * EMBED_DIM];
    mlp_backward_impl<TOKENS, EMBED_DIM, MLP_DIM>(
        W1, W2, inner_cache, dOut, dLayer4);

    float dLayer3_from_ln2[TOKENS * EMBED_DIM];
    layer_norm_backward_impl<TOKENS, EMBED_DIM>(
        layer3_cache, layer4_cache, dLayer4, dLayer3_from_ln2);

    float dLayer3[TOKENS * EMBED_DIM];
    for (int i = 0; i < TOKENS * EMBED_DIM; i++) {
#pragma HLS PIPELINE II=1
        dLayer3[i] = dOut[i] + dLayer3_from_ln2[i];
    }

    // layer3 = X + layer2  ->  dX gets a direct dLayer3 contribution,
    // dLayer2 = dLayer3.
    float dLayer1[TOKENS * EMBED_DIM];
    multi_head_lora_backward_impl<TOKENS, EMBED_DIM, NUM_HEADS, LORA_RANK>(
        layer1_cache, WQ, lora_AQ, lora_BQ, WK, WV, lora_AV, lora_BV, WO,
        lora_alpha, Q_cache, K_cache, V_cache, probs_cache,
        dLayer3, dLayer1, dAQ, dBQ, dAV, dBV);

    float dX_from_ln1[TOKENS * EMBED_DIM];
    layer_norm_backward_impl<TOKENS, EMBED_DIM>(
        X, layer1_cache, dLayer1, dX_from_ln1);

    for (int i = 0; i < TOKENS * EMBED_DIM; i++) {
#pragma HLS PIPELINE II=1
        dX[i] = dLayer3[i] + dX_from_ln1[i];
    }
}
