#pragma once
#include <cmath>


template<int ROWS, int COLS>
void gemv_impl(const unsigned int *W,
               const unsigned int *x,
               unsigned long long *out) {
    for (int r = 0; r < ROWS; r++) {
        unsigned long long sum = 0;
        for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II=1
            sum += (unsigned long long)W[r * COLS + c] *
                   (unsigned long long)x[c];
        }
        out[r] = sum;
    }
}


template<int N>
void softmax_impl(const float *in, float *out) {

    // 1. Find maximum
    float max_val = in[0];

    for (int i = 1; i < N; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }

    // 2. Exponentiate shifted values and compute sum
    float exp_vals[N];
    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        exp_vals[i] = std::exp(in[i] - max_val);
        sum += exp_vals[i];
    }

    // 3. Normalize
    for (int i = 0; i < N; i++) {
        out[i] = exp_vals[i] / sum;
    }
}


template<int ROWS, int COLS>
void layer_norm_impl(const float *in, float *out) {
    const int N = ROWS * COLS;
    const float EPS = 1e-5f;

    float sum = 0.0f;
    float mean = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += in[i];
    }

    mean = sum / (float)N;

    for (int i = 0; i < N; i++) {
        float diff = in[i] - mean;
        variance += diff * diff;
    }

    variance /= (float)N;

    float denom = std::sqrt(variance + EPS);

    for (int i = 0; i < N; i++) {
        out[i] = (in[i] - mean) / denom;
    }
}

template<int M, int K, int N>
void gemm_impl(const float *in1,
               const float *in2,
               float *out) {

    for (int i = 0; i < M; i++) {

        for (int j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += in1[i * K + k] *
                       in2[k * N + j];
            }

            out[i * N + j] = sum;
        }
    }
}


template<int M, int K>
void transpose_impl(const float *in, float *out)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            out[j * M + i] = in[i * K + j];
        }
    }
}

template<int TOKENS, int DIM>
void attention_scores_impl(
    const float *query,
    const float *key,
    float *score) {
    float key_T[DIM * TOKENS];

    transpose_impl<TOKENS, DIM>(key, key_T);

    gemm_impl<TOKENS, DIM, TOKENS>(
        query,
        key_T,
        score
    );

    float scale = std::sqrt((float)DIM);

    for (int i = 0; i < TOKENS * TOKENS; i++) {
        score[i] = score[i] / scale;
    }
}

template<int TOKENS, int DIM>
void attention_value_impl(const float *in, const float *V, float *out) {
    gemm_impl<TOKENS, TOKENS, DIM>(in, V, out);
}

template<int TOKENS>
void attention_softmax_impl(const float *scores,
                            float *scores_softmax)
{
    float score_row[TOKENS];
    float score_softmax_row[TOKENS];

    for (int i = 0; i < TOKENS; i++) {

        // Grab one row
        for (int j = 0; j < TOKENS; j++) {
            score_row[j] = scores[i * TOKENS + j];
        }

        // Softmax that row
        softmax_impl<TOKENS>(
            score_row,
            score_softmax_row
        );

        // Store the resulting row
        for (int k = 0; k < TOKENS; k++) {
            scores_softmax[i * TOKENS + k]
                = score_softmax_row[k];
        }
    }
}

template<int TOKENS, int DIM>
void qkv_project_impl(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    float *Q_proj,
    float *K_proj,
    float *V_proj)
{
    gemm_impl<TOKENS, DIM, DIM>(X, WQ, Q_proj);
    gemm_impl<TOKENS, DIM, DIM>(X, WK, K_proj);
    gemm_impl<TOKENS, DIM, DIM>(X, WV, V_proj);
}


template<int TOKENS, int DIM, int OUT_DIM>
void linear_impl(const float *in, const float *W, float *out) {
    gemm_impl<TOKENS,DIM, OUT_DIM>(in, W, out);
}


