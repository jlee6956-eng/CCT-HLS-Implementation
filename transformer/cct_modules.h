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


template<int TOKENS, int DIM>
void gelu_impl(const float *X, float *out)
{
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {

            int index = i * DIM + j;

            out[index] = 0.5f * X[index] *
                         (1.0f + std::erf(
                             X[index] / std::sqrt(2.0f)
                         ));
        }
    }
}

template<int TOKENS, int ORIG_DIM, int INNER_DIM>
void mlp_impl(
    const float *X,
    const float *W1,
    const float *W2,
    float *out)
{
    float inner_layer[TOKENS * INNER_DIM];
    float inner_layer_gelu[TOKENS * INNER_DIM];
    linear_impl<TOKENS, ORIG_DIM, INNER_DIM>(
        X,
        W1,
        inner_layer
    );
    gelu_impl<TOKENS, INNER_DIM>(
        inner_layer,
        inner_layer_gelu
    );
    linear_impl<TOKENS, INNER_DIM, ORIG_DIM>(
        inner_layer_gelu,
        W2,
        out
    );
}

template<int TOKENS, int DIM>
void residual_add_impl(const float *X, const float *MX, float *out) {
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            int index = i * DIM + j;
            out[index] = X[index] + MX[index];
        }
    }
}

template<int TOKENS, int DIM, int DIM_OUT>
void attention_impl(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    float *next_layer)
{
    float Q_proj[TOKENS * DIM];
    float K_proj[TOKENS * DIM];
    float V_proj[TOKENS * DIM];

    float scores[TOKENS * TOKENS];
    float scores_softmax[TOKENS * TOKENS];

    float scores_valued[TOKENS * DIM];

    // ========================================================
    // 1. QKV PROJECTION
    // ========================================================

    qkv_project_impl<TOKENS, DIM>(
        X,
        WQ,
        WK,
        WV,
        Q_proj,
        K_proj,
        V_proj
    );

    std::cout << "\nQ_proj:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            std::cout << Q_proj[i * DIM + j] << " ";
        }
        std::cout << std::endl;
    }


    std::cout << "\nK_proj:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            std::cout << K_proj[i * DIM + j] << " ";
        }
        std::cout << std::endl;
    }


    std::cout << "\nV_proj:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            std::cout << V_proj[i * DIM + j] << " ";
        }
        std::cout << std::endl;
    }


    // ========================================================
    // 2. ATTENTION SCORES
    // Includes QK^T / sqrt(DIM)
    // ========================================================

    attention_scores_impl<TOKENS, DIM>(
        Q_proj,
        K_proj,
        scores
    );

    std::cout << "\nScaled attention scores:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < TOKENS; j++) {
            std::cout << scores[i * TOKENS + j] << " ";
        }
        std::cout << std::endl;
    }


    // ========================================================
    // 3. SOFTMAX
    // ========================================================

    attention_softmax_impl<TOKENS>(
        scores,
        scores_softmax
    );

    std::cout << "\nAttention weights (softmax):\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < TOKENS; j++) {
            std::cout << scores_softmax[i * TOKENS + j] << " ";
        }
        std::cout << std::endl;
    }


    // ========================================================
    // 4. ATTENTION WEIGHTS * V
    // ========================================================

    attention_value_impl<TOKENS, DIM>(
        scores_softmax,
        V_proj,
        scores_valued
    );

    std::cout << "\nAttention x V:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM; j++) {
            std::cout << scores_valued[i * DIM + j] << " ";
        }
        std::cout << std::endl;
    }


    // ========================================================
    // 5. OUTPUT PROJECTION
    // ========================================================

    linear_impl<TOKENS, DIM, DIM_OUT>(
        scores_valued,
        WO,
        next_layer
    );

    std::cout << "\nFinal output:\n";
    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM_OUT; j++) {
            std::cout << next_layer[i * DIM_OUT + j] << " ";
        }
        std::cout << std::endl;
    }
}

template<int TOKENS, int EMBED_DIM, int NUM_HEADS>
void multi_head_impl(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    float *next_layer)
{

    const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

    float Q[TOKENS * EMBED_DIM];
    float K[TOKENS * EMBED_DIM];
    float V[TOKENS * EMBED_DIM];

    float concatenated[TOKENS * EMBED_DIM];

    float Q_h[TOKENS * HEAD_DIM];
    float K_h[TOKENS * HEAD_DIM];
    float V_h[TOKENS * HEAD_DIM];

    float scores[TOKENS * TOKENS];
    float scores_softmax[TOKENS * TOKENS];
    float head_output[TOKENS * HEAD_DIM];

    qkv_project_impl<TOKENS, EMBED_DIM>(
        X,
        WQ,
        WK,
        WV,
        Q,
        K,
        V
    );


    for (int h = 0; h < NUM_HEADS; h++) {
        for (int token = 0; token < TOKENS; token++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int full_index = token * EMBED_DIM
                               + h * HEAD_DIM
                               + d;
                int head_index = token * HEAD_DIM + d;
                Q_h[head_index] = Q[full_index];
                K_h[head_index] = K[full_index];
                V_h[head_index] = V[full_index];
            }
        }

        attention_scores_impl<TOKENS, HEAD_DIM>(
            Q_h,
            K_h,
            scores
        );


        attention_softmax_impl<TOKENS>(
            scores,
            scores_softmax
        );


        attention_value_impl<TOKENS, HEAD_DIM>(
            scores_softmax,
            V_h,
            head_output
        );


        for (int token = 0; token < TOKENS; token++) {

            for (int d = 0; d < HEAD_DIM; d++) {

                int head_index = token * HEAD_DIM + d;

                int full_index = token * EMBED_DIM
                               + h * HEAD_DIM
                               + d;

                concatenated[full_index] = head_output[head_index];
            }
        }
    }

    linear_impl<TOKENS, EMBED_DIM, EMBED_DIM >(
        concatenated,
        WO,
        next_layer
    );
}