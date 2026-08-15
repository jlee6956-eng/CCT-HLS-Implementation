#include <iostream>
#include "cct_modules.h"

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


extern "C" {

void attention(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    float *next_layer)
{
#pragma HLS INTERFACE m_axi port=X          bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=WQ         bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=WK         bundle=gmem2 offset=slave
#pragma HLS INTERFACE m_axi port=WV         bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=WO         bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=next_layer bundle=gmem2 offset=slave

#pragma HLS INTERFACE s_axilite port=X          bundle=control
#pragma HLS INTERFACE s_axilite port=WQ         bundle=control
#pragma HLS INTERFACE s_axilite port=WK         bundle=control
#pragma HLS INTERFACE s_axilite port=WV         bundle=control
#pragma HLS INTERFACE s_axilite port=WO         bundle=control
#pragma HLS INTERFACE s_axilite port=next_layer bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    attention_impl<3, 4, 4>(
        X,
        WQ,
        WK,
        WV,
        WO,
        next_layer
    );
}

}