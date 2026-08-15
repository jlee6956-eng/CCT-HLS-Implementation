#include <cmath>
#include "cct_modules.h"

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

extern "C" {
    void attention_scores(const float *query, const float *key, float *score) {
        #pragma HLS INTERFACE m_axi port=key bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=query bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=score bundle=gmemm2 offset=slave
        #pragma HLS INTERFACE s_axilite port=key bundle=control
        #pragma HLS INTERFACE s_axilite port=query bundle=control
        #pragma HLS INTERFACE s_axilite port=score bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        attention_scores<8, 8>(query, key, score);
    }
}