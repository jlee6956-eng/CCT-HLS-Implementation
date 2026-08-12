#include <cmath>
#include "cct_modules.h"

template<int TOKENS, int DIM>
void attention_scores_impl(
    const float *key,
    const float *query,
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
    void attention_scores_impl(const float *key, const float *query, float *score) {
        #pragma HLS INTERFACE port=key bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE port=query bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE port=score bundle=gmemm2 offset=slave
        #pragma HLS INTERFACE port=key bundle=control
        #pragma HLS INTERFACE port=query bundle=control
        #pragma HLS INTERFACE port=score bundle=control
        #pragma HLS INTERFACE port=return bundle=control
    }
}