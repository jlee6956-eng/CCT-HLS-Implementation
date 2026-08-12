#include "cct_modules.h"

template<int TOKENS, int DIM, int DIM_OUT>
void attention_impl(const float *X, const float *WQ, const float *WK, const float *WV, const float *WO, float *next_layer){
    float Q_proj[TOKENS * DIM];
    float K_proj[TOKENS * DIM];
    float V_proj[TOKENS * DIM];
    float scores[TOKENS * TOKENS];
    float scores_softmax[TOKENS * TOKENS];
    float scores_valued[TOKENS * DIM];
    qkv_project_impl<TOKENS, DIM>(X, WQ, WK, WV, Q_proj, K_proj, V_proj);
    attention_scores_impl<TOKENS, DIM>(Q_proj, K_proj, scores);
    attention_softmax_impl<TOKENS>(scores, scores_softmax);
    attention_value_impl<TOKENS, DIM>(scores_softmax, V_proj, scores_valued);
    linear_impl<TOKENS, DIM, DIM_OUT>(scores_valued, WO, next_layer);

}

extern "C" {
    void attention(const float *X, const float *WQ, const float *WK, const float *WV, const float *WO, float *next_layer) {
        #pragma HLS INTERFACE m_axi port=X bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=WQ bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=WK bundle=gmemm2 offset=slave
        #pragma HLS INTERFACE m_axi port=WV bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=WO bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=next_layer bundle=gmemm2 offset=slave
        #pragma HLS INTERFACE s_axilite port=X bundle=control
        #pragma HLS INTERFACE s_axilite port=WQ bundle=control
        #pragma HLS INTERFACE s_axilite port=WK bundle=control
        #pragma HLS INTERFACE s_axilite port=WV bundle=control
        #pragma HLS INTERFACE s_axilite port=WO bundle=control
        #pragma HLS INTERFACE s_axilite port=next_layer bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        attention_impl<8, 8, 8>(X, WQ, WK, WV, WO, next_layer);
    }
}