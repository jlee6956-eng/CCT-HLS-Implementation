#include "cct_modules.h"

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

extern "C" {
void qkv_project(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    float *Q_proj,
    float *K_proj,
    float *V_proj)
{
#pragma HLS INTERFACE m_axi port=X       bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=WQ      bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=WK      bundle=gmem2 offset=slave
#pragma HLS INTERFACE m_axi port=WV      bundle=gmem3 offset=slave
#pragma HLS INTERFACE m_axi port=Q_proj  bundle=gmem4 offset=slave
#pragma HLS INTERFACE m_axi port=K_proj  bundle=gmem5 offset=slave
#pragma HLS INTERFACE m_axi port=V_proj  bundle=gmem6 offset=slave

#pragma HLS INTERFACE s_axilite port=X      bundle=control
#pragma HLS INTERFACE s_axilite port=WQ     bundle=control
#pragma HLS INTERFACE s_axilite port=WK     bundle=control
#pragma HLS INTERFACE s_axilite port=WV     bundle=control
#pragma HLS INTERFACE s_axilite port=Q_proj bundle=control
#pragma HLS INTERFACE s_axilite port=K_proj bundle=control
#pragma HLS INTERFACE s_axilite port=V_proj bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    qkv_project_impl<8, 8>(
        X, WQ, WK, WV,
        Q_proj, K_proj, V_proj
    );
}
}