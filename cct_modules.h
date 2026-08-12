#ifndef CCT_MODULES_H
#define CCT_MODULES_H

#include <ap_int.h>


template<int ROWS, int COLS>
void gemv_impl(const unsigned int *W,
               const unsigned int *x,
               unsigned long long *out);


template<int N>
void softmax_impl(const float *in,
                  float *out);


template<int ROWS, int COLS>
void layer_norm_impl(const float *in,
                     float *out);

template<int M, int K, int N>
void gemm_impl(const float *in1,
               const float *in2,
               float *out);


template<int M, int K>
void transpose_impl(const float *int, float *out);

template<int TOKENS, int DIM> 
void attention_scores_impl (const float *query, const float *key, float score);

template<int TOKENS, int DIM>
void attention_value_impl(const float *in, const float *V, float *out)
    gemm<TOKENS, TOKENS, DIM>(in, V, out);

template<int TOKENS>
void attention_softmax_impl (const float *scores, float *scores_softmax);

template<int TOKENS, int DIM>
void qkv_project_impl(
    const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    float *Q_proj,
    float *K_proj,
    float *V_proj);


template<int TOKENS, int DIM, int OUT_DIM>
void linear_impl(const float *attention_weights, const float *W, float *next_layer);



#endif