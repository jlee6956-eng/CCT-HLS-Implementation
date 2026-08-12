#ifndef CCT_MODULES_H
#define CCT_MODULES_H

#include <ap_int.h>

template<int SIZE>
void mac_impl(const unsigned int *in1,
              const unsigned int *in2,
              unsigned long long *out);


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


template<int M, int K>
void transpose_impl(const float *int, float *out);

template<int TOKENS, int DIM> attention_scores (const float *key, const float *query, float score);

#endif