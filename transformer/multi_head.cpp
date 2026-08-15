#include "cct_modules.h"

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

extern "C" {

    void multi_head(const float *X,
    const float *WQ,
    const float *WK,
    const float *WV,
    const float *WO,
    float *next_layer) {
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
        multi_head_impl<3, 4, 2>(X, WQ, WK, WV, WO, next_layer);

    }

}