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

extern "C" {
    void attention_softmax(const float *scores, float *scores_softmax) {
        #pragma HLS INTERFACE m_axi port=scores bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=scores_softmax bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE s_axilite port=scores bundle=control
        #pragma HLS INTERFACE s_axilite port=scores_softmax bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        attention_softmax_impl<8>(scores, scores_softmax);
    }
}