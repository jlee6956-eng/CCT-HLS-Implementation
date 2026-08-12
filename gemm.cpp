#include <iostream>

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

extern "C" {
void gemm(const float *in1,
          const float *in2,
          float *out) {

#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave

#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=in2 bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    gemm_impl<8, 8, 8>(in1, in2, out);
}
}