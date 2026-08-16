template<int CH, int IN_H, int IN_W>
void relu_impl(const float *in, float *out)
{
    const int SIZE = CH * IN_H * IN_W;

    for (int i = 0; i < SIZE; i++) {
        if (in[i] < 0.0f) {
            out[i] = 0.0f;
        } else {
            out[i] = in[i];
        }
    }
}

extern "C" {
    void relu(const float *in, float *out) {
        #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle= gemm1 offset=slave
        #pragma HLS INTERFACE s_axilite port=in bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        relu_impl<64, 16, 16>(in, out);
    }
}