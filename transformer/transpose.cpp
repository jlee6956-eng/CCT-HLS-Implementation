template<int M, int K>
void transpose_impl(const float *in, float *out)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            out[j * M + i] = in[i * K + j];
        }
    }
}


extern "C" {

void transpose(const float *in, float *out)
{
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave

#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    transpose_impl<2, 3>(in, out);
}

}