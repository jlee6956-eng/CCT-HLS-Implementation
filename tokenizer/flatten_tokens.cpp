template<int CH, int H, int W>
void flatten_tokens_impl(
    const float *in,
    float *out
)
{
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {

            int token = y * W + x;

            for (int c = 0; c < CH; c++) {

                int input_index =
                    c * H * W
                    + y * W
                    + x;

                int output_index =
                    token * CH
                    + c;

                out[output_index] = in[input_index];
            }
        }
    }
}

extern "C" {
    void flatten_tokens(const float *in, float *out) {
        #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=in bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        flatten_tokens_impl<256, 2, 2>(in, out);
    }

}