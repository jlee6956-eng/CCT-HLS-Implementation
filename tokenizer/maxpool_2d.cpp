#include "tokenizer_modules.h"
template<int CH, int IN_H, int IN_W>
void maxpool_impl(
    const float *in,
    float *out
)
{
    const int KERNEL = 3;
    const int STRIDE = 2;
    const int PADDING = 1;
    const int OUT_H =
        (IN_H + 2 * PADDING - KERNEL) / STRIDE + 1;
    const int OUT_W =
        (IN_W + 2 * PADDING - KERNEL) / STRIDE + 1;
    for (int i = 0; i < CH; i++) {
    for (int j = 0; j < OUT_H; j++) {
        for (int k = 0; k < OUT_W; k++) {

            float max_value = -1e30f;

            for (int d = 0; d < KERNEL; d++) {
                for (int m = 0; m < KERNEL; m++) {

                    int iy = j * STRIDE + d - PADDING;
                    int ix = k * STRIDE + m - PADDING;

                    if (iy >= 0 && iy < IN_H &&
                        ix >= 0 && ix < IN_W) {

                        int input_index =
                            i * IN_H * IN_W
                            + iy * IN_W
                            + ix;

                        if (in[input_index] > max_value) {
                            max_value = in[input_index];
                        }
                    }
                }
            }

            int output_index =
                i * OUT_H * OUT_W
                + j * OUT_W
                + k;

            out[output_index] = max_value;
        }
    }
}

}

extern "C" {
    void max_pool(const float *in,float *out) {
        #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=in bundle=control
        #pragma HLS INTERFACE s_axilite port=out bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        maxpool_impl<64, 16, 16>(in, out);
    }
}