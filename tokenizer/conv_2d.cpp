template<
    int IN_CH,
    int OUT_CH,
    int IN_H,
    int IN_W
>
void conv_2d_impl(
    const float *input,
    const float *weights,
    float *output
)
{
    const int KERNEL  = 3;
    const int STRIDE  = 2;
    const int PADDING = 1;

    const int OUT_H =
        (IN_H + 2 * PADDING - KERNEL) / STRIDE + 1;

    const int OUT_W =
        (IN_W + 2 * PADDING - KERNEL) / STRIDE + 1;

    for (int oc = 0; oc < OUT_CH; oc++) {

        for (int oy = 0; oy < OUT_H; oy++) {

            for (int ox = 0; ox < OUT_W; ox++) {

                float acc = 0.0f;
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int ky = 0; ky < KERNEL; ky++) {
                        for (int kx = 0; kx < KERNEL; kx++) {
                            int iy = oy * STRIDE + ky - PADDING;
                            int ix = ox * STRIDE + kx - PADDING;
                            if (iy >= 0 && iy < IN_H &&
                                ix >= 0 && ix < IN_W) {
                                int input_idx =
                                    ic * IN_H * IN_W
                                    + iy * IN_W
                                    + ix;
                                int weight_idx =
                                    oc * IN_CH * KERNEL * KERNEL
                                    + ic * KERNEL * KERNEL
                                    + ky * KERNEL
                                    + kx;
                                acc +=
                                    input[input_idx]
                                    * weights[weight_idx];
                            }
                        }
                    }
                }
                int output_idx =
                    oc * OUT_H * OUT_W
                    + oy * OUT_W
                    + ox;

                output[output_idx] = acc;
            }
        }
    }
}

extern "C" {
    void conv_2d(const float *input, const float *weights, float *output) {
        #pragma HLS INTERFACE m_axi port=input bundle=gmemm0 offset=slave
        #pragma HLS INTERFACE m_axi port=weights bundle=gmemm1 offset=slave
        #pragma HLS INTERFACE m_axi port=output bundle=gmemm2 offset=slave
        #pragma HLS INTERFACE s_axilite port=input bundle=control
        #pragma HLS INTERFACE s_axilite port=weights bundle=control
        #pragma HLS INTERFACE s_axilite port=output bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        conv_2d_impl<3, 64, 32, 32>(input, weights, output);
    }
}