#include "tokenizer_modules.h"

template <int TOKENS>
void tokenizer_impl(
    const float *input,
    const float *conv1_weights,
    const float *conv2_weights,
    float *output
)
{
    float conv1_out[64 * 16 * 16];
    float pool1_out[64 * 8 * 8];

    float conv2_out[256 * 4 * 4];
    float pool2_out[256 * 2 * 2];

    // Conv 1
    conv_2d_impl<3, 64, 32, 32>(
        input,
        conv1_weights,
        conv1_out
    );

    // ReLU 1
    relu_impl<64, 16, 16>(
        conv1_out,
        conv1_out
    );

    // MaxPool 1
    maxpool_2d_impl<64, 16, 16>(
        conv1_out,
        pool1_out
    );

    // Conv 2
    conv_2d_impl<64, 256, 8, 8>(
        pool1_out,
        conv2_weights,
        conv2_out
    );

    // ReLU 2
    relu_impl<256, 4, 4>(
        conv2_out,
        conv2_out
    );

    // MaxPool 2
    maxpool_2d_impl<256, 4, 4>(
        conv2_out,
        pool2_out
    );

    // Convert feature map to tokens
    flatten_tokens_impl<256, 2, 2>(
        pool2_out,
        output
    );
}

extern "C" {
    void tokenizer(const float *in, const float *conv_1_weights, const float *conv_2_weights, float *out) {
        #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=out bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=conv_1_weights bundle=gmem0 offset=slave
        #pragma HLS INTERFACE m_axi port=conv_2_weights bundle=gmem0 offset=slave
        #pragma HLS INTERFACE s_axilite port=in bundle=slave
        #pragma HLS INTERFACE s_axilite port=conv_1_weights bundle=slave
        #pragma HLS INTERFACE s_axilite port=conv_2_weights bundle=slave
        #pragma HLS INTERFACE s_axilite port=out bundle=slave
        #pragma HLS INTERFACE s_axilite port=return bundle=slave

        tokenizer_impl<4>(
            in,
            conv_1_weights,
            conv_2_weights,
            out
        );

    }

}