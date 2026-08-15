#include <ap_int.h>

template<int ROWS, int COLS>
void gemv_impl(const unsigned int *W,
               const unsigned int *x,
               unsigned long long *out) {
    for (int r = 0; r < ROWS; r++) {
        unsigned long long sum = 0;
        for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II=1
            sum += (unsigned long long)W[r * COLS + c] *
                   (unsigned long long)x[c];
        }
        out[r] = sum;
    }
}

extern "C" {
void gemv(const unsigned int *W,
          const unsigned int *x,
          unsigned long long *out) {
#pragma HLS INTERFACE m_axi port=W   bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=x   bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave

#pragma HLS INTERFACE s_axilite port=W   bundle=control
#pragma HLS INTERFACE s_axilite port=x   bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    gemv_impl<8, 8>(W, x, out);
}
}