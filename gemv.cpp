#include <ap_int.h>

extern "C" {
   void gemv(const unsigned int *W, const unsigned int *x, unsigned long long *out, int rows, int cols) {
    #pragma HLS INTERFACE m_axi port=W bundle=gmem0 offset=slave
    #pragma HLS INTERFACE m_axi port=x bundle=gmem0 offset=slave
    #pragma HLS INTERFACE m_axi port=out bundle=gmem0 offset=slave
    #pragma HLS INTERFACE s_axilite port=W bundle=control
    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE x_axilite port=rows bundle=control
    #pragma HLS INTERFACE x_axilite port=cols bundle=control

    for (int i = 0; i < rows; i++) {
        #pragma PIPELINE II=1
        unsigned long long sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += W[i][j]

        }
        out[i, j] = sum
    }
   } 
}