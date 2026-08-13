#include <iostream>

extern "C" {
void attention(const float *X,
               const float *WQ,
               const float *WK,
               const float *WV,
               const float *WO,
               float *next_layer);
}

const int TOKENS = 3;
const int DIM = 4;
const int DIM_OUT = 4;

float X[TOKENS * DIM];

float WQ[DIM * DIM];
float WK[DIM * DIM];
float WV[DIM * DIM];
float WO[DIM * DIM_OUT];

float next_layer[TOKENS * DIM_OUT];

int main() {

    for (int i = 0; i < TOKENS * DIM; i++) {
        X[i] = (float)(i + 1);
    }

    for (int i = 0; i < DIM * DIM; i++) {
        WQ[i] = (float)(i + 1);
        WK[i] = (float)(i + 1);
        WV[i] = (float)(i + 1);
    }

    for (int i = 0; i < DIM * DIM_OUT; i++) {
        WO[i] = (float)(i + 1);
    }

    for (int i = 0; i < TOKENS * DIM_OUT; i++) {
        next_layer[i] = 0.0f;
    }

    attention(
        X,
        WQ,
        WK,
        WV,
        WO,
        next_layer
    );

    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < DIM_OUT; j++) {
            std::cout << next_layer[i * DIM_OUT + j] << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}