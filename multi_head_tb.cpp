#include <iostream>

extern "C" {
void multi_head(const float *X,
               const float *WQ,
               const float *WK,
               const float *WV,
               const float *WO,
               float *next_layer);
}

const int TOKENS = 4;
const int EMBED_DIM = 8;
const int NUM_HEADS = 2;

float X[TOKENS * EMBED_DIM];

float WQ[EMBED_DIM * EMBED_DIM];
float WK[EMBED_DIM * EMBED_DIM];
float WV[EMBED_DIM * EMBED_DIM];
float WO[EMBED_DIM * EMBED_DIM];
float next_layer[TOKENS * EMBED_DIM];
float expected[TOKENS * EMBED_DIM] = {
11396.000000000f, 12904.000000000f, 14412.000000000f, 15920.000000000f, 11396.000000000f, 12904.000000000f, 14412.000000000f, 15920.000000000f, 11396.000000000f, 12904.000000000f, 14412.000000000f, 15920.000000000
};

int main() {

    for (int i = 0; i < TOKENS * EMBED_DIM; i++) {
        X[i] = (float)(i + 1);
    }

    for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
        WQ[i] = (float)(i + 1);
        WK[i] = (float)(i + 1);
        WV[i] = (float)(i + 1);
    }

    for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
        WO[i] = (float)(i + 1);
    }

    for (int i = 0; i < TOKENS * EMBED_DIM; i++) {
        next_layer[i] = 0.0f;
    }

    multi_head(
        X,
        WQ,
        WK,
        WV,
        WO,
        next_layer
    );

    for (int i = 0; i < TOKENS; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            std::cout << next_layer[i * EMBED_DIM + j] << " ";
        }

        std::cout << std::endl;
    }

    for (int k = 0; k < TOKENS; k++) {
        for (int z = 0; z < EMBED_DIM; z++) {
            if (expected[k*EMBED_DIM + z] != next_layer[k*EMBED_DIM + z]) {
                std::cout << "FAIL!" << std::endl;
                return 1;
            }
        }
    }
    std::cout << "SUCCESS!" << std::endl;

    return 0;
}