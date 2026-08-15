#include <iostream>
#include <cstddef>

// include your kernel declaration here, or a header file
extern "C" {
    void gemv(const unsigned int *W,
              const unsigned int *x,
              unsigned long long *out,
              int rows,
              int cols);
}

const unsigned int WIDTH = 8;
const unsigned int HEIGHT = 8;
const unsigned int SIZE = WIDTH * HEIGHT;

static unsigned int W[SIZE] = {
    1,  2,  3,  4,  5,  6,  7,  8,
    9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64
};

static unsigned int x[WIDTH] = {
    1, 2, 3, 4, 5, 6, 7, 8
};

static unsigned long long expected_x[HEIGHT] = {
    204, 492, 780, 1068, 1356, 1644, 1932, 2220
};

int main() {
    bool passed = true;
    unsigned long long output_x[HEIGHT] = {0};

    gemv(W, x, output_x, HEIGHT, WIDTH);

    for (int i = 0; i < HEIGHT; i++) {
        if (output_x[i] != expected_x[i]) {
            passed = false;
            std::cout << "Mismatch at index " << i
                      << ": expected " << expected_x[i]
                      << ", got " << output_x[i] << std::endl;
        }
    }

    if (passed)
        std::cout << "TEST PASSED!" << std::endl;
    else
        std::cout << "TEST FAILED!" << std::endl;

    return passed ? 0 : 1;
}