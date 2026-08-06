#include <iostream>

const unsigned int WIDTH = 8;
const unsigned int HEIGHT = 8;
const unsigned int SIZE = WIDTH * HEIGHT;

static unsigned int in1[WIDTH * WEIGHT] = {
    1,  2,  3,  4,  5,  6,  7,  8,
    9, 10, 11, 12, 13, 14, 15, 16,
   17, 18, 19, 20, 21, 22, 23, 24,
   25, 26, 27, 28, 29, 30, 31, 32,
   33, 34, 35, 36, 37, 38, 39, 40,
   41, 42, 43, 44, 45, 46, 47, 48,
   49, 50, 51, 52, 53, 54, 55, 56,
   57, 58, 59, 60, 61, 62, 63, 64

};

static unsigned int in2[WIDTH * WEIGHT] = {
    1,  2,  3,  4,  5,  6,  7,  8,
    9, 10, 11, 12, 13, 14, 15, 16,
   17, 18, 19, 20, 21, 22, 23, 24,
   25, 26, 27, 28, 29, 30, 31, 32,
   33, 34, 35, 36, 37, 38, 39, 40,
   41, 42, 43, 44, 45, 46, 47, 48,
   49, 50, 51, 52, 53, 54, 55, 56,
   57, 58, 59, 60, 61, 62, 63, 64

};

const int expected = 89440;

 
extern "C" {
    mac(const unsigned int in1, const unsigned int in2, unsigned long long out, int size);
}

int main() {
    unsigned int output = 0;
    mac(in1, in2, output, SIZE);
    if (output == expected) {
        std::cout << "Test Successful" << std::endl;
        return 1;
    }
    std::cout << "Test Unsuccessful" << std::endl;
    return 0;
}

