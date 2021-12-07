#include <stdint.h>
    
extern "C"
__global__ void dummy(uint8_t *input, uint8_t *out)
{
    for(int i = 0; i < 1024; i++) {
        out[i] = input[i];
    }
    return;
}
