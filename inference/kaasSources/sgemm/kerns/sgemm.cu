#include <stdint.h>
#include <stdio.h>

extern "C"
__global__ void sumKern(uint32_t *input, uint32_t *out)
{
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            input[fst] += input[snd];
        }

        step_size <<= 1; 
        number_of_threads >>= 1;
        __syncthreads();
    }

    if(tid == 0) {
        *out = input[0];
    }
}

extern "C"
__global__ void prodKern(uint64_t len, uint32_t *v0, uint32_t *v1, uint32_t *vout)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len) {
        vout[id] = v0[id] * v1[id];    
    }
}

/* Taken from: https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/ */
extern "C"
__global__ void sgemm(int N, float* A, float* B, float* C) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}
