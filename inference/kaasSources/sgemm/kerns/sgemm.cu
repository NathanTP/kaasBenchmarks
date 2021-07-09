#include <stdint.h>
#include <stdio.h>

// This kernel actually supports arbitrary matrix shapes, but we just hard-code
// to square for now
#define DIM 128
#define NROW_A DIM
#define NCOL_A DIM
#define NROW_B DIM
#define NCOL_B DIM

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

/* #define CMAJ */

#ifdef CMAJ
    #define flatIdx(R,C,NROW,NCOL) ((C*NROW)+R)
#else
    #define flatIdx(R,C,NROW,NCOL) ((R*NCOL)+C)
#endif

// Generic matrix multiply.
// This one in row major is roughly as fast as sgemm in col major (sgemm is slightly faster, but only in colmajor) 
//      perf differences depend on shape, some shapes show no difference
// This one is only correct for some shapes (square works, some other combos as well, but e.g. 512x256, 256x512 fails)
// Original implementation by Aditi Singh (https://github.com/aditisingh/GPU-Gemm)
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
extern "C"
__global__ void matmul(float* array0, float* array1, float* outArr)
{
    //shared memory takes one tile at a time
    __shared__ float S1[TILE_WIDTH][TILE_HEIGHT];
    __shared__ float S2[TILE_HEIGHT][TILE_WIDTH];

    //threads x and y index for the current block
    unsigned int tx=threadIdx.x;	
    unsigned int ty=threadIdx.y;

    //row value using x and y index of current thread (respectively)
    unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	
    unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;

    //register to store multiplication result initialized to zero
    float val=0;

    //going over all tiles one by one, with each m
    for(int m=0; m<1+((NROW_B-1)/TILE_WIDTH); m++)
    {
        //x and y thread value for current tile
        int var1=m*TILE_WIDTH+tx;
        int var2=m*TILE_WIDTH+ty;

        //copying a tile from array0
        //if the value is associated to a valid matrix coordinate in array0
        //then store it to shared memory S1
        if (r < NROW_A && var1 < NROW_B) {
            //storing a "valid" value from array to shared memory
            S1[ty][tx] = array0[flatIdx(r, var1, NROW_A, NCOL_A)];
        } else {
            //storing zero, since there is no valid value
            S1[ty][tx]=0;					
        }
        __syncthreads();

        //copying a tile from array1
        //if value is associates to a valid matrix coordinate in array1 then
        //store it to shared memory S2
        if(c < NCOL_B && var2 < NROW_B) {
            S2[ty][tx] = array1[flatIdx(var2, c, NROW_B, NCOL_B)];
        } else { 
            //storing zero, since no valid value
            S2[ty][tx]=0;
        }
        __syncthreads();

        //going over entire tile, ty row in S1 and tx column in S2
        for(int i=0; i<TILE_WIDTH;i++) {
            val+=S1[ty][i]*S2[i][tx];
        }
        __syncthreads();
    }

    //removing degenerate cases
    if(r < NROW_A && c< NCOL_B) {
        //saving multiplication result to global memory
        outArr[flatIdx(r, c, NROW_A, NCOL_B)] = val;
    }
}

// Number of columns in tile (length of a tile row)
#define TILE_N 16
//#define TILE_N 32

// Number of rows in tile (length of a tile column)???
#define TILE_TB_HEIGHT 8
//#define TILE_TB_HEIGHT 16

// Total number of elements in a tile (why is this called M?, it's used to calculate a rowID in A...)
#define TILE_M (TILE_N*TILE_TB_HEIGHT)

/*
    N - NCOL_B, ncolC
    M - NROW_A, nrowC
    K - NCOL_A, NROW_B

    y - rowID
    x - colID

    ncol - number of elements in row (len(row))
    nrow - number of elements in col (len(col))

    rowMjr - row * ncol + col
    colMjr - col * nrow + row

    NCOL_Block - tileN
    NROW_Block - tile_tb_height

    ngridCol - NROW_A / ???? (it's NCOL_Block*NROW_Block, i.e. nelemBlock which they call TILE_M but that doesn't make sense)
    ngridRow - NCOL_B / NCOL_Block (blocks tile across columns
*/

/* Based on https://github.com/abduld/Parboil/blob/master/benchmarks/sgemm/src/cuda/sgemm_kernel.cu */
// This implementation is a bit faster than matmul above, but only in col-major mode
// While this works correctly in row-major, it's about 20% slower than
// col-major. I think I'd need to transpose the iteration/tiling order to be
// row-optimized. Probably not worth the effort for the slight performance
// boost over the other implementation above.
extern "C"
__global__ void sgemm(const float *A, const float *B, float *C)
{
    uint64_t nrowC = NROW_A;
    uint64_t ncolC = NCOL_B;

    // Partial results 
    float cTile[TILE_N];
    for (int i=0; i < TILE_N; i++) {
        cTile[i] = 0.0f;
    }

    int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id

    // global rowID for this thread (in A and C)
    int m = blockIdx.x * TILE_M + mid;

    // global column ID for this thread (in B and C)
    int n = blockIdx.y * TILE_N + threadIdx.x;

    // Tiles cover C
    __shared__ float bTile[TILE_TB_HEIGHT][TILE_N];

    // Iterate tiles down in row dimension, i is the global start row for the current tile (of B)
    // The global start row for this thread in the current tile is calculated in the loop
    for (int i = 0; i < NROW_B; i += TILE_TB_HEIGHT) {
        float a; 

        // fill in the shared bTile for this thread block (each thread does one elem)
        // i+threadIdx.y is the global rowID in B for this thread's portion of the tile
        bTile[threadIdx.y][threadIdx.x] = B[flatIdx((i+threadIdx.y), n, NROW_B, NCOL_B)];
        __syncthreads();
        // bTile now covers the tile of C starting at (blockIdx.y*TILE_N,i)

        // For each row of the tile (tileRow is the row index into the B tile)
        for (int tileRow = 0; tileRow < TILE_TB_HEIGHT; tileRow++) {
            // i+tileRow is the global k index (global colID of A, rowID of B)
            // m is the global rowID in A
            a = A[flatIdx(m, (i + tileRow), NROW_A, NCOL_A)];

            // iterate each column of the bTile
            for (int tileCol = 0; tileCol < TILE_N; tileCol++) {
                cTile[tileCol] += a * bTile[tileRow][tileCol];
            }
        }
        __syncthreads();
    }

    // t is the starting point in tile in C (tile is 1xTILE_N) 
    // i is the columnID within this tile
    int t = flatIdx(m, blockIdx.y*TILE_N, nrowC, ncolC);
    for (int i = 0; i < TILE_N; i++) {
        //cIdx is the global address of the tile element in C
        #ifdef CMAJ
            int cIdx = t+i*nrowC;
        #else
            int cIdx = t + i;
        #endif
        C[cIdx] = C[cIdx] + cTile[i];
    }
}
