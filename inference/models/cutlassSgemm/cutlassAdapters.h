#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor>; // Layout of C matrix

typedef struct CudaConfig {
    int gridX;
    int gridY;
    int gridZ;

    int blockX;
    int blockY;
    int blockZ;

    int smem_size;
} CudaConfig;

extern "C" 
CutlassGemm::GemmKernel::Params *adaptSGEMMArgs(int M, int N, int K, 
        float alpha,
        float const *A, int lda,
        float const *B, int ldb,
        float beta,
        float *C, int ldc);

extern "C"
CudaConfig *getCudaConfig(int M, int N, int K);

typedef struct _testStruct {
    float *dPtr;
    int anInt;
} testStruct;

extern "C"
testStruct *getTestStruct(int anInt, float *dPtr);

