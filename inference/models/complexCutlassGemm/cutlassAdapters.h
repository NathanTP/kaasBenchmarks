#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cutlass/numeric_types.h>
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;
// using RowMajor = cutlass:layout::RowMajor;

using precision = cutlass::complex<float>;
using CutlassGemm = cutlass::gemm::device::Gemm<
      precision, ColumnMajor,
      precision, ColumnMajor,
      precision, ColumnMajor,
      precision,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50
  >;

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
        precision const *A, int lda,
        precision const *B, int ldb,
        float beta,
        precision *C, int ldc);

extern "C"
CudaConfig *getCudaConfig(int M, int N, int K);

typedef struct _testStruct {
    precision *dPtr;
    int anInt;
} testStruct;

extern "C"
testStruct *getTestStruct(int anInt, precision *dPtr);

