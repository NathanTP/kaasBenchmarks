#include "cutlassAdapters.h"
#include "cutlass/gemm/device/gemm.h"

// This is a template kernel
extern "C" {
    template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);
}

extern "C"
__global__ void testKernel(testStruct s) {
    printf("from kernel: anInt=%d dPtr=%p\n", s.anInt, s.dPtr);

    precision sum = 0;
    for(int i = 0; i < 4096*2; i++) {
        sum += s.dPtr[i];
    }
    printf("sum is: %.3f + %.3fj\n", sum.real(), sum.imag());
}

extern "C"
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  precision const *A,
  int lda,
  precision const *B,
  int ldb,
  float beta,
  precision *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  precision alpha_c;
  precision beta_c;
  alpha_c.real() = alpha;
  alpha_c.imag() = 0.0;
  beta_c.real() = beta;
  beta_c.imag() = 0.0;
  if (i < M && j < N) {
    precision accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha_c * accumulator + beta_c * C[i + j * ldc];
  }
}

