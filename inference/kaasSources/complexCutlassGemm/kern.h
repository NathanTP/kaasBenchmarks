extern "C"
__global__ void testKernel(testStruct s);

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
  int ldc);
