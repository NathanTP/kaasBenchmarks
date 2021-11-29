//#include "helper.h"
#include "cutlassAdapters.h"

extern "C"
CutlassGemm::GemmKernel::Params *adaptSGEMMArgs(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel.
  cudaStream_t stream = nullptr;
  gemm_operator.initialize(args, stream=stream);
  CutlassGemm::GemmKernel::Params params_ = gemm_operator.get_params();
  CutlassGemm::GemmKernel::Params *params_ptr = (CutlassGemm::GemmKernel::Params*) malloc(sizeof(params_));
  memcpy(params_ptr, &params_, sizeof(params_));
  return params_ptr;
}

extern "C"
CudaConfig *getCudaConfig(int M, int N, int K) {
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},
                              {NULL, M},
                              {NULL, K},
                              {NULL, M},
                              {NULL, M},
                              {0.0, 0.0});

  // Launch the CUTLASS GEMM kernel.
  cudaStream_t stream = nullptr;
  gemm_operator.initialize(args, stream=stream);
  CutlassGemm::GemmKernel::Params params = gemm_operator.get_params();

  CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
  dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
  dim3 block(CutlassGemm::GemmKernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));

  CudaConfig* cfg = new CudaConfig;
  cfg->gridX = grid.x;
  cfg->gridY = grid.y;
  cfg->gridZ = grid.z;

  cfg->blockX = block.x;
  cfg->blockY = block.y;
  cfg->blockZ = block.z;

  cfg->smem_size = smem_size;

  return cfg;
}

extern "C"
testStruct *getTestStruct(int anInt, float *dPtr) {
    testStruct *s = (testStruct*)malloc(sizeof(testStruct));
    s->anInt = anInt;
    s->dPtr = dPtr;
    return s;
}
