//#include "helper.h"
#include "getDims.h"

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

