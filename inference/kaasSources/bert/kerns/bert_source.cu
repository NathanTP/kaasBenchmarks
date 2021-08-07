
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void fused_mean_kernel1(float* __restrict__ T_divide, float* __restrict__ placeholder_red) {
  T_divide[(((int)threadIdx.x))] = (placeholder_red[(((int)threadIdx.x))] * 9.765625e-04f);
}

extern "C" __global__ void fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
      T_transpose[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (placeholder[(((((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 384) * 1024) + ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 384)))] + placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 384))]);
    }
  }
}

extern "C" __global__ void fused_nn_batch_matmul_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[512];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      compute_local[(((i_c_init * 8) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 48; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = placeholder[(((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.y) * 24576)) + (((int)threadIdx.y) * 3072)) + (ax1_inner * 384)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 24576) + (((int)threadIdx.y) * 3072)) + (ax1_inner1 * 384)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((int)threadIdx.x) * 64) + (ax11 * 8)) + k_inner))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          compute_local[(((i_c * 8) + j_c))] = (compute_local[(((i_c * 8) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner))] = compute_local[(((i_inner_inner * 8) + j_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_nn_batch_matmul_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[512];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      compute_local[(((i_c_init * 8) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = placeholder[(((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (ax1_inner * 64)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 8)) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (ax1_inner1 * 64)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((int)threadIdx.x) * 64) + (ax11 * 8)) + k_inner))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          compute_local[(((i_c * 8) + j_c))] = (compute_local[(((i_c * 8) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      compute[((((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.y) * 24576)) + (((int)threadIdx.y) * 3072)) + (i_inner_inner * 384)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner))] = compute_local[(((i_inner_inner * 8) + j_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_nn_batch_matmul_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[512];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      compute_local[(((i_c_init * 8) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 128; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (ax1_inner * 1024)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 8192)) + (ax1_inner1 * 1024)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((int)threadIdx.x) * 64) + (ax11 * 8)) + k_inner))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          compute_local[(((i_c * 8) + j_c))] = (compute_local[(((i_c * 8) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      compute[(((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 32768)) + (i_inner_inner * 4096)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner))] = compute_local[(((i_inner_inner * 8) + j_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_add_sqrt_divide_multiply_add_reshape_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] / sqrtf((placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] + 1.000000e-12f))) * placeholder2[(((int)threadIdx.x))]) + placeholder3[(((int)threadIdx.x))]);
      }
    }
  }
}

extern "C" __global__ void fused_power_mean_kernel0(float* __restrict__ placeholder, float* __restrict__ T_power_red) {
  float T_power_red_rf[1];
  float red_buf0[1];
  T_power_red_rf[(0)] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 32; ++k2_outer) {
    T_power_red_rf[(0)] = (T_power_red_rf[(0)] + powf(placeholder[(((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 1024)) + (k2_outer * 32)) + ((int)threadIdx.x)))], 2.000000e+00f));
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = T_power_red_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    T_power_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_add_sqrt_divide_multiply_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder1, float* __restrict__ placeholder, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_add[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] / sqrtf((placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] + 1.000000e-12f))) * placeholder2[(((int)threadIdx.x))]) + placeholder3[(((int)threadIdx.x))]);
      }
    }
  }
}

extern "C" __global__ void fused_reshape_add_reshape_transpose_reshape_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) < 6144) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (placeholder[((((((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 384) * 1024) + (((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 384) * 64)) + (((int)threadIdx.x) & 63)))] + placeholder1[(((((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 384) * 64) + (((int)threadIdx.x) & 63)))]);
      }
    }
  }
}

extern "C" __global__ void fused_subtract_exp_kernel0(float* __restrict__ T_exp, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    T_exp[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = __expf((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] - placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 384))]));
  }
}

extern "C" __global__ void fused_divide_reshape_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 9; ++ax0_ax1_fused_ax2_fused_outer) {
    T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] / placeholder1[(((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 384))]);
  }
}

extern "C" __global__ void fused_sum_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k3_outer = 0; k3_outer < 12; ++k3_outer) {
    placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[(((((((int)blockIdx.x) * 12288) + (((int)threadIdx.y) * 384)) + (k3_outer * 32)) + ((int)threadIdx.x)))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = placeholder_red_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_expand_dims_expand_dims_cast_subtract_multiply_kernel0(float* __restrict__ T_multiply, int64_t* __restrict__ placeholder) {
  T_multiply[(((int)threadIdx.x))] = ((1.000000e+00f - ((float)placeholder[(((int)threadIdx.x))])) * -1.000000e+04f);
}

extern "C" __global__ void fused_mean_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 32; ++k2_outer) {
    placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[(((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 1024)) + (k2_outer * 32)) + ((int)threadIdx.x)))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = placeholder_red_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_squeeze_kernel0(float* __restrict__ T_squeeze, float* __restrict__ placeholder) {
  T_squeeze[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void fused_nn_batch_matmul_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[512];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      compute_local[(((i_c_init * 8) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 512; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 32768)) + (ax1_inner * 4096)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.y) * 32768)) + (ax1_inner1 * 4096)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((int)threadIdx.x) * 64) + (ax11 * 8)) + k_inner))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          compute_local[(((i_c * 8) + j_c))] = (compute_local[(((i_c * 8) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      compute[(((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_inner_inner * 1024)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner))] = compute_local[(((i_inner_inner * 8) + j_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_nn_batch_matmul_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[512];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      compute_local[(((i_c_init * 8) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 128; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (ax1_inner * 1024)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 8192)) + (ax1_inner1 * 1024)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((int)threadIdx.x) * 64) + (ax11 * 8)) + k_inner))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          compute_local[(((i_c * 8) + j_c))] = (compute_local[(((i_c * 8) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      compute[(((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_inner_inner * 1024)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner))] = compute_local[(((i_inner_inner * 8) + j_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_nn_batch_matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[16];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[1];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 128; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      #pragma unroll
      for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        placeholder_shared[(((((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + (((int)threadIdx.x) * 4)) + ax2_inner))] = placeholder[(((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (ax1_inner * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) * 4)) + ax2_inner))];
      }
    }
    #pragma unroll
    for (int ax2_inner1 = 0; ax2_inner1 < 4; ++ax2_inner1) {
      if (((int)threadIdx.y) < 2) {
        placeholder_d_shared[((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + ax2_inner1))] = placeholder1[(((((((int)threadIdx.y) * 1024) + (k_outer * 8)) + (((int)threadIdx.x) * 4)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner))];
      }
      placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((int)threadIdx.x) * 8) + k_inner))];
      #pragma unroll
      for (int i_c = 0; i_c < 8; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(0)]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    compute[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 16)) + (i_inner_inner * 2)) + ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}

extern "C" __global__ void fused_reshape_add_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_add[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] + placeholder1[(((int)threadIdx.x))]) + placeholder2[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]);
      }
    }
  }
}

extern "C" __global__ void fused_reshape_transpose_reshape_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = placeholder[((((((((int)threadIdx.x) >> 6) * 24576) + (ax0_ax1_fused_ax2_fused_outer * 16384)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)))];
      }
    }
  }
}

extern "C" __global__ void fused_reshape_divide_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    T_add[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] * 1.250000e-01f) + placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 384))]);
  }
}

extern "C" __global__ void fused_max_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = -3.402823e+38f;
  for (int k3_outer = 0; k3_outer < 12; ++k3_outer) {
    placeholder_red_rf[(0)] = max(placeholder_red_rf[(0)], placeholder[(((((((int)blockIdx.x) * 12288) + (((int)threadIdx.y) * 384)) + (k3_outer * 32)) + ((int)threadIdx.x)))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = placeholder_red_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_power_mean_kernel1(float* __restrict__ T_divide, float* __restrict__ T_power_red) {
  T_divide[(((int)threadIdx.x))] = (T_power_red[(((int)threadIdx.x))] * 9.765625e-04f);
}

extern "C" __global__ void fused_reshape_add_split_kernel0(float* __restrict__ T_split, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_split[(((int)threadIdx.x))] = (placeholder[((((int)threadIdx.x) * 2))] + placeholder1[(0)]);
}

extern "C" __global__ void fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 6; ++ax0_ax1_fused_ax2_fused_outer) {
    T_reshape[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) & 4095))]) * 5.000000e-01f) * (erff(((placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) & 4095))]) * 7.071068e-01f)) + 1.000000e+00f));
  }
}

extern "C" __global__ void fused_reshape_add_reshape_transpose_reshape_transpose_kernel0(float* __restrict__ T_transpose, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) < 6144) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_transpose[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (placeholder[((((((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 384) * 1024) + (((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 384) * 64)) + (((int)threadIdx.x) & 63)))] + placeholder1[(((((((ax0_ax1_fused_ax2_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 384) * 64) + (((int)threadIdx.x) & 63)))]);
      }
    }
  }
}

extern "C" __global__ void fused_subtract_kernel0(float* __restrict__ T_subtract, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_subtract[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (placeholder[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] - placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))]);
      }
    }
  }
}

extern "C" __global__ void fused_reshape_add_split_kernel1(float* __restrict__ T_split, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_split[(((int)threadIdx.x))] = (placeholder[(((((int)threadIdx.x) * 2) + 1))] + placeholder1[(1)]);
}

extern "C" __global__ void fused_squeeze_1_kernel0(float* __restrict__ T_squeeze, float* __restrict__ placeholder) {
  T_squeeze[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void fused_less_add_where_take_add_less_add_where_take_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3, int64_t* __restrict__ placeholder4) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)) < 384) {
      if ((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 393216) {
        T_add[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((placeholder[(((min(max((int64_t)0, ((((int)(placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] < (int64_t)0)) != 0) ? (placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] + (int64_t)30522) : placeholder1[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))])), (int64_t)30521) * (int64_t)1024) + ((int64_t)((int)threadIdx.x))))] + placeholder2[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) + placeholder3[(((min(max((int64_t)0, ((((int)(placeholder4[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] < (int64_t)0)) != 0) ? (placeholder4[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))] + (int64_t)2) : placeholder4[(((ax0_ax1_fused_ax2_fused_outer * 256) + ((int)blockIdx.x)))])), (int64_t)1) * (int64_t)1024) + ((int64_t)((int)threadIdx.x))))]);
      }
    }
  }
}

