
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
extern "C" __global__ void fused_divide_kernel0(float* __restrict__ T_divide, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_divide[(((int)threadIdx.x))] = (placeholder[(((int)threadIdx.x))] / placeholder1[(0)]);
}

extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)threadIdx.x)))] * placeholder1[((((((int)blockIdx.x) * 2048) + (k_outer * 64)) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_dense[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[570];
  __shared__ float placeholder_shared[1152];
  compute[(0)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 570) {
      if (((int)threadIdx.x) < 12) {
        pad_temp_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)))] = (((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) % 285) / 57))) && (1 <= (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) % 57))) ? placeholder[(((((((rc_outer * 6272) + ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) / 285) * 3136)) + (((int)blockIdx.y) * 224)) + (((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) % 285) / 57) * 56)) + (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) % 57)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 569) {
      if (((int)threadIdx.x) < 12) {
        pad_temp_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= ((((int)blockIdx.y) * 4) + (((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1) % 285) / 57))) && (1 <= ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1) % 57))) ? placeholder[(((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1) / 285) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1) % 285) / 57) * 56)) + ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1) % 57)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 568) {
      if (((int)threadIdx.x) < 12) {
        pad_temp_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= ((((int)blockIdx.y) * 4) + (((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2) % 285) / 57))) && (1 <= ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2) % 57))) ? placeholder[(((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2) / 285) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2) % 285) / 57) * 56)) + ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2) % 57)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 64) {
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1152) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 1152)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 64) {
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1151) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 1152)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 64) {
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1150) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 1152)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)) + 2))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1149) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 1152)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1148) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 1152)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) < 1147) {
            if (((int)threadIdx.x) < 12) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 6)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 73728) + (((int)threadIdx.z) * 4608)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 1152)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)) + 2))];
            }
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 72))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[((((int)threadIdx.z) * 72))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[((((int)threadIdx.z) * 72))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[((((int)threadIdx.z) * 72))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 18))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 18))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 18))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 18))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 36))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 36))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 36))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 36))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 54))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 54))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 54))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 54))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 1))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 1))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 1))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 1))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 19))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 19))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 19))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 19))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 37))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 37))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 37))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 37))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 55))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 55))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 55))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 55))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 2))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 2))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 2))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 2))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 20))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 20))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 20))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 20))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 38))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 38))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 38))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 38))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 56))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 56))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 56))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 56))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 3))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 3))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 171))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 3))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 199))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 3))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 21))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 21))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 171))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 21))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 199))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 21))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 39))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 39))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 171))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 39))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 199))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 39))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 57))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 57))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 171))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 57))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 199))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 57))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 4))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 4))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 172))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 4))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 200))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 4))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 22))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 22))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 172))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 22))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 200))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 22))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 40))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 40))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 172))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 40))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 200))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 40))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 58))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 58))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 172))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 58))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 200))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 58))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 5))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 5))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 173))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 5))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 201))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 5))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 23))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 23))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 173))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 23))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 201))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 23))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 41))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 41))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 173))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 41))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 201))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 41))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 59))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 59))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 173))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 59))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 201))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 59))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 6))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 6))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 6))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 6))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 24))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 24))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 24))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 24))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 42))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 42))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 42))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 42))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 60))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 142))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 60))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 60))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 60))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 7))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 7))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 7))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 7))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 25))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 25))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 25))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 25))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 43))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 43))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 43))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 43))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 61))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 61))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 61))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 61))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 8))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 8))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 8))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 8))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 26))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 26))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 26))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 26))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 44))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 44))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 44))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 44))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 62))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 144))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 62))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 62))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 62))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 9))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 313))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 9))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 9))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 9))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 27))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 313))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 27))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 27))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 27))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 45))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 313))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 45))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 45))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 45))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 63))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 313))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 63))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 63))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 63))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 10))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 314))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 10))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 10))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 10))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 28))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 314))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 28))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 28))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 28))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 46))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 314))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 46))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 46))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 46))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 64))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 314))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 64))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 64))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 64))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 11))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 11))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 11))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 11))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 29))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 29))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 29))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 29))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 47))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 47))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 47))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 47))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 65))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 65))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 65))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 65))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 342))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 12))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 370))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 12))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 456))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 12))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 12))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 342))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 30))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 370))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 30))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 456))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 30))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 30))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 342))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 48))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 370))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 48))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 456))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 48))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 48))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 342))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 66))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 370))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 66))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 456))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 66))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 66))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 343))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 13))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 371))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 13))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 457))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 13))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 485))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 13))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 343))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 31))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 371))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 31))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 457))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 31))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 485))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 31))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 343))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 49))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 371))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 49))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 457))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 49))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 485))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 49))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 343))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 67))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 371))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 67))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 457))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 67))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 485))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 67))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 344))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 14))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 372))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 14))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 458))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 14))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 14))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 344))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 32))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 372))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 32))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 458))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 32))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 32))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 344))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 50))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 372))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 50))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 458))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 50))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 50))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 344))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 68))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 372))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 68))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 458))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 68))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 68))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 15))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 15))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 513))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 15))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 541))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 15))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 33))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 33))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 513))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 33))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 541))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 33))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 51))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 51))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 513))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 51))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 541))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 51))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 399))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 69))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 427))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 69))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 513))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 69))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 541))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 69))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 16))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 16))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 514))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 16))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 542))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 16))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 34))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 34))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 514))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 34))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 542))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 34))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 52))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 52))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 514))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 52))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 542))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 52))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 400))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 70))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 428))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 70))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 514))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 70))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 542))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 70))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 17))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 17))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 515))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 17))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 543))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 17))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 35))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 35))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 515))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 35))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 543))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 35))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 53))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 53))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 515))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 53))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 543))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 53))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 401))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 71))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 71))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 515))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 71))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 543))] * placeholder_shared[(((((int)threadIdx.z) * 72) + 71))]));
  }
  T_relu[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 14))] = max((compute[(8)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 42))] = max((compute[(9)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 784))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 798))] = max((compute[(10)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 812))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 826))] = max((compute[(11)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1568))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1582))] = max((compute[(12)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1596))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1610))] = max((compute[(13)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2352))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2366))] = max((compute[(14)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2380))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2394))] = max((compute[(15)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_10_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 56) * 3136)) + (((int)blockIdx.y) * 56)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 56)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 128) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 512) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 384))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 640))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 896))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 384))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 640))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 896))]));
    }
  }
  T_relu[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25088))] = max((compute[(2)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75264))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 100352))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 125440))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 40))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 150528))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 175616))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 56))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25116))] = max((compute[(3)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75292))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 100380))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 125468))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 40))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 150556))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 175644))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 56))]), 0.000000e+00f);
}

extern "C" __global__ void fused_mean_kernel1(float* __restrict__ T_divide, float* __restrict__ placeholder_red) {
  T_divide[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = (placeholder_red[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 2.040816e-02f);
}

extern "C" __global__ void fused_nn_max_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor_local[(0)] = max(tensor_local[(0)], (((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 3136) / 56) * 2) + rv0) < 112) && ((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 56) * 2) + rv1) < 112)) ? placeholder[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 56) * 224) + (rv0 * 112)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 56) * 2)) + rv1))] : -3.402823e+38f));
    }
  }
  tensor[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[256];
  #pragma unroll
  for (int yy_init = 0; yy_init < 2; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 2))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 28) + ((int)threadIdx.x)))] = placeholder[((((((rc_outer * 6272) + ((((int)threadIdx.z) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((int)threadIdx.z) & 1) * 28)) + ((int)threadIdx.x)))];
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.x) >> 3)) < 32) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) < 256) {
        if (((int)threadIdx.x) < 16) {
          placeholder_shared[(((((int)threadIdx.z) * 16) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_outer * 8)) + (((int)threadIdx.x) & 7)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int yy = 0; yy < 2; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 56) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[((((rc_inner * 56) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_max_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = -3.402823e+38f;
  for (int k1_outer = 0; k1_outer < 32; ++k1_outer) {
    if (((int)threadIdx.y) < 1) {
      if (((k1_outer * 32) + ((int)threadIdx.x)) < 1001) {
        placeholder_red_rf[(0)] = max(placeholder_red_rf[(0)], placeholder[((((((int)threadIdx.y) * 1001) + (k1_outer * 32)) + ((int)threadIdx.x)))]);
      }
    }
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
    placeholder_red[(((int)threadIdx.y))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[28];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 512)) + (rc_outer * 16)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((rc_inner * 56) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
    }
  }
  T_relu[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12544))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25088))] = max((compute[(14)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37632))] = max((compute[(21)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 2))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12546))] = max((compute[(8)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25090))] = max((compute[(15)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37634))] = max((compute[(22)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 4))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12548))] = max((compute[(9)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25092))] = max((compute[(16)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37636))] = max((compute[(23)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 6))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12550))] = max((compute[(10)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25094))] = max((compute[(17)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37638))] = max((compute[(24)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 8))] = max((compute[(4)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12552))] = max((compute[(11)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25096))] = max((compute[(18)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37640))] = max((compute[(25)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 10))] = max((compute[(5)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12554))] = max((compute[(12)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25098))] = max((compute[(19)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37642))] = max((compute[(26)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12))] = max((compute[(6)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 12556))] = max((compute[(13)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 25100))] = max((compute[(20)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 37644))] = max((compute[(27)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_9_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 56) * 3136)) + (((int)blockIdx.y) * 56)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 56)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 32) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 384))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 384))]));
    }
  }
  T_relu[(((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25088))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75264))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25116))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75292))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
}

extern "C" __global__ void fused_mean_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k2_k3_fused_outer = 0; k2_k3_fused_outer < 2; ++k2_k3_fused_outer) {
    if ((((k2_k3_fused_outer * 32) + ((int)threadIdx.x)) < 49) && (((k2_k3_fused_outer * 32) + ((int)threadIdx.x)) < 49)) {
      placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[(((((((int)blockIdx.x) * 1568) + (((int)threadIdx.y) * 49)) + (k2_k3_fused_outer * 32)) + ((int)threadIdx.x)))]);
    }
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

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[112];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 112) {
      if (((int)threadIdx.x) < 4) {
        pad_temp_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder[(((((rc_outer * 784) + ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) / 7) * 49)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) % 7)))];
      }
    }
    if ((((((int)threadIdx.x) * 3) >> 4) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 512) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((int)blockIdx.z) * 65536) + (((int)threadIdx.z) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 3)))];
        }
      }
    }
    if (((((((int)threadIdx.x) * 3) + 1) >> 4) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 511) {
        if (((int)threadIdx.x) < 5) {
          placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 65536) + (((int)threadIdx.z) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 3)) + 1))];
        }
      }
    }
    if (((((((int)threadIdx.x) * 3) + 2) >> 4) + ((int)threadIdx.z)) < 32) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 510) {
        if (((int)threadIdx.x) < 5) {
          placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 65536) + (((int)threadIdx.z) * 2048)) + (rc_outer * 16)) + (((int)threadIdx.x) * 3)) + 2))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
  }
  T_relu[(((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 24576))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 98304))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 122880))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      if (((((((int)threadIdx.x) & 15) >> 2) * 2) + ax2_inner) < 7) {
        if ((((((int)threadIdx.x) & 3) * 2) + ax3_inner) < 7) {
          T_relu[(((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[(((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 2) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 2) + eps) < 29)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2) + nu) < 29)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (eps * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + nu) - 29))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(10)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(11)] * -1.000000e+00f));
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(14)] * -1.000000e+00f));
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 100352) + (nu1 * 25088)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[8];
  __shared__ float placeholder_shared[1024];
  __shared__ float data_pack_shared[256];
  for (int co_c_init = 0; co_c_init < 4; ++co_c_init) {
    for (int p_c_init = 0; p_c_init < 2; ++p_c_init) {
      bgemm_local[(((co_c_init * 2) + p_c_init))] = 0.000000e+00f;
    }
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      placeholder_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1) {
      data_pack_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = data_pack[((((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      for (int co_c = 0; co_c < 4; ++co_c) {
        for (int p_c = 0; p_c < 2; ++p_c) {
          bgemm_local[(((co_c * 2) + p_c))] = (bgemm_local[(((co_c * 2) + p_c))] + (placeholder_shared[((((ci_inner * 64) + (((int)threadIdx.y) * 4)) + co_c))] * data_pack_shared[((((ci_inner * 16) + (((int)threadIdx.x) * 2)) + p_c))]));
        }
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 4; ++co_inner_inner_inner) {
    for (int p_inner_inner_inner = 0; p_inner_inner_inner < 2; ++p_inner_inner_inner) {
      bgemm[(((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 64)) + (co_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + p_inner_inner_inner))] = bgemm_local[(((co_inner_inner_inner * 2) + p_inner_inner_inner))];
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float placeholder_shared[512];
  __shared__ float data_pack_shared[1568];
  bgemm_local[(0)] = 0.000000e+00f;
  bgemm_local[(4)] = 0.000000e+00f;
  bgemm_local[(8)] = 0.000000e+00f;
  bgemm_local[(12)] = 0.000000e+00f;
  bgemm_local[(2)] = 0.000000e+00f;
  bgemm_local[(6)] = 0.000000e+00f;
  bgemm_local[(10)] = 0.000000e+00f;
  bgemm_local[(14)] = 0.000000e+00f;
  bgemm_local[(1)] = 0.000000e+00f;
  bgemm_local[(5)] = 0.000000e+00f;
  bgemm_local[(9)] = 0.000000e+00f;
  bgemm_local[(13)] = 0.000000e+00f;
  bgemm_local[(3)] = 0.000000e+00f;
  bgemm_local[(7)] = 0.000000e+00f;
  bgemm_local[(11)] = 0.000000e+00f;
  bgemm_local[(15)] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    placeholder_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) & 31)))];
    placeholder_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196))] = placeholder[((((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 4) & 31)))];
    if (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) < 120) {
      if (((int)threadIdx.y) < 3) {
        placeholder_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392))] = placeholder[((((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 8) & 31)))];
      }
    }
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 392))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 784))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 588))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 1176))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 784))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 1568))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 980))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 1960))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1176))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 2352))];
    data_pack_shared[((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1372))] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) / 98) * 196)) + (((int)blockIdx.x) * 98)) + (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) % 98)) + 2744))];
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      bgemm_local[(0)] = (bgemm_local[(0)] + (placeholder_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(4)] = (bgemm_local[(4)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(8)] = (bgemm_local[(8)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(12)] = (bgemm_local[(12)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(2)] = (bgemm_local[(2)] + (placeholder_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(6)] = (bgemm_local[(6)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(10)] = (bgemm_local[(10)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(14)] = (bgemm_local[(14)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(1)] = (bgemm_local[(1)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(5)] = (bgemm_local[(5)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(9)] = (bgemm_local[(9)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(13)] = (bgemm_local[(13)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)))]));
      bgemm_local[(3)] = (bgemm_local[(3)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(7)] = (bgemm_local[(7)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(11)] = (bgemm_local[(11)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
      bgemm_local[(15)] = (bgemm_local[(15)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25))] * data_pack_shared[((((ci_inner * 98) + ((int)threadIdx.x)) + 49))]));
    }
  }
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)))] = bgemm_local[(0)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1568))] = bgemm_local[(4)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3136))] = bgemm_local[(8)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4704))] = bgemm_local[(12)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 49))] = bgemm_local[(2)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1617))] = bgemm_local[(6)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3185))] = bgemm_local[(10)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4753))] = bgemm_local[(14)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 196))] = bgemm_local[(1)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1764))] = bgemm_local[(5)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3332))] = bgemm_local[(9)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4900))] = bgemm_local[(13)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 245))] = bgemm_local[(3)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1813))] = bgemm_local[(7)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3381))] = bgemm_local[(11)];
  bgemm[(((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4949))] = bgemm_local[(15)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = placeholder[((((((rc_outer * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4)) + ((int)threadIdx.y)) < 32) {
        if (((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 512) {
          if ((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 32) {
            if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 16) {
              placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (rc_outer * 16)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
    }
  }
  T_relu[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3136))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[14];
  __shared__ float pad_temp_shared[450];
  __shared__ float placeholder_shared[1152];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) / 5) + ((int)threadIdx.z)) < 30) {
      if (((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)) < 450) {
        if (((int)threadIdx.y) < 5) {
          pad_temp_shared[(((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)))] = (((1 <= (((int)threadIdx.z) % 15)) && (1 <= ((int)threadIdx.y))) ? placeholder[((((((((((int)threadIdx.z) / 30) * 100352) + (rc_outer * 392)) + (((((int)threadIdx.z) % 30) / 15) * 196)) + ((((int)threadIdx.z) % 15) * 14)) + (((int)threadIdx.y) * 3)) - 15))] : 0.000000e+00f);
        }
      }
    }
    if (((((((int)threadIdx.y) * 3) + 1) / 15) + ((int)threadIdx.z)) < 30) {
      if (((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)) < 449) {
        if (((int)threadIdx.y) < 5) {
          pad_temp_shared[((((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)) + 1))] = ((1 <= (((int)threadIdx.z) % 15)) ? placeholder[((((((((((int)threadIdx.z) / 30) * 100352) + (rc_outer * 392)) + (((((int)threadIdx.z) % 30) / 15) * 196)) + ((((int)threadIdx.z) % 15) * 14)) + (((int)threadIdx.y) * 3)) - 14))] : 0.000000e+00f);
        }
      }
    }
    if (((((((int)threadIdx.y) * 3) + 2) / 15) + ((int)threadIdx.z)) < 30) {
      if (((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)) < 448) {
        if (((int)threadIdx.y) < 5) {
          pad_temp_shared[((((((int)threadIdx.z) * 15) + (((int)threadIdx.y) * 3)) + 2))] = ((1 <= (((int)threadIdx.z) % 15)) ? placeholder[((((((((((int)threadIdx.z) / 30) * 100352) + (rc_outer * 392)) + (((((int)threadIdx.z) % 30) / 15) * 196)) + ((((int)threadIdx.z) % 15) * 14)) + (((int)threadIdx.y) * 3)) - 13))] : 0.000000e+00f);
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 3)) < 64) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 384) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1152) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)))] = placeholder1[((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.y) / 3) * 4608)) + (rc_outer * 18)) + ((((int)threadIdx.y) % 3) * 6)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 3)) < 64) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 384) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1151) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.y) / 3) * 4608)) + (rc_outer * 18)) + ((((int)threadIdx.y) % 3) * 6)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.y) / 3)) < 64) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 384) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1150) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.y) / 3) * 4608)) + (rc_outer * 18)) + ((((int)threadIdx.y) % 3) * 6)) + 2))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 383) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1149) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 6) * 4608)) + (rc_outer * 18)) + ((((((int)threadIdx.y) * 2) + 1) % 6) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 383) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1148) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 6) * 4608)) + (rc_outer * 18)) + ((((((int)threadIdx.y) * 2) + 1) % 6) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + 1) / 6)) < 64) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 128) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 383) {
          if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 1147) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 294912) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 6) * 4608)) + (rc_outer * 18)) + ((((((int)threadIdx.y) * 2) + 1) % 6) * 3)) + 2))];
            }
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 259))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 265))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 267))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 259))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 265))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 267))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 260))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 268))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 258))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 260))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 268))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 259))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 265))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 267))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 269))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 259))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 265))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 267))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 269))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
  }
  T_relu[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 2))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 3))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 4))] = max((compute[(4)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 5))] = max((compute[(5)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 6))] = max((compute[(6)] + placeholder2[(((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 49))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 50))] = max((compute[(8)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 51))] = max((compute[(9)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 52))] = max((compute[(10)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 53))] = max((compute[(11)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 54))] = max((compute[(12)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 55))] = max((compute[(13)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float placeholder_shared[128];
  __shared__ float data_pack_shared[392];
  #pragma unroll
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[(co_c_init)] = 0.000000e+00f;
    bgemm_local[((co_c_init + 8))] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) < 128) {
        placeholder_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 65536) + (ci_outer * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) & 15)))];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 49) + ((int)threadIdx.x)))] = data_pack[(((((((int)blockIdx.z) * 12544) + (ci_outer * 392)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 49)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    #pragma unroll
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      #pragma unroll
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[(co_c)] = (bgemm_local[(co_c)] + (placeholder_shared[(((ci_inner * 16) + co_c))] * data_pack_shared[(((ci_inner * 49) + ((int)threadIdx.x)))]));
        bgemm_local[((co_c + 8))] = (bgemm_local[((co_c + 8))] + (placeholder_shared[((((ci_inner * 16) + co_c) + 8))] * data_pack_shared[(((ci_inner * 49) + ((int)threadIdx.x)))]));
      }
    }
  }
  #pragma unroll
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x)))] = bgemm_local[(co_inner_inner_inner)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x)) + 392))] = bgemm_local[((co_inner_inner_inner + 8))];
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[36];
  float data_pack_local[36];
  for (int eps = 0; eps < 6; ++eps) {
    for (int nu = 0; nu < 6; ++nu) {
      d[(((eps * 6) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 4) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 196) / 14) * 4) + eps) < 57)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4) + nu) < 57)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 224) + (eps * 56)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4)) + nu) - 57))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(1)] * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(2)] * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(3)] * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(4)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(6)] * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(7)] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(8)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(9)] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(10)] * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(12)] * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(13)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(14)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(15)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(16)] * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(18)] * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(19)] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(20)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(21)] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(22)] * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(24)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(25)] * -1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(26)] * -2.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(27)] * 1.500000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(28)]);
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(1)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(2)] * -2.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(3)] * 5.000000e-01f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(4)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(7)] * -1.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(8)] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(9)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(10)] * -1.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(13)] * -2.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(14)] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(15)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(16)] * -2.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(19)] * 1.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(20)] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(21)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(22)] * 1.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(25)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(26)] * -2.500000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(27)] * 5.000000e-01f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(28)]);
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(2)] * 5.000000e-01f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(3)] * 2.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(4)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(7)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(8)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(9)] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(10)] * -1.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(13)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(14)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(15)] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(16)] * -2.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(19)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(20)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(2)] = (data_pack_local[(2)] + ((d[(21)] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(22)] * 1.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(25)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(26)] * 5.000000e-01f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(27)] * 2.500000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(28)]);
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(1)] * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(3)] * 2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(4)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(7)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(8)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(9)] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(10)] * -1.500000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(13)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(14)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(15)] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(16)] * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(19)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(20)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(21)] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(22)] * 1.500000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(25)] * -2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(27)] * 2.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(28)]);
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(1)] * 5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(3)] * -5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(4)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(7)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(8)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(9)] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(10)] * -1.500000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(13)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(14)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(15)] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(16)] * -2.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(19)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(20)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(21)] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(22)] * 1.500000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(25)] * 5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(27)] * -5.000000e-01f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(28)]);
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(1)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(2)] * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(3)] * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(4)] * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(5)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(7)] * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(8)] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(9)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(10)] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(11)] * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(13)] * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(14)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(15)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(16)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(17)] * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(19)] * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(20)] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(21)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(22)] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(23)] * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(25)]);
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(26)] * -1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(27)] * -2.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(29)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(6)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(7)] * -1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(8)] * -2.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(9)] * 1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(12)] * -2.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(13)] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(14)] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(15)] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(16)] * -2.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(18)] * 5.000000e-01f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(19)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(20)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + ((d[(21)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(22)] * 5.000000e-01f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(24)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(25)] * -1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(26)] * -2.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(27)] * 1.500000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(28)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(7)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(8)] * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(9)] * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(10)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(13)] * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(14)] * -2.500000e+00f) * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(15)] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(16)] * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(19)] * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(20)] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(21)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(22)] * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(25)]);
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(26)] * -2.500000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(27)] * 5.000000e-01f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(28)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(8)] * 5.000000e-01f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(9)] * 2.500000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(10)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(13)] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(14)] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(15)] * -2.500000e+00f) * 2.500000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(16)] * -2.500000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(19)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(20)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(8)] = (data_pack_local[(8)] + ((d[(21)] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(22)] * 5.000000e-01f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(25)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(26)] * 5.000000e-01f));
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(27)] * 2.500000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(28)]);
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(7)] * -2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(9)] * 2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(13)] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(14)] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(15)] * -2.500000e+00f) * 2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(16)] * -2.500000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(19)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(20)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + ((d[(21)] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(22)] * 5.000000e-01f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(25)] * -2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(27)] * 2.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(28)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(7)] * 5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(9)] * -5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(13)] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(14)] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(15)] * -2.500000e+00f) * -5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(16)] * -2.500000e+00f));
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(19)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(20)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(10)] = (data_pack_local[(10)] + ((d[(21)] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(22)] * 5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(25)] * 5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(10)] = (data_pack_local[(10)] + (d[(27)] * -5.000000e-01f));
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(28)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(8)] * -1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(9)] * -2.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(10)] * 1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(13)] * -2.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(14)] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(15)] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(16)] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(17)] * -2.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(19)] * 5.000000e-01f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(20)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(21)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + ((d[(22)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(23)] * 5.000000e-01f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(25)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(26)] * -1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(27)] * -2.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(29)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(7)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(8)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(9)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(12)] * 5.000000e-01f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(13)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(14)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(15)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(16)] * 5.000000e-01f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(18)] * 2.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(19)] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(20)] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(21)] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(22)] * 2.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(24)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(25)] * -1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(26)] * -2.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(27)] * 1.500000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(28)]);
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(8)] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(9)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(13)] * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(14)] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(15)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(16)] * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(19)] * 2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(20)] * 2.500000e+00f) * -2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(21)] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(22)] * 2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(25)]);
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(26)] * -2.500000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(27)] * 5.000000e-01f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(28)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(7)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(8)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(9)] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(13)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(14)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(15)] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(16)] * 5.000000e-01f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(19)] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(20)] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(14)] = (data_pack_local[(14)] + ((d[(21)] * 2.500000e+00f) * 2.500000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(22)] * 2.500000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(25)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(26)] * 5.000000e-01f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(27)] * 2.500000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(28)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(7)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(8)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(9)] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(13)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(14)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(15)] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(16)] * 5.000000e-01f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(19)] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(20)] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(21)] * 2.500000e+00f) * 2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(22)] * 2.500000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(25)] * -2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(27)] * 2.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(28)]);
  data_pack_local[(16)] = 0.000000e+00f;
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(7)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(8)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(9)] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(13)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(14)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(15)] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(16)] * 5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(19)] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(20)] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + ((d[(21)] * 2.500000e+00f) * -5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(22)] * 2.500000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(25)] * 5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(16)] = (data_pack_local[(16)] + (d[(27)] * -5.000000e-01f));
  data_pack_local[(16)] = (data_pack_local[(16)] + d[(28)]);
  data_pack_local[(17)] = 0.000000e+00f;
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(8)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(9)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(10)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(11)] * -1.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(13)] * 5.000000e-01f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(14)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(15)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(16)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(17)] * 5.000000e-01f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(19)] * 2.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(20)] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(21)] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + ((d[(22)] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(23)] * 2.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + d[(25)]);
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(26)] * -1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(27)] * -2.000000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(17)] = (data_pack_local[(17)] + d[(29)]);
  data_pack_local[(18)] = 0.000000e+00f;
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(6)] * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(7)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(8)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(9)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(10)] * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(12)] * -1.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(13)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(14)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(15)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(18)] * 2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(19)] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(20)] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + ((d[(21)] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(22)] * 2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + d[(24)]);
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(25)] * -1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(26)] * -2.000000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + (d[(27)] * 1.500000e+00f));
  data_pack_local[(18)] = (data_pack_local[(18)] + d[(28)]);
  data_pack_local[(19)] = 0.000000e+00f;
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(7)] * -2.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(8)] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(9)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(10)] * -2.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(14)] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(15)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(19)] * 2.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(20)] * 2.000000e+00f) * -2.500000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + ((d[(21)] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(22)] * 2.000000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + d[(25)]);
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(26)] * -2.500000e+00f));
  data_pack_local[(19)] = (data_pack_local[(19)] + (d[(27)] * 5.000000e-01f));
  data_pack_local[(19)] = (data_pack_local[(19)] + d[(28)]);
  data_pack_local[(20)] = 0.000000e+00f;
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(7)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(8)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(9)] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(10)] * -2.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(13)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(14)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(15)] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(19)] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(20)] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(20)] = (data_pack_local[(20)] + ((d[(21)] * 2.000000e+00f) * 2.500000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(22)] * 2.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(25)] * -1.000000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(26)] * 5.000000e-01f));
  data_pack_local[(20)] = (data_pack_local[(20)] + (d[(27)] * 2.500000e+00f));
  data_pack_local[(20)] = (data_pack_local[(20)] + d[(28)]);
  data_pack_local[(21)] = 0.000000e+00f;
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(7)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(8)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(9)] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(10)] * -2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(13)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(14)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(15)] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(19)] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(20)] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + ((d[(21)] * 2.000000e+00f) * 2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(22)] * 2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(25)] * -2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + (d[(27)] * 2.000000e+00f));
  data_pack_local[(21)] = (data_pack_local[(21)] + d[(28)]);
  data_pack_local[(22)] = 0.000000e+00f;
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(7)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(8)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(9)] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(10)] * -2.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(13)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(14)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(15)] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(19)] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(20)] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + ((d[(21)] * 2.000000e+00f) * -5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(22)] * 2.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(25)] * 5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(22)] = (data_pack_local[(22)] + (d[(27)] * -5.000000e-01f));
  data_pack_local[(22)] = (data_pack_local[(22)] + d[(28)]);
  data_pack_local[(23)] = 0.000000e+00f;
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(7)] * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(8)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(9)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(10)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(11)] * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(14)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(15)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(16)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(17)] * -1.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(19)] * 2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(20)] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(21)] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + ((d[(22)] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(23)] * 2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + d[(25)]);
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(26)] * -1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(27)] * -2.000000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(23)] = (data_pack_local[(23)] + d[(29)]);
  data_pack_local[(24)] = 0.000000e+00f;
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(6)] * 5.000000e-01f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(7)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(8)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(9)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(10)] * 5.000000e-01f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(12)] * -1.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(13)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(14)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(15)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(18)] * -5.000000e-01f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(19)] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(20)] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + ((d[(21)] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(22)] * -5.000000e-01f));
  data_pack_local[(24)] = (data_pack_local[(24)] + d[(24)]);
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(25)] * -1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(26)] * -2.000000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + (d[(27)] * 1.500000e+00f));
  data_pack_local[(24)] = (data_pack_local[(24)] + d[(28)]);
  data_pack_local[(25)] = 0.000000e+00f;
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(7)] * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(8)] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(9)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(10)] * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(14)] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(15)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(19)] * -5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(20)] * -5.000000e-01f) * -2.500000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + ((d[(21)] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(22)] * -5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + d[(25)]);
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(26)] * -2.500000e+00f));
  data_pack_local[(25)] = (data_pack_local[(25)] + (d[(27)] * 5.000000e-01f));
  data_pack_local[(25)] = (data_pack_local[(25)] + d[(28)]);
  data_pack_local[(26)] = 0.000000e+00f;
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(7)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(8)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(9)] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(10)] * 5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(13)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(14)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(15)] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(19)] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(20)] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + ((d[(21)] * -5.000000e-01f) * 2.500000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(22)] * -5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(25)] * -1.000000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(26)] * 5.000000e-01f));
  data_pack_local[(26)] = (data_pack_local[(26)] + (d[(27)] * 2.500000e+00f));
  data_pack_local[(26)] = (data_pack_local[(26)] + d[(28)]);
  data_pack_local[(27)] = 0.000000e+00f;
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(7)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(8)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(9)] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(10)] * 5.000000e-01f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(13)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(14)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(15)] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(19)] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(20)] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + ((d[(21)] * -5.000000e-01f) * 2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(22)] * -5.000000e-01f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(25)] * -2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + (d[(27)] * 2.000000e+00f));
  data_pack_local[(27)] = (data_pack_local[(27)] + d[(28)]);
  data_pack_local[(28)] = 0.000000e+00f;
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(7)] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(8)] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(9)] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(10)] * 5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(13)] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(14)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(15)] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(16)] * -1.000000e+00f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(19)] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(20)] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[(28)] = (data_pack_local[(28)] + ((d[(21)] * -5.000000e-01f) * -5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(22)] * -5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(25)] * 5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(26)] * -1.000000e+00f));
  data_pack_local[(28)] = (data_pack_local[(28)] + (d[(27)] * -5.000000e-01f));
  data_pack_local[(28)] = (data_pack_local[(28)] + d[(28)]);
  data_pack_local[(29)] = 0.000000e+00f;
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(7)] * 5.000000e-01f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(8)] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(9)] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(10)] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(11)] * 5.000000e-01f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(14)] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(15)] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(16)] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(17)] * -1.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(19)] * -5.000000e-01f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(20)] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(21)] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + ((d[(22)] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(23)] * -5.000000e-01f));
  data_pack_local[(29)] = (data_pack_local[(29)] + d[(25)]);
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(26)] * -1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(27)] * -2.000000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(29)] = (data_pack_local[(29)] + d[(29)]);
  data_pack_local[(30)] = 0.000000e+00f;
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(6)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(7)] * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(8)] * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(9)] * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(10)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(12)] * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(13)] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(14)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(15)] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(16)] * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(18)] * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(19)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(20)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(21)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(22)] * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(24)] * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(25)] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(26)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + ((d[(27)] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(30)]);
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(31)] * -1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(32)] * -2.000000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + (d[(33)] * 1.500000e+00f));
  data_pack_local[(30)] = (data_pack_local[(30)] + d[(34)]);
  data_pack_local[(31)] = 0.000000e+00f;
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(7)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(8)] * -2.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(9)] * 5.000000e-01f));
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(10)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(13)] * -1.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(14)] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(15)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(16)] * -1.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(19)] * -2.000000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(20)] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(21)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(22)] * -2.000000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(25)] * 1.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(26)] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + ((d[(27)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(31)]);
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(32)] * -2.500000e+00f));
  data_pack_local[(31)] = (data_pack_local[(31)] + (d[(33)] * 5.000000e-01f));
  data_pack_local[(31)] = (data_pack_local[(31)] + d[(34)]);
  data_pack_local[(32)] = 0.000000e+00f;
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(8)] * 5.000000e-01f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(9)] * 2.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + d[(10)]);
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(13)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(14)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(15)] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(16)] * -1.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(19)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(20)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(21)] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(22)] * -2.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(25)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(26)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(32)] = (data_pack_local[(32)] + ((d[(27)] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(31)] * -1.000000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(32)] * 5.000000e-01f));
  data_pack_local[(32)] = (data_pack_local[(32)] + (d[(33)] * 2.500000e+00f));
  data_pack_local[(32)] = (data_pack_local[(32)] + d[(34)]);
  data_pack_local[(33)] = 0.000000e+00f;
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(7)] * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(9)] * 2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + d[(10)]);
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(13)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(14)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(15)] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(16)] * -1.500000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(19)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(20)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(21)] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(22)] * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(25)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(26)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + ((d[(27)] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(31)] * -2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(32)] * -1.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + (d[(33)] * 2.000000e+00f));
  data_pack_local[(33)] = (data_pack_local[(33)] + d[(34)]);
  data_pack_local[(34)] = 0.000000e+00f;
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(7)] * 5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(9)] * -5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + d[(10)]);
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(13)] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(14)] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(15)] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(16)] * -1.500000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(19)] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(20)] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(21)] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(22)] * -2.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(25)] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(26)] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + ((d[(27)] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(28)] * 1.500000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(31)] * 5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(32)] * -1.000000e+00f));
  data_pack_local[(34)] = (data_pack_local[(34)] + (d[(33)] * -5.000000e-01f));
  data_pack_local[(34)] = (data_pack_local[(34)] + d[(34)]);
  data_pack_local[(35)] = 0.000000e+00f;
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(7)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(8)] * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(9)] * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(10)] * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(11)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(13)] * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(14)] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(15)] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(16)] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(17)] * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(19)] * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(20)] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(21)] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(22)] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(23)] * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(25)] * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(26)] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(27)] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + ((d[(28)] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(29)] * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(31)]);
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(32)] * -1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(33)] * -2.000000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + (d[(34)] * 1.500000e+00f));
  data_pack_local[(35)] = (data_pack_local[(35)] + d[(35)]);
  for (int eps1 = 0; eps1 < 6; ++eps1) {
    for (int nu1 = 0; nu1 < 6; ++nu1) {
      data_pack[(((((eps1 * 75264) + (nu1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 6) + nu1))];
    }
  }
}

extern "C" __global__ void fused_sum_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k1_outer = 0; k1_outer < 32; ++k1_outer) {
    if (((int)threadIdx.y) < 1) {
      if (((k1_outer * 32) + ((int)threadIdx.x)) < 1001) {
        placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[((((((int)threadIdx.y) * 1001) + (k1_outer * 32)) + ((int)threadIdx.x)))]);
      }
    }
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
    placeholder_red[(((int)threadIdx.y))] = red_buf0[(0)];
  }
}

extern "C" __global__ void fused_cast_kernel0(int64_t* __restrict__ T_cast, int* __restrict__ placeholder) {
  T_cast[(0)] = ((int64_t)placeholder[(0)]);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[32];
  __shared__ float placeholder_shared[256];
  __shared__ float data_pack_shared[1568];
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[(co_c_init)] = 0.000000e+00f;
    bgemm_local[((co_c_init + 16))] = 0.000000e+00f;
    bgemm_local[((co_c_init + 8))] = 0.000000e+00f;
    bgemm_local[((co_c_init + 24))] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) < 256) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2) + ((int)threadIdx.y)) < 3) {
          placeholder_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)))] = placeholder[((((((((int)blockIdx.z) * 4096) + (ci_outer * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) >> 5) * 64)) + (((int)blockIdx.y) * 32)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)) & 31)))];
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1) {
      data_pack_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)))] = data_pack[((((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 196)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[(co_c)] = (bgemm_local[(co_c)] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c))] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)))]));
        bgemm_local[((co_c + 16))] = (bgemm_local[((co_c + 16))] + (placeholder_shared[(((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16))] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)))]));
        bgemm_local[((co_c + 8))] = (bgemm_local[((co_c + 8))] + (placeholder_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c))] * data_pack_shared[((((ci_inner * 196) + ((int)threadIdx.x)) + 98))]));
        bgemm_local[((co_c + 24))] = (bgemm_local[((co_c + 24))] + (placeholder_shared[(((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16))] * data_pack_shared[((((ci_inner * 196) + ((int)threadIdx.x)) + 98))]));
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)))] = bgemm_local[(co_inner_inner_inner)];
    bgemm[(((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3136))] = bgemm_local[((co_inner_inner_inner + 16))];
    bgemm[(((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 98))] = bgemm_local[((co_inner_inner_inner + 8))];
    bgemm[(((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3234))] = bgemm_local[((co_inner_inner_inner + 24))];
  }
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[16];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))] * 5.000000e-01f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))] * -2.000000e+00f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 5.000000e-01f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -2.000000e+00f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 5.000000e-01f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -2.000000e+00f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * -2.000000e+00f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 5.000000e-01f));
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f));
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))] * 2.500000e-01f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))] * 4.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 2.500000e-01f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * 4.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 2.500000e-01f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * 4.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 4.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 2.500000e-01f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f));
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))] * 1.250000e-01f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))] * -8.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 1.250000e-01f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -8.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 1.250000e-01f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -8.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * -8.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 1.250000e-01f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))]);
  inverse[(4)] = 0.000000e+00f;
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))] * -1.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(4)] = (inverse[(4)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * 5.000000e-01f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 5.000000e-01f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))] * -2.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -2.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f));
  inverse[(4)] = (inverse[(4)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f));
  inverse[(5)] = 0.000000e+00f;
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 5.000000e-01f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -2.000000e+00f));
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(5)] = (inverse[(5)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 5.000000e-01f));
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -2.000000e+00f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f) * -1.000000e+00f));
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 5.000000e-01f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f) * 5.000000e-01f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f) * -2.000000e+00f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f) * -1.000000e+00f));
  inverse[(5)] = (inverse[(5)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -2.000000e+00f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f) * 5.000000e-01f));
  inverse[(5)] = (inverse[(5)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f) * -2.000000e+00f));
  inverse[(6)] = 0.000000e+00f;
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 2.500000e-01f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * 4.000000e+00f));
  inverse[(6)] = (inverse[(6)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(6)] = (inverse[(6)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 2.500000e-01f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * 4.000000e+00f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 5.000000e-01f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f) * 2.500000e-01f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f) * 4.000000e+00f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f));
  inverse[(6)] = (inverse[(6)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -2.000000e+00f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f) * 2.500000e-01f));
  inverse[(6)] = (inverse[(6)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f) * 4.000000e+00f));
  inverse[(7)] = 0.000000e+00f;
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 1.250000e-01f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -8.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))] * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 1.250000e-01f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -8.000000e+00f));
  inverse[(7)] = (inverse[(7)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 5.000000e-01f) * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 5.000000e-01f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 5.000000e-01f) * 1.250000e-01f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 5.000000e-01f) * -8.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))] * 5.000000e-01f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -2.000000e+00f) * -1.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -2.000000e+00f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -2.000000e+00f) * 1.250000e-01f));
  inverse[(7)] = (inverse[(7)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -2.000000e+00f) * -8.000000e+00f));
  inverse[(7)] = (inverse[(7)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))] * -2.000000e+00f));
  inverse[(8)] = 0.000000e+00f;
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(8)] = (inverse[(8)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * 2.500000e-01f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 2.500000e-01f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))] * 4.000000e+00f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * 4.000000e+00f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f));
  inverse[(8)] = (inverse[(8)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f));
  inverse[(9)] = 0.000000e+00f;
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(9)] = (inverse[(9)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 5.000000e-01f));
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -2.000000e+00f));
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(9)] = (inverse[(9)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 5.000000e-01f));
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -2.000000e+00f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f) * -1.000000e+00f));
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 2.500000e-01f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f) * 5.000000e-01f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f) * -2.000000e+00f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f) * -1.000000e+00f));
  inverse[(9)] = (inverse[(9)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * 4.000000e+00f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f) * 5.000000e-01f));
  inverse[(9)] = (inverse[(9)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f) * -2.000000e+00f));
  inverse[(10)] = 0.000000e+00f;
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * 4.000000e+00f));
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(10)] = (inverse[(10)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * 4.000000e+00f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f) * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f) * 4.000000e+00f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f));
  inverse[(10)] = (inverse[(10)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * 4.000000e+00f));
  inverse[(10)] = (inverse[(10)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f) * 2.500000e-01f));
  inverse[(10)] = (inverse[(10)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f) * 4.000000e+00f));
  inverse[(11)] = 0.000000e+00f;
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * 1.250000e-01f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -8.000000e+00f));
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 1.250000e-01f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -8.000000e+00f));
  inverse[(11)] = (inverse[(11)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 2.500000e-01f) * -1.000000e+00f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 2.500000e-01f));
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 2.500000e-01f) * 1.250000e-01f));
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 2.500000e-01f) * -8.000000e+00f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))] * 2.500000e-01f));
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * 4.000000e+00f) * -1.000000e+00f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * 4.000000e+00f));
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * 4.000000e+00f) * 1.250000e-01f));
  inverse[(11)] = (inverse[(11)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * 4.000000e+00f) * -8.000000e+00f));
  inverse[(11)] = (inverse[(11)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))] * 4.000000e+00f));
  inverse[(12)] = 0.000000e+00f;
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))] * -1.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * 1.250000e-01f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 1.250000e-01f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))] * -8.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -8.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f));
  inverse[(12)] = (inverse[(12)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f));
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))]);
  inverse[(12)] = (inverse[(12)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))]);
  inverse[(13)] = 0.000000e+00f;
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 5.000000e-01f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -2.000000e+00f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 5.000000e-01f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -2.000000e+00f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f) * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 1.250000e-01f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f) * 5.000000e-01f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f) * -2.000000e+00f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f) * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -8.000000e+00f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f) * 5.000000e-01f));
  inverse[(13)] = (inverse[(13)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f) * -2.000000e+00f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))] * -1.000000e+00f));
  inverse[(13)] = (inverse[(13)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))] * 5.000000e-01f));
  inverse[(13)] = (inverse[(13)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))] * -2.000000e+00f));
  inverse[(14)] = 0.000000e+00f;
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 2.500000e-01f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * 4.000000e+00f));
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 2.500000e-01f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * 4.000000e+00f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 1.250000e-01f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f) * 2.500000e-01f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f) * 4.000000e+00f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -8.000000e+00f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f) * 2.500000e-01f));
  inverse[(14)] = (inverse[(14)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f) * 4.000000e+00f));
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))]);
  inverse[(14)] = (inverse[(14)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))] * 2.500000e-01f));
  inverse[(14)] = (inverse[(14)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))] * 4.000000e+00f));
  inverse[(15)] = 0.000000e+00f;
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f) * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))] * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))] * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))] * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248))]);
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336))] * 1.250000e-01f) * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))] * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424))] * 1.250000e-01f) * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))] * 1.250000e-01f) * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512))] * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600))] * -8.000000e+00f) * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688))] * -8.000000e+00f) * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))] * -8.000000e+00f) * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776))] * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864))] * -1.000000e+00f));
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408))]);
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952))] * 1.250000e-01f));
  inverse[(15)] = (inverse[(15)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496))] * -8.000000e+00f));
  inverse[(15)] = (inverse[(15)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 439040))]);
  for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner))] = max((inverse[(((ax2_inner * 4) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[104];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 104) {
      if (((int)threadIdx.x) < 4) {
        pad_temp_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder[(((((rc_outer * 1568) + ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) / 13) * 196)) + (((int)blockIdx.y) * 28)) + (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) % 13)))];
      }
    }
    placeholder_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)))] = placeholder1[((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 4096)) + (((((int)threadIdx.x) * 5) >> 3) * 1024)) + (rc_outer * 8)) + ((((int)threadIdx.x) * 5) & 7)))];
    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + 1) & 7)))];
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1022) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + 2) & 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1021) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + 3) & 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 1020) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + 4) & 7)))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 512))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 768))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 513))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 769))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 514))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 770))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 515))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 771))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 516))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 772))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 517))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 773))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 518))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 774))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 519))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 775))]));
  }
  T_add[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)) + 1568))] = (compute[(1)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)) + 3136))] = (compute[(2)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)) + 4704))] = (compute[(3)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[14];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[512];
  for (int xx_init = 0; xx_init < 7; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
    compute[((xx_init + 7))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 32) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if (((((int)threadIdx.y) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.y) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 512)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[(xx)] = (compute[(xx)] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + xx))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((xx + 7))] = (compute[((xx + 7))] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + xx))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      }
    }
  }
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 7; ++ax3_inner_inner_inner) {
    T_relu[(((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ax3_inner_inner_inner))] = max(((compute[(ax3_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]) + placeholder3[(((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ax3_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ax3_inner_inner_inner) + 784))] = max(((compute[((ax3_inner_inner_inner + 7))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]) + placeholder3[((((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ax3_inner_inner_inner) + 784))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[1024];
  for (int yy_init = 0; yy_init < 4; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 28) * 56)) + (((int)blockIdx.x) * 28)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int yy = 0; yy < 4; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
    T_add[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)))] = (compute[(ax2_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]);
    T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50176))] = (compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]);
    T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 100352))] = (compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]);
    T_add[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 150528))] = (compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 32) {
        if (((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if ((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
              placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 1024)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 384))]));
    }
  }
  T_relu[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3136))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4704))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps) < 15)) && (1 <= (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2) + nu))) && ((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2) + nu) < 15)) ? placeholder[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (eps * 14)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2)) + nu) - 15))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(10)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(11)] * -1.000000e+00f));
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(14)] * -1.000000e+00f));
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 50176) + (nu1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[28];
  __shared__ float pad_temp_shared[440];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 14) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 440) {
        if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 14) {
          pad_temp_shared[((((((int)threadIdx.z) * 14) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((rc_outer * 25088) + (((((((int)threadIdx.z) * 14) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 55) * 3136)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.z) * 14) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)))];
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.x) * 256)) + (rc_outer * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 8))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 24))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 24))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 24))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 40))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 40))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 40))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
    }
  }
  T_add[(((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25088))] = (compute[(7)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50176))] = (compute[(14)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75264))] = (compute[(21)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 4))] = (compute[(1)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25092))] = (compute[(8)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50180))] = (compute[(15)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75268))] = (compute[(22)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 8))] = (compute[(2)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25096))] = (compute[(9)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50184))] = (compute[(16)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75272))] = (compute[(23)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 12))] = (compute[(3)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25100))] = (compute[(10)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50188))] = (compute[(17)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75276))] = (compute[(24)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 16))] = (compute[(4)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25104))] = (compute[(11)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50192))] = (compute[(18)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75280))] = (compute[(25)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 20))] = (compute[(5)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25108))] = (compute[(12)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50196))] = (compute[(19)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75284))] = (compute[(26)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 24))] = (compute[(6)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25112))] = (compute[(13)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 50200))] = (compute[(20)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 75288))] = (compute[(27)] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]);
}

extern "C" __global__ void fused_argmax_kernel0(float* __restrict__ placeholder, int* __restrict__ placeholder_red) {
  int placeholder_red_temp_rf_v0[1];
  float placeholder_red_temp_rf_v1[1];
  int red_buf0[1];
  float red_buf1[1];
  placeholder_red_temp_rf_v0[(0)] = -1;
  placeholder_red_temp_rf_v1[(0)] = -3.402823e+38f;
  for (int k1_outer = 0; k1_outer < 32; ++k1_outer) {
    if (((int)threadIdx.y) < 1) {
      if (((k1_outer * 32) + ((int)threadIdx.x)) < 1001) {
        placeholder_red_temp_rf_v0[(0)] = ((placeholder[((((((int)threadIdx.y) * 1001) + (k1_outer * 32)) + ((int)threadIdx.x)))] <= placeholder_red_temp_rf_v1[(0)]) ? placeholder_red_temp_rf_v0[(0)] : ((k1_outer * 32) + ((int)threadIdx.x)));
        placeholder_red_temp_rf_v1[(0)] = ((placeholder[((((((int)threadIdx.y) * 1001) + (k1_outer * 32)) + ((int)threadIdx.x)))] <= placeholder_red_temp_rf_v1[(0)]) ? placeholder_red_temp_rf_v1[(0)] : placeholder[((((((int)threadIdx.y) * 1001) + (k1_outer * 32)) + ((int)threadIdx.x)))]);
      }
    }
  }
  uint mask[1];
  float t1[1];
  int t0[1];
  red_buf0[(0)] = placeholder_red_temp_rf_v0[(0)];
  red_buf1[(0)] = placeholder_red_temp_rf_v1[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  t1[(0)] = __shfl_down_sync(mask[(0)], red_buf1[(0)], 16, 32);
  red_buf0[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf0[(0)] : t0[(0)]);
  red_buf1[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf1[(0)] : t1[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  t1[(0)] = __shfl_down_sync(mask[(0)], red_buf1[(0)], 8, 32);
  red_buf0[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf0[(0)] : t0[(0)]);
  red_buf1[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf1[(0)] : t1[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  t1[(0)] = __shfl_down_sync(mask[(0)], red_buf1[(0)], 4, 32);
  red_buf0[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf0[(0)] : t0[(0)]);
  red_buf1[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf1[(0)] : t1[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  t1[(0)] = __shfl_down_sync(mask[(0)], red_buf1[(0)], 2, 32);
  red_buf0[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf0[(0)] : t0[(0)]);
  red_buf1[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf1[(0)] : t1[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  t1[(0)] = __shfl_down_sync(mask[(0)], red_buf1[(0)], 1, 32);
  red_buf0[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf0[(0)] : t0[(0)]);
  red_buf1[(0)] = ((t1[(0)] <= red_buf1[(0)]) ? red_buf1[(0)] : t1[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  red_buf1[(0)] = __shfl_sync(mask[(0)], red_buf1[(0)], 0, 32);
  if (((int)threadIdx.y) < 1) {
    if (((int)threadIdx.x) == 0) {
      placeholder_red[(((int)threadIdx.y))] = red_buf0[(0)];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[16];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[1024];
  for (int yy_init = 0; yy_init < 4; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 28) + ((int)threadIdx.x)))] = placeholder[((((((rc_outer * 6272) + ((((int)threadIdx.z) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + ((((int)threadIdx.z) & 3) * 28)) + ((int)threadIdx.x)))];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3)) < 128) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1024) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3) * 128)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int yy = 0; yy < 4; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 512))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 768))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
    T_relu[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = max(((compute[(ax2_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 128) + ((int)threadIdx.z)))]) + placeholder3[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25088))] = max(((compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 32))]) + placeholder3[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25088))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 50176))] = max(((compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 64))]) + placeholder3[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 50176))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 75264))] = max(((compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((((int)blockIdx.z) * 128) + ((int)threadIdx.z)) + 96))]) + placeholder3[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 75264))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_11_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[56];
  __shared__ float pad_temp_shared[229];
  __shared__ float placeholder_shared[448];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 14))] = 0.000000e+00f;
    compute[((ff_init + 28))] = 0.000000e+00f;
    compute[((ff_init + 42))] = 0.000000e+00f;
    compute[((ff_init + 2))] = 0.000000e+00f;
    compute[((ff_init + 16))] = 0.000000e+00f;
    compute[((ff_init + 30))] = 0.000000e+00f;
    compute[((ff_init + 44))] = 0.000000e+00f;
    compute[((ff_init + 4))] = 0.000000e+00f;
    compute[((ff_init + 18))] = 0.000000e+00f;
    compute[((ff_init + 32))] = 0.000000e+00f;
    compute[((ff_init + 46))] = 0.000000e+00f;
    compute[((ff_init + 6))] = 0.000000e+00f;
    compute[((ff_init + 20))] = 0.000000e+00f;
    compute[((ff_init + 34))] = 0.000000e+00f;
    compute[((ff_init + 48))] = 0.000000e+00f;
    compute[((ff_init + 8))] = 0.000000e+00f;
    compute[((ff_init + 22))] = 0.000000e+00f;
    compute[((ff_init + 36))] = 0.000000e+00f;
    compute[((ff_init + 50))] = 0.000000e+00f;
    compute[((ff_init + 10))] = 0.000000e+00f;
    compute[((ff_init + 24))] = 0.000000e+00f;
    compute[((ff_init + 38))] = 0.000000e+00f;
    compute[((ff_init + 52))] = 0.000000e+00f;
    compute[((ff_init + 12))] = 0.000000e+00f;
    compute[((ff_init + 26))] = 0.000000e+00f;
    compute[((ff_init + 40))] = 0.000000e+00f;
    compute[((ff_init + 54))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 7; ++ry_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 229) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 29) {
            pad_temp_shared[((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((3 <= ((((int)blockIdx.y) * 2) + ry_outer)) && (((((int)blockIdx.y) * 2) + ry_outer) < 227)) && (3 <= (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))) && ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 227)) ? placeholder[((((((((rc_outer * 50176) + (((int)blockIdx.y) * 448)) + (ry_outer * 224)) + (((int)threadIdx.z) * 29)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) - 675))] : 0.000000e+00f);
          }
        }
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
        if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 7)) < 64) {
          if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 448) {
            if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 56) {
              placeholder_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)threadIdx.z) * 1176) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 7) * 147)) + (rc_outer * 49)) + (ry_outer * 7)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 7)))];
            }
          }
        }
      }
      __syncthreads();
      for (int rx_inner = 0; rx_inner < 7; ++rx_inner) {
        for (int ff = 0; ff < 2; ++ff) {
          compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 14))] = (compute[((ff + 14))] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 28))] = (compute[((ff + 28))] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 42))] = (compute[((ff + 42))] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 2))] = (compute[((ff + 2))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 32))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 16))] = (compute[((ff + 16))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 32))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 30))] = (compute[((ff + 30))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 32))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 44))] = (compute[((ff + 44))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 32))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 4))] = (compute[((ff + 4))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 64))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 18))] = (compute[((ff + 18))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 64))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 32))] = (compute[((ff + 32))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 64))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 46))] = (compute[((ff + 46))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 64))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 6))] = (compute[((ff + 6))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 96))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 20))] = (compute[((ff + 20))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 96))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 34))] = (compute[((ff + 34))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 96))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 48))] = (compute[((ff + 48))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 96))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 8))] = (compute[((ff + 8))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 128))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 22))] = (compute[((ff + 22))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 128))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 36))] = (compute[((ff + 36))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 128))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 50))] = (compute[((ff + 50))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 128))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 10))] = (compute[((ff + 10))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 160))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 24))] = (compute[((ff + 24))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 160))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 38))] = (compute[((ff + 38))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 160))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 52))] = (compute[((ff + 52))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 160))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
          compute[((ff + 12))] = (compute[((ff + 12))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 192))] * placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
          compute[((ff + 26))] = (compute[((ff + 26))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 192))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112))]));
          compute[((ff + 40))] = (compute[((ff + 40))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 192))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224))]));
          compute[((ff + 54))] = (compute[((ff + 54))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 192))] * placeholder_shared[(((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336))]));
        }
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)))] = max((compute[(ax1_inner_inner_inner)] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200704))] = max((compute[((ax1_inner_inner_inner + 14))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401408))] = max((compute[((ax1_inner_inner_inner + 28))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602112))] = max((compute[((ax1_inner_inner_inner + 42))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 16))] = max((compute[((ax1_inner_inner_inner + 2))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200720))] = max((compute[((ax1_inner_inner_inner + 16))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401424))] = max((compute[((ax1_inner_inner_inner + 30))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602128))] = max((compute[((ax1_inner_inner_inner + 44))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 32))] = max((compute[((ax1_inner_inner_inner + 4))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200736))] = max((compute[((ax1_inner_inner_inner + 18))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401440))] = max((compute[((ax1_inner_inner_inner + 32))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602144))] = max((compute[((ax1_inner_inner_inner + 46))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 48))] = max((compute[((ax1_inner_inner_inner + 6))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200752))] = max((compute[((ax1_inner_inner_inner + 20))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401456))] = max((compute[((ax1_inner_inner_inner + 34))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602160))] = max((compute[((ax1_inner_inner_inner + 48))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 64))] = max((compute[((ax1_inner_inner_inner + 8))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200768))] = max((compute[((ax1_inner_inner_inner + 22))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401472))] = max((compute[((ax1_inner_inner_inner + 36))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602176))] = max((compute[((ax1_inner_inner_inner + 50))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 80))] = max((compute[((ax1_inner_inner_inner + 10))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200784))] = max((compute[((ax1_inner_inner_inner + 24))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401488))] = max((compute[((ax1_inner_inner_inner + 38))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602192))] = max((compute[((ax1_inner_inner_inner + 52))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 96))] = max((compute[((ax1_inner_inner_inner + 12))] + placeholder2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200800))] = max((compute[((ax1_inner_inner_inner + 26))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401504))] = max((compute[((ax1_inner_inner_inner + 40))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[((((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602208))] = max((compute[((ax1_inner_inner_inner + 54))] + placeholder2[((((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[256];
  for (int yy_init = 0; yy_init < 2; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
    compute[((yy_init + 2))] = 0.000000e+00f;
    compute[((yy_init + 6))] = 0.000000e+00f;
    compute[((yy_init + 10))] = 0.000000e+00f;
    compute[((yy_init + 14))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 28) * 56)) + (((int)blockIdx.x) * 28)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 32) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 256) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 256)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int yy = 0; yy < 2; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 25088))] = max((compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50176))] = max((compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 75264))] = max((compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 112))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 25200))] = max((compute[((ax2_inner_inner_inner + 6))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50288))] = max((compute[((ax2_inner_inner_inner + 10))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 75376))] = max((compute[((ax2_inner_inner_inner + 14))] + placeholder2[((((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_squeeze_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner))] = max((inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_subtract_exp_kernel0(float* __restrict__ T_exp, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_exp[(((int)threadIdx.x))] = __expf((placeholder[(((int)threadIdx.x))] - placeholder1[(0)]));
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[56];
  __shared__ float pad_temp_shared[435];
  __shared__ float placeholder_shared[576];
  compute[(0)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(49)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(50)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(51)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(52)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(53)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(54)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(48)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(55)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) < 435) {
      pad_temp_shared[(((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)))] = (((15 <= ((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4))) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) % 15)))) ? placeholder[((((((rc_outer * 784) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) / 15) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) % 15)) - 29))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) < 434) {
      pad_temp_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 1))] = (((14 <= ((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 1) % 15)))) ? placeholder[((((((rc_outer * 784) + (((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 1) / 15) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 1) % 15)) - 29))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) < 433) {
      pad_temp_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 2))] = (((13 <= ((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 2) % 15)))) ? placeholder[((((((rc_outer * 784) + (((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 2) / 15) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 2) % 15)) - 29))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) < 432) {
      pad_temp_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 3))] = (((12 <= ((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 3) % 15)))) ? placeholder[((((((rc_outer * 784) + (((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 3) / 15) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.y) * 4)) + 3) % 15)) - 29))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 192) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 576) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + (((((int)threadIdx.y) * 2) / 3) * 2304)) + (rc_outer * 9)) + (((((int)threadIdx.y) * 2) % 3) * 3)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 192) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 575) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 1))] = placeholder1[(((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + (((((int)threadIdx.y) * 2) / 3) * 2304)) + (rc_outer * 9)) + (((((int)threadIdx.y) * 2) % 3) * 3)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.y) * 2) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 192) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 574) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 2))] = placeholder1[(((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + (((((int)threadIdx.y) * 2) / 3) * 2304)) + (rc_outer * 9)) + (((((int)threadIdx.y) * 2) % 3) * 3)) + 2))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 191) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 573) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 3) * 2304)) + (rc_outer * 9)) + ((((((int)threadIdx.y) * 2) + 1) % 3) * 3)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 191) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 572) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 4))] = placeholder1[(((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 3) * 2304)) + (rc_outer * 9)) + ((((((int)threadIdx.y) * 2) + 1) % 3) * 3)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 2) + 1) / 3)) < 64) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 2)) < 191) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) < 571) {
          if (((int)threadIdx.y) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 6)) + 5))] = placeholder1[(((((((((int)blockIdx.z) * 147456) + (((int)threadIdx.z) * 9216)) + ((((((int)threadIdx.y) * 2) + 1) / 3) * 2304)) + (rc_outer * 9)) + ((((((int)threadIdx.y) * 2) + 1) % 3) * 3)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.y) * 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 210))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[((((int)threadIdx.z) * 9))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 144))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 288))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 432))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 213))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 213))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 213))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 213))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 215))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 215))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 215))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 215))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 217))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 217))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 217))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 217))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 219))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 219))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 219))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 219))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 223))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 1))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 223))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 145))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 223))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 289))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 223))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 433))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 214))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 218))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 222))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 2))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 146))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 290))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 434))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 3))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 147))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 291))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 435))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 228))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 230))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 236))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 4))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 148))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 292))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 436))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 229))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 235))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 237))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 5))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 149))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 293))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 239))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 437))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 6))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 150))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 294))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 438))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 245))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 37))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 39))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 249))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 41))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 251))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 43))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 7))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 151))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 295))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 439))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 244))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 36))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 246))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 38))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 248))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 40))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 250))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 8))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 152))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 296))]));
    compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.y) * 30) + 254))] * placeholder_shared[(((((int)threadIdx.z) * 9) + 440))]));
  }
  T_relu[(((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3136))] = max((compute[(14)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6272))] = max((compute[(28)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9408))] = max((compute[(42)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 98))] = max((compute[(7)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3234))] = max((compute[(21)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6370))] = max((compute[(35)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9506))] = max((compute[(49)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3137))] = max((compute[(15)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6273))] = max((compute[(29)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9409))] = max((compute[(43)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 99))] = max((compute[(8)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3235))] = max((compute[(22)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6371))] = max((compute[(36)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9507))] = max((compute[(50)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 2))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3138))] = max((compute[(16)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6274))] = max((compute[(30)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9410))] = max((compute[(44)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 100))] = max((compute[(9)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3236))] = max((compute[(23)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6372))] = max((compute[(37)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9508))] = max((compute[(51)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3139))] = max((compute[(17)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6275))] = max((compute[(31)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9411))] = max((compute[(45)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 101))] = max((compute[(10)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3237))] = max((compute[(24)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6373))] = max((compute[(38)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9509))] = max((compute[(52)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 4))] = max((compute[(4)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3140))] = max((compute[(18)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6276))] = max((compute[(32)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9412))] = max((compute[(46)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 102))] = max((compute[(11)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3238))] = max((compute[(25)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6374))] = max((compute[(39)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9510))] = max((compute[(53)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 5))] = max((compute[(5)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3141))] = max((compute[(19)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6277))] = max((compute[(33)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9413))] = max((compute[(47)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 103))] = max((compute[(12)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3239))] = max((compute[(26)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6375))] = max((compute[(40)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9511))] = max((compute[(54)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6))] = max((compute[(6)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3142))] = max((compute[(20)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6278))] = max((compute[(34)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9414))] = max((compute[(48)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 104))] = max((compute[(13)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 3240))] = max((compute[(27)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 6376))] = max((compute[(41)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + 9512))] = max((compute[(55)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[(((eps * 4) + nu))] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && (((((((int)threadIdx.x) & 15) >> 2) * 2) + eps) < 8)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && ((((((int)threadIdx.x) & 3) * 2) + nu) < 8)) ? placeholder[((((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8))] : 0.000000e+00f);
    }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(10)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(11)] * -1.000000e+00f));
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(14)] * -1.000000e+00f));
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
    for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 32768) + (nu1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[648];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((((int)threadIdx.x) * 6) / 27) * 28)) + ((((int)threadIdx.x) * 6) % 27)))];
    pad_temp_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) + 1))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 6) + 1) / 27) * 28)) + (((((int)threadIdx.x) * 6) + 1) % 27)))];
    pad_temp_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) + 2))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 6) + 2) / 27) * 28)) + (((((int)threadIdx.x) * 6) + 2) % 27)))];
    if (((((((int)threadIdx.x) * 6) + 3) / 81) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.x) * 6) + 3) / 27)) < 24) {
        if (((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) < 645) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) + 3))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 6) + 3) / 27) * 28)) + (((((int)threadIdx.x) * 6) + 3) % 27)))];
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 6) + 4) / 81) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.x) * 6) + 4) / 27)) < 24) {
        if (((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) < 644) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) + 4))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 6) + 4) / 27) * 28)) + (((((int)threadIdx.x) * 6) + 4) % 27)))];
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 6) + 5) / 81) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.x) * 6) + 5) / 27)) < 24) {
        if (((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) < 643) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.x) * 6)) + 5))] = placeholder[((((((rc_outer * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 6) + 5) / 27) * 28)) + (((((int)threadIdx.x) * 6) + 5) % 27)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 3) >> 3)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 256) {
        if (((int)threadIdx.x) < 11) {
          placeholder_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((((int)threadIdx.x) * 3) >> 3) * 512)) + (rc_outer * 8)) + ((((int)threadIdx.x) * 3) & 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 1) >> 3)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 255) {
        if (((int)threadIdx.x) < 11) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 1) >> 3) * 512)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + 1) & 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 2) >> 3)) < 32) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 254) {
        if (((int)threadIdx.x) < 10) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 2) >> 3) * 512)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + 2) & 7)))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 54))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 216))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 243))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 324))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 324))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 324))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 324))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 459))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 459))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 459))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 459))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 540))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 540))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 486))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 540))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 540))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 567))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 567))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 621))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 621))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 567))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 567))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 621))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 621))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
  }
  T_add[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 3136))] = (compute[(4)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 16))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = (compute[(2)] + placeholder2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 3150))] = (compute[(6)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 16))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 196))] = (compute[(1)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 3332))] = (compute[(5)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 17))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 210))] = (compute[(3)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 1))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 3346))] = (compute[(7)] + placeholder2[((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + 17))]);
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[8];
  __shared__ float pad_temp_shared[224];
  __shared__ float placeholder_shared[512];
  #pragma unroll
  for (int xx_init = 0; xx_init < 2; ++xx_init) {
    compute[(xx_init)] = 0.000000e+00f;
    compute[((xx_init + 2))] = 0.000000e+00f;
    compute[((xx_init + 4))] = 0.000000e+00f;
    compute[((xx_init + 6))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 14) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder[(((((((rc_outer * 1568) + ((((int)threadIdx.z) >> 1) * 196)) + (((int)blockIdx.y) * 28)) + ((((int)threadIdx.z) & 1) * 14)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3)) < 64) {
        if (((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 512) {
          if ((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 32) {
            if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 16) {
              placeholder_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3) * 256)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7)))];
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int xx = 0; xx < 2; ++xx) {
        compute[(xx)] = (compute[(xx)] + (pad_temp_shared[(((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + xx))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((xx + 2))] = (compute[((xx + 2))] + (pad_temp_shared[(((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + xx))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute[((xx + 4))] = (compute[((xx + 4))] + (pad_temp_shared[(((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + xx))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute[((xx + 6))] = (compute[((xx + 6))] + (pad_temp_shared[(((((rc_inner * 28) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + xx))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 384))]));
      }
    }
  }
  #pragma unroll
  for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
    T_relu[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))] = max(((compute[(ax3_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]) + placeholder3[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 3136))] = max(((compute[((ax3_inner_inner_inner + 2))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]) + placeholder3[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 3136))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))] = max(((compute[((ax3_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]) + placeholder3[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 6272))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 9408))] = max(((compute[((ax3_inner_inner_inner + 6))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]) + placeholder3[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner) + 9408))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float compute[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[1024];
  for (int yy_init = 0; yy_init < 4; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 28) * 56)) + (((int)blockIdx.x) * 28)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int yy = 0; yy < 4; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)))] = max(((compute[(ax2_inner_inner_inner)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]) + placeholder3[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50176))] = max(((compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]) + placeholder3[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50176))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 100352))] = max(((compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]) + placeholder3[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 100352))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 150528))] = max(((compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]) + placeholder3[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 150528))]), 0.000000e+00f);
  }
}

