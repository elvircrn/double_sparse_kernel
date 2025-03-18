/*
 * Copyright (C) DoubleSparse Kernel.2024 Elvir Crncevic (elvircrn@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Torch takes too damn long to compile!

#include <torch/torch.h>

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

static __device__ __host__ __forceinline__ int updiv(int x, int y) { return (x + y - 1) / y; }


static constexpr u32 SHARED_OFFSET = 32;
static constexpr u32 WARP_SIZE = 32;
static constexpr u32 FULL_MASK = 0xFFFFFFFFu;

__device__ void printf_half2(const half2 &_x) {
  auto x = __half22float2(_x);
  printf("%f %f\n", x.x, x.y);
}

__device__ void printf_float2(float2 x) { printf("%f %f\n", x.x, x.y); }

#define DEVICE_INLINE __forceinline__ __device__

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

#define INT2_TO_HALF2(v)                                                       \
  make_half2(__int2half_rd((v) & 0b111), __int2half_rd(((v) >> 3) & 0b111))

template <class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))
#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))

template <class Scalar_t> __host__ __device__ auto vectorize(Scalar_t *ptr) {
  if constexpr (std::is_same<Scalar_t, float>::value) {
    return reinterpret_cast<float2 *>(ptr);
  } else if constexpr (std::is_same<Scalar_t, half>::value) {
    return reinterpret_cast<half2 *>(ptr);
  } else {
    return ptr;
  }
}

template <class Vec_t> __host__ __device__ auto scalarize(void *ptr) {
  if constexpr (std::is_same<Vec_t, float>::value ||
                std::is_same<Vec_t, float2>::value) {
    return reinterpret_cast<float *>(ptr);
  } else if constexpr (std::is_same<Vec_t, half2>::value) {
    return reinterpret_cast<half *>(ptr);
  } else {
    return ptr;
  }
}

template <class T> DEVICE_INLINE u16 get_col(T m) {
  return static_cast<u16>(m & T((1u << 16u) - 1u));
}

DEVICE_INLINE half get_val(u32 m) {
  u16 _v = m >> 16u;
  half v = *reinterpret_cast<half *>(&_v);
  return v;
}

// Wait until at most `n` async copy stages are still pending.
template <int n> DEVICE_INLINE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

DEVICE_INLINE void cp_async(half2 *__restrict__ dst,
                            const half2 *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s_dst),
               "l"(src));
}

using Load_t = __int128_t;

DEVICE_INLINE void cp_async128(Load_t *__restrict__ dst,
                               const Load_t *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s_dst),
               "l"(src));
}

DEVICE_INLINE void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

__device__ __forceinline__ uint32_t __ld_stream(const uint32_t *ptr) {
  uint32_t v;
  asm volatile("{\n"
               "   ld.global.ca.u32 %0, [%1];\n"
               "}\n"
               : "=r"(v)
               : "l"(ptr));
  return v;
}

template <class T, int sz> struct Vec {
  T d[sz];
};

union Features {
  uint32_t _;

  struct {
    uint32_t is_async : 1;
    uint32_t is_csc : 1;
    uint32_t is_split : 1;
    uint32_t is_naive : 1;
    uint32_t is_fp8 : 1;
    uint32_t is_torch_sparse : 1;
    uint32_t is_sputnik : 1;
    uint32_t rest : 25;
  } flags;
};

template <class T> DEVICE_INLINE T reduce_final(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <int PHASE, int WARP_COUNT>
__global__ void
doublesparse_naive(int m, int n, int k, const u32 *__restrict__ a_row_offsets,
                   const ColVal *__restrict__ a_col_vals,
                   const u32 *__restrict__ b_row_offsets,
                   const ColVal *__restrict__ b_col_vals, int non_zero_rows,
                   int batch_size, const half *__restrict__ _x, half *_y,
                   float *__restrict__ _workspace, u32 smem_size_fp32,
                   int global_row_offset = 0) {
  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[WARP_COUNT + 1];

  const half *x = _x + blockIdx.y * n;
  half *y = _y + blockIdx.y * m;

  float *workspace_out = _workspace + blockIdx.y * k;
  float *workspace_in = _workspace + blockIdx.y * k;

  const auto warp_id = threadIdx.x / WARP_SIZE;
  auto lane_id = threadIdx.x & 0x1f;

  auto row = blockIdx.x * WARP_COUNT + warp_id;

  if constexpr (!PHASE) {
    if (row >= non_zero_rows) {
      return;
    }
  } else {
    if (row >= m) {
      return;
    }
  }

  unsigned int row_to_load = row + threadIdx.x;

  if constexpr (!PHASE) {
    if (threadIdx.x <= WARP_COUNT && row_to_load <= non_zero_rows) {
      // Bug is here?
      s_row_offsets[threadIdx.x] = b_row_offsets[row_to_load];
    }
  } else {
    if (threadIdx.x <= WARP_COUNT && row_to_load <= m) {
      s_row_offsets[threadIdx.x] = a_row_offsets[row_to_load];
    }
  }

  __syncthreads();

  auto row_start = s_row_offsets[warp_id];
  auto row_end = s_row_offsets[warp_id + 1];
  auto row_ptr = row_start + lane_id;

  float acc{};

  const ColVal *col_vals;
  if constexpr (!PHASE) {
    col_vals = b_col_vals;
  } else {
    col_vals = a_col_vals;
  }

  {
    for (; row_ptr < row_end; row_ptr += WARP_SIZE) {
      // We ran out of x.
      ColVal col_val = col_vals[row_ptr];

      auto local_c = col_val.members.c;
      if constexpr (!PHASE) {
        acc += __half2float(x[local_c]) * __half2float(col_val.members.v);
      } else {
        if (col_val.members.c < non_zero_rows) {
          acc += workspace_in[local_c] * __half2float(col_val.members.v);
        } else {
          break;
        }
      }
    }
  }

  acc = reduce_final(acc);

  if (!lane_id) {
    if constexpr (!PHASE) {
      workspace_out[row] = acc;
    } else {
      y[row] = __float2half(acc);
    }
  }
}

DEVICE_INLINE __half_raw cvt_fp8_to_halfraw(const __nv_fp8_storage_t x) {
  __half_raw res;
  res.x = 0U;
  unsigned short int ur = x;
  ur = (unsigned short int)(ur << 8U);
  unsigned short int sign = ur & 0x8000U;
  unsigned short int exponent =
      (unsigned short int)(((ur & 0x7800U) >> 1U) + 0x2000U);
  unsigned short int mantissa = (ur & 0x0700U) >> 1U;
  unsigned char absx = 0x7FU & (unsigned char)x;
  if (absx == 0x7FU) // NaN
  {
    ur = 0x7FFFU; // fp16 canonical NaN, discard sign
  } else if (exponent == 0x2000U) {
    // zero or denormal
    if (mantissa != 0U) {
      // normalize
      mantissa = (unsigned short int)(mantissa << 1U);
      while ((mantissa & 0x0400U) == 0U) {
        mantissa = (unsigned short int)(mantissa << 1U);
        exponent = (unsigned short int)(exponent - 0x0400U);
      }
      // discard implicit leading bit
      mantissa &= 0x03FFU;
    } else { // Zero
      exponent = 0U;
    }

    ur = (sign | exponent) | mantissa;
  } else {
    ur = (sign | exponent) | mantissa;
  }
  res.x = ur;
  return res;
}

DEVICE_INLINE Vec<float, 4> _to_float_x4(uint32_t src_packed) {
  Vec<float, 4> res;
  float2 *f2 = reinterpret_cast<float2 *>(&res.d);
  half h0 = cvt_fp8_to_halfraw((__nv_fp8_storage_t)(src_packed & 0xFFu));
  src_packed >>= 8u;
  half h1 = cvt_fp8_to_halfraw((__nv_fp8_storage_t)(src_packed & 0xFFu));
  src_packed >>= 8u;
  half h2 = cvt_fp8_to_halfraw((__nv_fp8_storage_t)(src_packed & 0xFFu));
  src_packed >>= 8u;
  half h3 = cvt_fp8_to_halfraw((__nv_fp8_storage_t)(src_packed & 0xFFu));

  f2[0] = __half22float2(make_half2(h0, h1));
  f2[1] = __half22float2(make_half2(h2, h3));

  return res;
}

DEVICE_INLINE Vec<float, 4> to_float_x4(uint32_t src_packed, const float *lut) {
  Vec<float, 4> res;
  res.d[0] = lut[src_packed & 0xFFu];
  src_packed >>= 8u;
  res.d[1] = lut[src_packed & 0xFFu];
  src_packed >>= 8u;
  res.d[2] = lut[src_packed & 0xFFu];
  src_packed >>= 8u;
  res.d[3] = lut[src_packed & 0xFFu];
  return res;
}

template <int PHASE, int WARP_COUNT>
__global__ void doublesparse_fp8(
    int m, int n, int k,

    const u32 *__restrict__ a_row_offsets, const u64 *__restrict__ a_columns,
    const u32 *__restrict__ a_values,

    const u32 *__restrict__ b_row_offsets, const u64 *__restrict__ b_columns,
    const u32 *__restrict__ b_values,

    int non_zero_rows, int batch_size, const half *__restrict__ x, half *_y,
    float *__restrict__ _workspace, u32 smem_size_fp32) {
  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[WARP_COUNT + 1];
  __shared__ float s_lut_fp8[1 << 8];
  for (int i = threadIdx.x; i < 256; i += WARP_COUNT * WARP_SIZE) {
    s_lut_fp8[i] = __half2float(cvt_fp8_to_halfraw(i));
  }

  half *y = _y + blockIdx.y * m;
  u32 smem_size_fp16 = smem_size_fp32 * 2;
  auto s_x = reinterpret_cast<half *>(s_x2);

  float *workspace = _workspace + blockIdx.y * k;
  const half2 *x2 = reinterpret_cast<const half2 *>(x + blockIdx.y * n);

  const auto warp_id = threadIdx.x / WARP_SIZE;
  auto lane_id = threadIdx.x & 0x1f;

  auto row = blockIdx.x * WARP_COUNT + warp_id;

  u32 available_warps{};

  if constexpr (!PHASE) {
    available_warps = min(WARP_COUNT, non_zero_rows - blockIdx.x * WARP_COUNT);
    if (row >= non_zero_rows) {
      return;
    }
  } else {
    available_warps = min(WARP_COUNT, m - blockIdx.x * WARP_COUNT);
    if (row >= m) {
      return;
    }
  }

  int row_to_load = (blockIdx.x * WARP_COUNT) + threadIdx.x;
  int rows_to_load =
      min(WARP_COUNT, ((!PHASE ? non_zero_rows : m) - blockIdx.x * WARP_COUNT));

  const u32 *row_offsets = !PHASE ? b_row_offsets : a_row_offsets;

  if (threadIdx.x < WARP_COUNT &&
      row_to_load < (blockIdx.x * WARP_COUNT) + rows_to_load) {
    s_row_offsets[threadIdx.x] = row_offsets[row_to_load];
  }

  if (!threadIdx.x) {
    s_row_offsets[rows_to_load] =
        row_offsets[blockIdx.x * WARP_COUNT + rows_to_load];
  }

  if (warp_id >= rows_to_load) {
    return;
  }

  float acc{};
  // Pages of x strips.
  auto pages = !PHASE ? updiv(n / 2, smem_size_fp32)
                      : updiv(non_zero_rows, smem_size_fp32);

  const u64 *columns;
  const u32 *values;
  if (!PHASE) {
    columns = b_columns;
    values = b_values;
  } else {
    columns = a_columns;
    values = a_values;
  }

  u32 avilable_threads = available_warps * WARP_SIZE;

  if (pages == 1) {
    auto x2_to_load = !PHASE ? (n / 2) : non_zero_rows;
    for (int i = threadIdx.x; i < x2_to_load; i += avilable_threads) {
      if constexpr (!PHASE) {
        s_x2[i] = x2[i];
      } else {
        reinterpret_cast<float *>(s_x2)[i] = workspace[i];
      }
    }

    __syncthreads();

    int row_start = s_row_offsets[warp_id];
    int row_end = s_row_offsets[warp_id + 1];
    auto row_ptr = row_start + lane_id;

    for (; row_ptr < row_end; row_ptr += WARP_SIZE) {
      auto val4 = to_float_x4(values[row_ptr], s_lut_fp8);
      auto col4 =
          *reinterpret_cast<const Vec<unsigned short, 4> *>(columns + row_ptr);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        auto v = val4.d[i];
        if constexpr (!PHASE) {
          acc += __half2float(s_x[col4.d[i]]) * v;
        } else {
          acc += reinterpret_cast<float *>(s_x2)[col4.d[i]] * v;
        }
      }
    }
  } else {
    __syncthreads();

    auto row_start = s_row_offsets[warp_id];
    auto row_end = s_row_offsets[warp_id + 1];
    auto row_ptr = row_start + lane_id;

    for (int page = 0; page < pages; page++) {
      const u32 page_offset = page * smem_size_fp32;

      __syncthreads();

      int x_limit{};
      if constexpr (!PHASE) {
        auto x2_to_load = n / 2 - page_offset;
        x_limit = min(smem_size_fp32, x2_to_load);
        for (int i = threadIdx.x; i < x_limit; i += avilable_threads) {
          s_x2[i] = x2[i];
        }
      } else {
        auto x2_to_load = non_zero_rows - page_offset;
        x_limit = min(smem_size_fp32, x2_to_load);
        for (int i = threadIdx.x; i < x_limit; i += avilable_threads) {
          reinterpret_cast<float *>(s_x2)[i] = workspace[page_offset + i];
        }
      }

      __syncthreads();

      u32 upper_col_limit{}, lower_col_limit{};

      if constexpr (!PHASE) {
        lower_col_limit = smem_size_fp16 * page;
        upper_col_limit = smem_size_fp16 * (page + 1);
      } else {
        lower_col_limit = smem_size_fp32 * page;
        upper_col_limit = smem_size_fp32 * (page + 1);
      }

      for (; row_ptr < row_end; row_ptr += WARP_SIZE) {
        auto val4 = to_float_x4(values[row_ptr], s_lut_fp8);
        auto col4 = *reinterpret_cast<const Vec<unsigned short, 4> *>(columns +
                                                                      row_ptr);
        bool done = false;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          if (col4.d[i] < lower_col_limit || (col4.d[i] >= upper_col_limit)) {
            done = true;
          } else {
            done = false;
            auto v = val4.d[i];
            if constexpr (!PHASE) {
              auto local_c = col4.d[i] - page * smem_size_fp16;
              acc += __half2float(s_x[local_c]) * v;
            } else {
              auto local_c = col4.d[i] - page * smem_size_fp32;
              acc += reinterpret_cast<float *>(s_x2)[local_c] * v;
            }
          }
        }
        if (done) {
          break;
        }
      }

      x2 += smem_size_fp32;
    }
  }

  acc = reduce_final(acc);

  if (!lane_id) {
    if constexpr (!PHASE) {
      workspace[row] = acc;
    } else {
      y[row] = __float2half(acc);
    }
  }
}

namespace sputnik {
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
                     const int *__restrict__ row_indices,
                     const float *__restrict__ values,
                     const int *__restrict__ row_offsets,
                     const int *__restrict__ column_indices,
                     const float *__restrict__ dense_matrix,
                     float *__restrict__ output_matrix, cudaStream_t stream);
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
                     const int *__restrict__ row_indices,
                     const half2 *__restrict__ values,
                     const int *__restrict__ row_offsets,
                     const short2 *__restrict__ column_indices,
                     const half2 *__restrict__ dense_matrix,
                     half2 *__restrict__ output_matrix, cudaStream_t stream);
} // namespace sputnik

template <int WARP_COUNT>
__global__ void
doublesparse_csc(int m, int n, int k, const u32 *__restrict__ a_row_offsets,
                 const ColVal *__restrict__ a_col_vals,
                 const u32 *__restrict__ b_row_offsets,
                 const ColVal *__restrict__ b_col_vals, int non_zero_rows,
                 int batch_size, const half *__restrict__ x, half *_y,
                 float *__restrict__ _workspace, u32 smem_size_fp32) {
  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[WARP_COUNT + 1];

  half *y = _y + blockIdx.y * m;
  u32 smem_size_fp16 = smem_size_fp32 * 2;
  auto s_x = reinterpret_cast<half *>(s_x2);

  float *workspace = _workspace + blockIdx.y * k;
  const half2 *x2 = reinterpret_cast<const half2 *>(x + blockIdx.y * n / 2);

  const auto warp_id = threadIdx.x / WARP_SIZE;
  auto lane_id = threadIdx.x & 0x1f;

  auto row = blockIdx.x * WARP_COUNT + warp_id;

  if (row >= non_zero_rows) {
    return;
  }

  if (threadIdx.x < WARP_COUNT) {
    s_row_offsets[threadIdx.x] = b_row_offsets[row + threadIdx.x];
  }
  if (threadIdx.x == WARP_COUNT) {
    s_row_offsets[threadIdx.x] = b_row_offsets[row + threadIdx.x];
  }

  __syncthreads();
  auto b_row_start = s_row_offsets[warp_id];
  auto b_row_end = s_row_offsets[warp_id + 1];
  auto b_row_ptr = b_row_start + lane_id;

  float acc{};
  // Pages of x strips.
  auto pages = updiv(n / 2, smem_size_fp32);

  const ColVal *col_vals = b_col_vals;

  if (pages == 1) {
    auto x2_to_load = n / 2;
    for (int i = threadIdx.x; i < x2_to_load; i += blockDim.x) {
      s_x2[i] = x2[i];
    }

    __syncthreads();

    for (; b_row_ptr < b_row_end; b_row_ptr += WARP_SIZE) {
      // We ran out of x.
      ColVal col_val = col_vals[b_row_ptr];

      auto local_c = col_val.members.c;
      acc += __half2float(s_x[local_c]) * __half2float(col_val.members.v);
    }
  } else {
    for (int page = 0; page < pages; page++) {
      const u32 page_offset = page * smem_size_fp32;
#ifndef SKIP_XLOAD
      auto x2_to_load = n / 2 - page_offset;
      const u32 x_limit = min(smem_size_fp32, x2_to_load);
      for (int i = threadIdx.x; i < x_limit; i += blockDim.x) {
        s_x2[i] = x2[i];
      }
#endif

      __syncthreads();

      u32 column_limit{};

      column_limit = smem_size_fp16 * (page + 1);

      for (; b_row_ptr < b_row_end; b_row_ptr += WARP_SIZE) {
        // We ran out of x.
        ColVal col_val = col_vals[b_row_ptr];

        if (col_val.members.c >= column_limit) {
          break;
        }

        auto local_c = col_val.members.c - page * smem_size_fp16;
        acc += __half2float(s_x[local_c]) * __half2float(col_val.members.v);
      }

      __syncthreads();

      x2 += smem_size_fp32;
    }
  }

  acc = reduce_final(acc);
}

#define CALL_DOUBLE_MATMUL_FP8(P, SMEM_SIZE_FP32, K, given_stream)             \
  doublesparse_fp8<P, WARP_COUNT>                                              \
      <<<dim3(updiv((!P ? non_zero_rows : m), WARP_COUNT), batch_size, 1),     \
         dim3(THREAD_COUNT), sizeof(int) * (SMEM_SIZE_FP32), given_stream>>>(  \
          m, n, k, mat8[0].d_values_row_offsets, mat8[0].d_columns,            \
          mat8[0].d_values, mat8[1].d_values_row_offsets, mat8[1].d_columns,   \
          mat8[1].d_values, non_zero_rows, batch_size, (half *)X, (half *)y,   \
          d_workspace, SMEM_SIZE_FP32)

#define KERNEL_CALL_FP8                                                        \
  CALL_DOUBLE_MATMUL_FP8(0, std::min((1 << 13), updiv(n, 2)), stream, 0);      \
  CALL_DOUBLE_MATMUL_FP8(1, std::min((1 << 13), k), stream, 0);

static constexpr int FP8_BITS = 8;
static constexpr int FP16_BITS = 16;
static constexpr int FP8_VALUES_CHUNK_SIZE = 4;
static constexpr int FP8_COLUMNS_CHUNK_SIZE = FP8_VALUES_CHUNK_SIZE / 2;

__global__ void fill_offsets_fp8(int non_zero_row_count, const int *row_offsets,
                                 int *row_offsets_value_aligned_fp8) {
  int i{};
  row_offsets_value_aligned_fp8[0] = 0;
  for (; i < non_zero_row_count; i++) {
    int nnz = row_offsets[i + 1] - row_offsets[i];
    int nnz_vals = UPDIV(nnz, FP8_VALUES_CHUNK_SIZE);
    row_offsets_value_aligned_fp8[i + 1] =
        row_offsets_value_aligned_fp8[i] + nnz_vals;
  }
}

__global__ void convert_to_fp8(
    // Inputs
    // FP16
    u32 rows, const int *row_offsets, const int *col_vals,
    // FP8
    const int *row_offsets_value_aligned_fp8,
    // Outputs
    u64 *columns_fp8, u32 *values_fp8) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= rows) {
    return;
  }
  int colval_fp8_id{};
  for (int j = row_offsets[i]; j < row_offsets[i + 1];
       j += FP8_VALUES_CHUNK_SIZE) {
    u32 vals_x4{};
    u64 cols_x4{};
    u32 colval_id{};
    ColVal col_val;
    for (int k = 0; k < FP8_VALUES_CHUNK_SIZE; k++) {
      if (j + k < row_offsets[i + 1]) {
        col_val = ColVal{._ = (u32)col_vals[j + k]};
      } else {
        col_val.members.v = __int2half_rd(0);
      }
      auto value_fp8 = __nv_fp8_e4m3(col_val.members.v);
      vals_x4 |= uint32_t(*reinterpret_cast<unsigned char *>(&(value_fp8)))
                 << (colval_id * FP8_BITS);
      cols_x4 |= static_cast<uint64_t>(col_val.members.c)
                 << (static_cast<uint64_t>(colval_id) *
                     static_cast<uint64_t>(FP16_BITS));
      colval_id++;
    }
    values_fp8[row_offsets_value_aligned_fp8[i] + colval_fp8_id] = vals_x4;
    columns_fp8[row_offsets_value_aligned_fp8[i] + colval_fp8_id] = cols_x4;
    colval_fp8_id++;
  }
}

struct FP16MatrixCSR {
  int rows;
  int *d_row_offsets;
  int *d_col_vals;
};

struct FP8MatrixCSR {
  int rows;
  int *d_values_row_offsets;
  u32 *d_values;
  u64 *d_columns;

  void allocate() {
    cudaMalloc(reinterpret_cast<void **>(&d_values_row_offsets),
               sizeof(u32) * (rows + 1));
  }

  void free() {
    cudaFree(d_values_row_offsets);
    cudaFree(d_values);
    cudaFree(d_columns);
  }
};

struct SputnikCSR {
  int rows;
  int cols;
  int *d_row_offsets = nullptr;
  short *d_columns = nullptr;
  half *d_values = nullptr;
  int nnz;
  int *d_row_indices;

  void allocate_offsets() {
    cudaMalloc(reinterpret_cast<void **>(&d_row_offsets),
               (rows + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_row_indices), rows * sizeof(int));
  }

  void allocate_nnz(int nnz) {
    this->nnz = nnz;
    cudaMalloc(reinterpret_cast<void **>(&d_columns), nnz * sizeof(short));
    cudaMalloc(reinterpret_cast<void **>(&d_values), nnz * sizeof(half));
  }

  void free() {
    cudaFree(d_row_offsets);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_row_indices);
  }
};

struct TorchCSR {
  int rows;
  int *d_row_offsets;
  int *d_columns;
  float *d_values;
  int nnz;

  void allocate_offsets() {
    cudaMalloc(reinterpret_cast<void **>(&d_row_offsets),
               (rows + 1) * sizeof(int));
  }

  void allocate_nnz(int nnz) {
    this->nnz = nnz;
    cudaMalloc(reinterpret_cast<void **>(&d_columns), nnz * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_values), nnz * sizeof(int));
  }

  void free() {
    cudaFree(d_row_offsets);
    cudaFree(d_columns);
    cudaFree(d_values);
  }
};

__global__ void convert_to_torch( // Inputs
                                  // FP16
    int fp16_rows, int *d_fp16_row_offsets, int *d_fp16_col_vals,
    int torch_csr_rows, int *d_torch_csr_row_offsets, int *d_torch_csr_columns,
    float *d_torch_csr_values, int torch_csr_nnz) {
  if (!threadIdx.x) {
    int i;
    for (i = 0; i <= fp16_rows; i++) {
      d_torch_csr_row_offsets[i] = d_fp16_row_offsets[i];
    }
    for (; i <= torch_csr_rows; i++) {
      d_torch_csr_row_offsets[i] = d_fp16_row_offsets[fp16_rows];
    }
  }

  int nnz = d_fp16_row_offsets[fp16_rows];
  for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
    ColVal col_val = *reinterpret_cast<const ColVal *>(d_fp16_col_vals);
    d_torch_csr_columns[i] = col_val.members.c;
    d_torch_csr_values[i] = __half2float(col_val.members.v);
  }
}

__global__ void convert_to_sputnik( // Inputs
                                    // FP16
    int nonzero_row_count, int *d_fp16_row_offsets, int *d_fp16_col_vals,
    int *d_torch_csr_row_offsets, short *d_torch_csr_columns,
    half *d_torch_csr_values, int full_row_count, int *row_indices) {
  if (!threadIdx.x) {
    int i;
    for (i = 0; i <= nonzero_row_count; i++) {
      d_torch_csr_row_offsets[i] = d_fp16_row_offsets[i];
    }
    for (; i <= full_row_count; i++) {
      d_torch_csr_row_offsets[i] = d_fp16_row_offsets[nonzero_row_count];
    }
  }

  int nnz = d_fp16_row_offsets[nonzero_row_count];
  for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
    ColVal col_val = *reinterpret_cast<const ColVal *>(d_fp16_col_vals);
    d_torch_csr_columns[i] = col_val.members.c;
    d_torch_csr_values[i] = col_val.members.v;
  }

  for (int i = threadIdx.x; i < full_row_count; i += blockDim.x) {
    row_indices[i] = i;
  }
}

void convert_sync(FP16MatrixCSR mat_fp16[2], SputnikCSR mat_sputnik_csr[2]) {
  int nnzs[2]{0, 0};
  for (int i = 0; i < 2; i++) {
    mat_sputnik_csr[i].allocate_offsets();
    cudaMemcpy(nnzs + i, mat_fp16[i].d_row_offsets + mat_fp16[i].rows,
               sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < 2; i++) {
    mat_sputnik_csr[i].allocate_nnz(nnzs[i]);
    convert_to_sputnik<<<1, 256>>>(
        mat_fp16[i].rows, mat_fp16[i].d_row_offsets, mat_fp16[i].d_col_vals,
        mat_sputnik_csr[i].d_row_offsets, mat_sputnik_csr[i].d_columns,
        mat_sputnik_csr[i].d_values, mat_sputnik_csr[i].rows,
        mat_sputnik_csr[i].d_row_indices);
  }
}

template <class T>
torch::Tensor tensor_from_cuda_ptr(T *d_ptr, int64_t size,
                                   torch::ScalarType dtype) {
  // Create a tensor from the raw CUDA pointer.
  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
  return torch::from_blob(d_ptr, {size}, options);
}

void convert_sync(FP16MatrixCSR mat_fp16[2], TorchCSR mat_torch_csr[2]) {
  u32 nnzs[2];
  for (int i = 0; i < 2; i++) {
    mat_torch_csr[i].allocate_offsets();
    cudaMemcpy(nnzs + i, mat_fp16[i].d_row_offsets + mat_fp16[i].rows,
               sizeof(u32), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < 2; i++) {
    mat_torch_csr[i].allocate_nnz(nnzs[i]);
    convert_to_torch<<<1, 256>>>(
        mat_fp16[i].rows, mat_fp16[i].d_row_offsets, mat_fp16[i].d_col_vals,
        mat_torch_csr[i].rows, mat_torch_csr[i].d_row_offsets,
        mat_torch_csr[i].d_columns, mat_torch_csr[i].d_values,
        mat_torch_csr[i].nnz);
  }
}

void convert_sync(FP16MatrixCSR mat_fp16[2], FP8MatrixCSR mat_fp8[2]) {
  for (int i = 0; i < 2; i++) {
    fill_offsets_fp8<<<1, 1>>>(mat_fp16[i].rows, mat_fp16[i].d_row_offsets,
                               mat_fp8[i].d_values_row_offsets);
  }
  cudaDeviceSynchronize();
  int nnzs[2];
  for (int i = 0; i < 2; i++) {
    cudaMemcpy(nnzs + i, mat_fp8[i].d_values_row_offsets + mat_fp8[i].rows,
               sizeof(u32), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < 2; i++) {
    cudaMalloc(reinterpret_cast<void **>(&mat_fp8[i].d_values),
               sizeof(u32) * nnzs[i]);
    cudaMalloc(reinterpret_cast<void **>(&mat_fp8[i].d_columns),
               sizeof(u64) * nnzs[i]);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < 2; i++) {
    convert_to_fp8<<<UPDIV(mat_fp8[i].rows, 256), 256>>>(
        mat_fp8[i].rows, mat_fp16[i].d_row_offsets, mat_fp16[i].d_col_vals,
        mat_fp8[i].d_values_row_offsets, mat_fp8[i].d_columns,
        mat_fp8[i].d_values);
  }
  cudaDeviceSynchronize();
}

#if ENABLE_TORCH
template <class T>
torch::Tensor tensor_from_cuda_ptr(T *d_ptr, int64_t size,
                                   torch::ScalarType dtype) {
  // Create a tensor from the raw CUDA pointer.
  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
  return torch::from_blob(d_ptr, {size}, options);
}
#endif


namespace doublesparse_external {
int doublesparse_matmul_external_timer(int m, int n, int k, int *a_row_offsets,
                                       int *a_col_vals, int *b_row_offsets,
                                       int *b_col_vals, int non_zero_rows,
                                       int batch_size,
                                       // 16-bit
                                       // Input
                                       void *X, void *y, void *d_workspace_ptr,
                                       // Output
                                       cudaStream_t stream, void *measurements,
                                       uint32_t feature_flag) {
  Features features{._ = feature_flag};

  constexpr u32 WARP_COUNT = 16;
  constexpr u32 THREAD_COUNT = WARP_SIZE * WARP_COUNT;

  float *d_workspace = static_cast<float *>(d_workspace_ptr);

  FP16MatrixCSR mat16[2] = {FP16MatrixCSR{.rows = m,
                                          .d_row_offsets = a_row_offsets,
                                          .d_col_vals = a_col_vals},
                            FP16MatrixCSR{.rows = non_zero_rows,
                                          .d_row_offsets = b_row_offsets,
                                          .d_col_vals = b_col_vals}};

  float measurement = 0;

  if (features.flags.is_torch_sparse) {
    TorchCSR mat_torch[2] = {TorchCSR{.rows = m}, TorchCSR{.rows = k}};
    std::vector<torch::Tensor> torch_csr_tensors(2);
    torch::Tensor dense_inputs[2];
    torch::Tensor dense_outputs[2];

    convert_sync(mat16, mat_torch);
    for (int i = 0; i < 2; i++) {
      torch::Tensor crow_indices = tensor_from_cuda_ptr(
          mat_torch[i].d_row_offsets, mat_torch[i].rows + 1, torch::kInt32);
      torch::Tensor col_indices = tensor_from_cuda_ptr(
          mat_torch[i].d_columns, mat_torch[i].nnz, torch::kInt32);
      torch::Tensor values = tensor_from_cuda_ptr(
          mat_torch[i].d_values, mat_torch[i].nnz, torch::kFloat);
      torch_csr_tensors[i] = torch::sparse_csr_tensor(
          crow_indices, col_indices, values,
          ((i == 0) ? torch::IntArrayRef{m, k} : torch::IntArrayRef{k, n}),
          torch::TensorOptions().device(torch::kCUDA));
    }
    dense_inputs[0] =
        torch::rand(
            {k * batch_size, 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
            .contiguous();
    dense_inputs[1] =
        torch::rand(
            {n * batch_size, 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
            .contiguous();

    dense_outputs[0] =
        torch::rand(
            {m * batch_size, 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
            .contiguous();
    dense_outputs[1] =
        torch::rand(
            {k * batch_size, 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
            .contiguous();

    for (int i = 0; i < WARMUPS; i++) {
      matmul_out(dense_outputs[1], torch_csr_tensors[1], dense_inputs[1]);
      cudaDeviceSynchronize();
      matmul_out(dense_outputs[0], torch_csr_tensors[0], dense_inputs[0]);
    }

    MyTimer timer(stream);
    timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
      matmul_out(dense_outputs[1], torch_csr_tensors[1], dense_inputs[1]);
      cudaDeviceSynchronize();
      matmul_out(dense_outputs[0], torch_csr_tensors[0], dense_inputs[0]);
    }

    measurement = timer.end_and_measure() / NUM_RUNS;

    for (int i = 0; i < 2; i++) {
      mat_torch[i].free();
    }
  } else if (features.flags.is_fp8) {
    FP8MatrixCSR mat8[2] = {FP8MatrixCSR{.rows = m},
                            FP8MatrixCSR{.rows = non_zero_rows}};

    for (int i = 0; i < 2; i++) {
      mat8[i].allocate();
    }

    cudaDeviceSynchronize();

    convert_sync(mat16, mat8);

    for (int j = 0; j < WARMUPS; j++) {
      //      KERNEL_CALL_FP8
    }

    cudaDeviceSynchronize();

    MyTimer timer(stream);

    timer.start();
    for (int j = 0; j < NUM_RUNS; j++) {
      //      KERNEL_CALL_FP8
    }

    measurement = timer.end_and_measure() / NUM_RUNS;

    for (int i = 0; i < 2; i++) {
      mat8[i].free();
    }
  } else if (features.flags.is_sputnik) {
    SputnikCSR mat_sputnik[2] = {SputnikCSR{.rows = m, .cols = k},
                                 SputnikCSR{.rows = k, .cols = n}};

    half *dense_sputnik_inputs[2];
    half *dense_sputnik_outputs[2];
    convert_sync(mat16, mat_sputnik);
    cudaMalloc(reinterpret_cast<void **>(&dense_sputnik_inputs[1]),
               n * batch_size * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&dense_sputnik_inputs[0]),
               k * batch_size * sizeof(half));

    cudaMalloc(reinterpret_cast<void **>(&dense_sputnik_outputs[1]),
               k * batch_size * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&dense_sputnik_outputs[0]),
               m * batch_size * sizeof(half));

    cudaDeviceSynchronize();
    for (int j = 0; j < WARMUPS; j++) {
      for (int i = 1; i >= 0; i--) {
        sputnik::CudaSpmm(
            (i == 1) ? k : m, (i == 1 ? n : k), std::max(batch_size, 2),
            mat_sputnik[i].nnz, mat_sputnik[i].d_row_indices,
            reinterpret_cast<half2 *>(mat_sputnik[i].d_values),
            mat_sputnik[i].d_row_offsets,
            reinterpret_cast<short2 *>(mat_sputnik[i].d_columns),
            reinterpret_cast<half2 *>(dense_sputnik_inputs[i]),
            reinterpret_cast<half2 *>(dense_sputnik_outputs[i]), stream);
        if (i) {
          cudaDeviceSynchronize();
        }
      }
    }

    cudaDeviceSynchronize();

    MyTimer timer(stream);
    timer.start();
    for (int j = 0; j < NUM_RUNS; j++) {
      for (int i = 1; i >= 0; i--) {
        sputnik::CudaSpmm(
            (i == 1) ? k : m, (i == 1 ? n : k), std::max(batch_size, 2),
            mat_sputnik[i].nnz, mat_sputnik[i].d_row_indices,
            reinterpret_cast<half2 *>(mat_sputnik[i].d_values),
            mat_sputnik[i].d_row_offsets,
            reinterpret_cast<short2 *>(mat_sputnik[i].d_columns),
            reinterpret_cast<half2 *>(dense_sputnik_inputs[i]),
            reinterpret_cast<half2 *>(dense_sputnik_outputs[i]), stream);
        if (i) {
          cudaDeviceSynchronize();
        }
      }
    }

    measurement = timer.end_and_measure() / NUM_RUNS;

    for (int i = 0; i < 2; i++) {
      mat_sputnik[i].free();
    }
  }

  static_cast<float *>(measurements)[0] = measurement;

  return 0;
}

} // namespace doublesparse_external
