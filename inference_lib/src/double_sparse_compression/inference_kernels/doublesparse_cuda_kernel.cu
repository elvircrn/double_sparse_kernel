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

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

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

half2 DEVICE_INLINE dequantize2(const half2 &q, const half2 &s,
                                const half2 &z) {
  const half2 &res = __hmul2(s, __hsub2(q, z));
  return res;
}

template <class Bit_t, class Scalar_t>
DEVICE_INLINE Scalar_t dequantize(Bit_t q, Scalar_t s, Scalar_t z) {
  if constexpr (std::is_same<Bit_t, half>::value) {
    return __hmul(s, __hsub(q, z));
  } else {
    return __hmul(s, __hsub(__uint2half_rd(q, z)));
  }
}

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

[[nodiscard]] __device__ __host__ CUINLINE int updiv(int x, int y) {
  return (x + y - 1) / y;
}

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

DEVICE_INLINE float add_and_accum(float a, float b) { return a + b; }

DEVICE_INLINE half add_and_accum(const half2 &a, const half2 &b) {
  half2 r = __hadd2(a, b);
  return __hadd(r.x, r.y);
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

union Features {
  uint32_t _;

  struct {
    uint32_t is_async : 1;
    uint32_t is_csc : 1;
    uint32_t is_split : 1;
    uint32_t is_naive : 1;
    uint32_t rest : 28;
  } flags;
};

DEVICE_INLINE float reduce_final(float val) {
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

template <int PHASE, int WARP_COUNT>
__global__ void
doublesparse(int m, int n, int k, const u32 *__restrict__ a_row_offsets,
             const ColVal *__restrict__ a_col_vals,
             const u32 *__restrict__ b_row_offsets,
             const ColVal *__restrict__ b_col_vals, int non_zero_rows,
             int batch_size, const half *__restrict__ x, half *_y,
             float *__restrict__ _workspace, u32 smem_size_fp32,
             int global_row_offset = 0) {
  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[WARP_COUNT + 1];

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

  int num_rows = !PHASE ? non_zero_rows : m;

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

  const ColVal *col_vals;
  if (!PHASE) {
    col_vals = b_col_vals;
  } else {
    col_vals = a_col_vals;
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
      // We ran out of x.
      ColVal col_val = col_vals[row_ptr];
      auto v = __half2float(col_val.members.v);
      if constexpr (!PHASE) {
        acc += __half2float(s_x[col_val.members.c]) * v;
      } else {
        acc += reinterpret_cast<float *>(s_x2)[col_val.members.c] * v;
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

      u32 column_limit{};

      if constexpr (!PHASE) {
        column_limit = smem_size_fp16 * (page + 1);
      } else {
        column_limit = smem_size_fp32 * (page + 1);
      }

      for (; row_ptr < row_end; row_ptr += WARP_SIZE) {
        // We ran out of x.
        ColVal col_val = col_vals[row_ptr];

        if (col_val.members.c >= column_limit) {
          break;
        }

        if constexpr (!PHASE) {
          auto local_c = col_val.members.c - page * smem_size_fp16;
          acc += __half2float(s_x[local_c]) * __half2float(col_val.members.v);
        } else {
          int local_c = (int)col_val.members.c - (int)page_offset;
          acc += reinterpret_cast<float *>(s_x2)[local_c] *
                 __half2float(col_val.members.v);
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

#define CALL_DOUBLE_MATMUL(P, SMEM_SIZE_FP32, K, given_stream, global_offset)  \
  K<P, WARP_COUNT>                                                             \
      <<<dim3(updiv((!P ? non_zero_rows : m), WARP_COUNT), batch_size, 1),     \
         dim3(THREAD_COUNT), sizeof(int) * (SMEM_SIZE_FP32), given_stream>>>(  \
          m, n, k, (u32 *)a_row_offsets, (ColVal *)a_col_vals,                 \
          (u32 *)b_row_offsets, (ColVal *)b_col_vals, non_zero_rows,           \
          batch_size, (half *)X, (half *)y, d_workspace, SMEM_SIZE_FP32,       \
          global_offset)

#define CALL_DOUBLE_MATMUL_CSC(P, SMEM_SIZE_FP32)                              \
  doublesparse_csc<WARP_COUNT>                                                 \
      <<<dim3(updiv(!P ? non_zero_rows : m, WARP_COUNT), batch_size, 1),       \
         dim3(THREAD_COUNT), sizeof(int) * (SMEM_SIZE_FP32), stream>>>(        \
          m, n, k, (u32 *)a_row_offsets, (ColVal *)a_col_vals,                 \
          (u32 *)b_row_offsets, (ColVal *)b_col_vals, non_zero_rows,           \
          batch_size, (half *)X, (half *)y, d_workspace, SMEM_SIZE_FP32)

#define KERNEL_CALL                                                            \
  if (!features.flags.is_csc && !features.flags.is_naive) {                    \
    CALL_DOUBLE_MATMUL(0, std::min((1 << 13), updiv(n, 2)), doublesparse,      \
                       stream, 0);                                             \
    CALL_DOUBLE_MATMUL(1, std::min((1 << 13), k), doublesparse, stream, 0);    \
  } else if (features.flags.is_naive) {                                        \
    CALL_DOUBLE_MATMUL(0, std::min((1 << 13), updiv(n, 2)),                    \
                       doublesparse_naive, stream, 0);                         \
    CALL_DOUBLE_MATMUL(1, std::min((1 << 13), k), doublesparse_naive, stream,  \
                       0);                                                     \
  } else {                                                                     \
    CALL_DOUBLE_MATMUL_CSC(0, std::min((1 << 13), updiv(n, 2)));               \
    CALL_DOUBLE_MATMUL_CSC(1, std::min((1 << 13), k));                         \
  }

int doublesparse_matmul(int m, int n, int k, void *a_row_offsets,
                        void *a_col_vals, void *b_row_offsets, void *b_col_vals,
                        int non_zero_rows, int batch_size,
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

  Timer *timer{};
  if (measurements) {
    constexpr u32 NUM_RUNS = 500;
    constexpr u32 WARMUPS = 100;
    for (int i = 0; i < WARMUPS; i++) {
      KERNEL_CALL;
    }
    timer = new Timer(stream);
    timer->start();

    for (int i = 0; i < NUM_RUNS; i++) {
      KERNEL_CALL;
    }

    static_cast<float *>(measurements)[0] = timer->end_and_measure() / NUM_RUNS;
    delete timer;
  } else {
    KERNEL_CALL;
    if (!features.flags.is_async) {
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  }

  return 0;
}
