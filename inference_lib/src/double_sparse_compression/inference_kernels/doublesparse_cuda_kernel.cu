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

#include <ATen/core/interned_strings.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

static __device__ __host__ __forceinline__ int updiv(int x, int y) {
  return (x + y - 1) / y;
}

namespace doublesparse {
static constexpr u32 SHARED_OFFSET = 32;
static constexpr u32 WARP_SIZE = 32;
static constexpr u32 FULL_MASK = 0xFFFFFFFFu;

#define DEVICE_INLINE __forceinline__ __device__

#define INT2_TO_HALF2(v)                                                       \
  make_half2(__int2half_rd((v) & 0b111), __int2half_rd(((v) >> 3) & 0b111))

template <class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

#define CUINLINE __forceinline__

#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))

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
doublesparse(int m, int n, int k, const u32 *__restrict__ a_row_offsets,
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
  const half2 *x2 = reinterpret_cast<const half2 *>(x + blockIdx.y * n);

  const auto warp_id = threadIdx.x / WARP_SIZE;
  auto lane_id = threadIdx.x & 0x1f;

  auto row = blockIdx.x * WARP_COUNT + warp_id;
  int num_rows = !PHASE ? non_zero_rows : m;

  int available_warps = min(WARP_COUNT, num_rows - blockIdx.x * WARP_COUNT);
  if (row >= num_rows) {
    return;
  }
  const u32 *row_offsets = !PHASE ? b_row_offsets : a_row_offsets;

  int row_to_load = (blockIdx.x * WARP_COUNT) + threadIdx.x;
  int num_rows_to_load =
      min(WARP_COUNT, ((!PHASE ? non_zero_rows : m) - blockIdx.x * WARP_COUNT));


  if (threadIdx.x <= available_warps) {
    s_row_offsets[threadIdx.x] = row_offsets[row_to_load];
  }

  if (warp_id >= num_rows_to_load) {
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

  auto x2_to_load = !PHASE ? (n / 2) : non_zero_rows;
  auto fp32_to_load = !PHASE ? (n / 2) : non_zero_rows;

  __syncthreads();

  auto row_start = s_row_offsets[warp_id];
  auto row_end = s_row_offsets[warp_id + 1];
  auto row_ptr = row_start + lane_id;

  for (int page = 0; page < pages; page++) {
    const u32 page_offset = page * smem_size_fp32;

    __syncthreads();

    auto x2_to_load = fp32_to_load - page_offset;
    int x_limit = min(smem_size_fp32, x2_to_load);
    for (int i = threadIdx.x; i < x_limit; i += avilable_threads) {
      if constexpr (!PHASE) {
        s_x2[i] = x2[i];
      } else {
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

  acc = reduce_final(acc);

  if (!lane_id) {
    if constexpr (!PHASE) {
      workspace[row] = acc;
    } else {
      y[row] = __float2half(acc);
    }
  }
}

#define CALL_DOUBLE_MATMUL(P, SMEM_SIZE_FP32)                                  \
  doublesparse<P, WARP_COUNT>                                                  \
      <<<dim3(updiv((!P ? non_zero_rows : m), WARP_COUNT), batch_size, 1),     \
         dim3(THREAD_COUNT), sizeof(int) * (SMEM_SIZE_FP32), stream>>>(        \
          m, n, k, (u32 *)a_row_offsets, (ColVal *)a_col_vals,                 \
          (u32 *)b_row_offsets, (ColVal *)b_col_vals, non_zero_rows,           \
          batch_size, (half *)X, (half *)y, d_workspace, SMEM_SIZE_FP32)

#define KERNEL_CALL_FP16                                                       \
  CALL_DOUBLE_MATMUL(0, std::min((1 << 12), updiv(n, 2)));                     \
  CALL_DOUBLE_MATMUL(1, std::min((1 << 12), k));

static constexpr int FP16_BITS = 16;

struct FP16MatrixCSR {
  int rows;
  u32 *d_row_offsets;
  u32 *d_col_vals;
};

int doublesparse_matmul(int m, int n, int k, u32 *a_row_offsets,
                        u32 *a_col_vals, u32 *b_row_offsets, u32 *b_col_vals,
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

  if (measurements) {
    for (int i = 0; i < WARMUPS; i++) {
      KERNEL_CALL_FP16
    }

    MyTimer timer(stream);
    timer.start();

    for (int i = 0; i < NUM_RUNS; i++) {
      KERNEL_CALL_FP16
    }

    static_cast<float *>(measurements)[0] = timer.end_and_measure() / NUM_RUNS;
  } else {
    KERNEL_CALL_FP16
    if (!features.flags.is_async) {
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  }

  return 0;
}
} // namespace doublesparse