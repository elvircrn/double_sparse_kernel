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
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

static constexpr u32 WARP_SIZE = 32;

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

struct FP16MatrixCSR {
  int rows;
  int *d_row_offsets;
  int *d_col_vals;
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
