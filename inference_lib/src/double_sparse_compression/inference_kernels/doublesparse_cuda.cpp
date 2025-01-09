/*
 * Copyright (C) SPQR Kernel.2024 Elvir Crncevic (elvircrn@gmail.com)
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
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/python.h>
#include <torch/script.h> // One-stop header.
#include <vector>

int doublesparse_matmul(
    // W and meta
    int m, int n, int k,
    void *a_row_offsets,
    void *a_col_ids,
    void *b_row_offsets,
    void *b_col_ids,
    int non_zero_rows,
    int batch_size,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y,
    // GPU meta
    cudaStream_t stream = nullptr,
    void *measurements = nullptr,
    u32 feature_flag = 0);

void doublesparse_mul(
    int m, int n, int k,
    const torch::Tensor &a_row_offsets,
    const torch::Tensor &a_col_val,
    const torch::Tensor &b_row_offsets,
    const torch::Tensor &b_col_val,
    int non_zero_rows,
    int batch_size,
    const torch::Tensor &X, s64 _feature_flag,
    const torch::Tensor &Y, torch::Tensor &out) {
  u32 feature_flag = static_cast<u32>(_feature_flag);
  int dev = a_row_offsets.get_device();

  int err = doublesparse_matmul(
      m, n, k,
      a_row_offsets.data_ptr(),
      a_col_val.data_ptr(),
      b_row_offsets.data_ptr(),
      b_col_val.data_ptr(),
      non_zero_rows,
      batch_size,
      X.data_ptr(), Y.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev),
      nullptr, feature_flag);
}


void doublesparse_mul_timer(
    int m, int n, int k,
    const torch::Tensor &a_row_offsets,
    const torch::Tensor &a_col_val,
    const torch::Tensor &b_row_offsets,
    const torch::Tensor &b_col_val,
    int non_zero_rows,
    int batch_size,
    // 16-bit
    const torch::Tensor &X, torch::Tensor &Y,
    torch::Tensor &measurements, u32 feature_flag
) {
  int dev = a_row_offsets.get_device();

  int err = doublesparse_matmul(
      m, n, k,
      a_row_offsets.data_ptr(),
      a_col_val.data_ptr(),
      b_row_offsets.data_ptr(),
      b_col_val.data_ptr(),
      non_zero_rows,
      batch_size,
      X.data_ptr(), Y.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev),
      measurements.data_ptr(), feature_flag);

}

enum class SparseCompressionStrategy { CSR = 0, PTCSR = 1 };


#ifndef PYBIND_SKIP
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("doublesparse_mul_timer", &doublesparse_mul_timer, "SPQR matvec.");
m.def("doublesparse_mul", &doublesparse_mul, "SPQR matvec.");
}
#endif
