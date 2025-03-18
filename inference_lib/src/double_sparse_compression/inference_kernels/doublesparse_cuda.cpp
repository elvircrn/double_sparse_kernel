/*
 * Copyright (C) Double Sparse Kernel.2024 Elvir Crncevic (elvircrn@gmail.com)
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
#include <torch/python.h> // One-stop header.
#include <torch/script.h> // One-stop header.

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
                                 uint32_t feature_flag);
};

namespace doublesparse {
int doublesparse_matmul(
    // W and meta
    int m, int n, int k,
    u32 *a_row_offsets,
    u32 *a_col_ids,
    u32 *b_row_offsets,
    u32 *b_col_ids,
    int non_zero_rows,
    int batch_size,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y,
    void *d_workspace,
    // GPU meta
    cudaStream_t stream = nullptr,
    void *measurements = nullptr,
    u32 feature_flag = 0);
int doublesparse_matmul_fp8(
    // W and meta
    int m, int n, int k,
    int *a_row_offsets,
    int *a_col_ids,
    int *b_row_offsets,
    int *b_col_ids,
    int non_zero_rows,
    int batch_size,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y,
    void *d_workspace,
    // GPU meta
    cudaStream_t stream = nullptr,
    void *measurements = nullptr,
    u32 feature_flag = 0);
};

void doublesparse_mul(
    int m, int n, int k,
    const torch::Tensor &a_row_offsets,
    const torch::Tensor &a_col_val,
    const torch::Tensor &b_row_offsets,
    const torch::Tensor &b_col_val,
    int non_zero_rows,
    int batch_size,
    const torch::Tensor &X,
    s32 f,
    const torch::Tensor &workspace,
    const torch::Tensor &Y,
    torch::Tensor &out) {
  u32 feature_flag = static_cast<u32>(f);
  int dev = a_row_offsets.get_device();

  int err = doublesparse::doublesparse_matmul(
      m, n, k,
      (u32*) a_row_offsets.data_ptr<int>(),
      (u32*) a_col_val.data_ptr<int>(),
      (u32*) b_row_offsets.data_ptr<int>(),
      (u32*) b_col_val.data_ptr<int>(),
      non_zero_rows,
      batch_size,
      X.data_ptr(),
      Y.data_ptr(),
      workspace.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev),
      nullptr, feature_flag);
}

void doublesparse_mul_timer(
    int m, int n, int k, const torch::Tensor &a_row_offsets,
    const torch::Tensor &a_col_val, const torch::Tensor &b_row_offsets,
    const torch::Tensor &b_col_val, int non_zero_rows, int batch_size,
    // 16-bit
    const torch::Tensor &X, torch::Tensor &Y, torch::Tensor &measurements,
    u32 feature_flag, const torch::Tensor &workspace) {
  int dev = a_row_offsets.get_device();
  Features features{._ = feature_flag};
  if (features.flags.is_fp8) {
      int err = doublesparse::doublesparse_matmul_fp8(
            m, n, k, a_row_offsets.data_ptr<int>(),
            a_col_val.data_ptr<int>(), b_row_offsets.data_ptr<int>(),
            b_col_val.data_ptr<int>(), non_zero_rows, batch_size,
            X.data_ptr(), Y.data_ptr(), workspace.data_ptr(),
            at::cuda::getCurrentCUDAStream(dev), measurements.data_ptr(),
            feature_flag);
  } else if (!features.flags.is_sputnik &&
      !features.flags.is_torch_sparse) {
    int err = doublesparse::doublesparse_matmul(
        m, n, k, (u32 *)a_row_offsets.data_ptr<int>(),
        (u32 *)a_col_val.data_ptr<int>(), (u32 *)b_row_offsets.data_ptr<int>(),
        (u32 *)b_col_val.data_ptr<int>(), non_zero_rows, batch_size,
        X.data_ptr(), Y.data_ptr(), workspace.data_ptr(),
        at::cuda::getCurrentCUDAStream(dev), measurements.data_ptr(),
        feature_flag);
  } else {
    int err = doublesparse_external::doublesparse_matmul_external_timer(
        m, n, k, a_row_offsets.data_ptr<int>(),
        a_col_val.data_ptr<int>(), b_row_offsets.data_ptr<int>(),
        b_col_val.data_ptr<int>(), non_zero_rows, batch_size,
        X.data_ptr(), Y.data_ptr(), workspace.data_ptr(),
        at::cuda::getCurrentCUDAStream(dev), measurements.data_ptr(),
        feature_flag);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("doublesparse_mul_timer", &doublesparse_mul_timer, "Double Sparse matmul with benchmarking.");
m.def("doublesparse_mul", &doublesparse_mul, "Double Sparse matmul.");
}
