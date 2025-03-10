#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
import torch
from torch import Tensor as T, nn
from double_sparse_compression.inference_kernels.kernel_selector import get_doublesparse_mul

# from .inference_kernels.cuda_kernel import (
#     call_dequantize_compressed,
#     call_spqr_mul,
#     call_spqr_mul_fused,
#     call_tensor_compress_interleaved, call_spqr_mul_batched,
# )
from .sparse_util import init_ptcsr, merge_col_val

ASYNC = 1 << 0
IS_CSC = 1 << 1
IS_NAIVE = 1 << 3
IS_FP8 = 1 << 4
IS_TORCH_SPARSE = 1 << 5


class FeatureFlags(IntEnum):
    CSR = 0
    CSC = IS_CSC
    CSR_ASYNC = CSR | ASYNC
    CSR_NAIVE = CSR | IS_NAIVE
    CSR_NAIVE_ASYNC = CSR | IS_NAIVE | ASYNC
    CSR_FP8 = CSR | IS_FP8
    CSR_FP8_ASYNC = CSR | IS_FP8 | ASYNC
    CSR_TORCH_SPARSE = CSR | IS_TORCH_SPARSE
    CSR_TORCH_SPARSE_ASYNC = CSR | IS_TORCH_SPARSE | ASYNC

    def pretty(self):
        if self.value == self.CSR:
            return "CSR"
        elif self.value == self.CSC:
            return "CSC"


# Utility functions
class SparseStorageConfiguration(StrEnum):
    CSR = "csr"
    PTCSR = "ptcsr"
    OPTIMIZE_LATENCY = "optimize_latency"


@dataclass
class DoubleSparseLegacy:
    m: int
    n: int
    k: int
    a: torch.Tensor
    b: torch.Tensor


class SparsifiedLinearFP8(torch.nn.Module):
    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 a_row_offsets: torch.Tensor,
                 a_col_vals: torch.Tensor,
                 b_row_offsets: torch.Tensor,
                 b_col_vals: torch.Tensor,
                 non_zero_rows: int):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k

        self.a_row_offsets = nn.Parameter(a_row_offsets, requires_grad=False)
        self.a_col_vals = nn.Parameter(a_col_vals, requires_grad=False)
        self.b_row_offsets = nn.Parameter(b_row_offsets, requires_grad=False)
        self.b_col_vals = nn.Parameter(b_col_vals, requires_grad=False)
        self.non_zero_rows = non_zero_rows
        self.workspace = None

    @staticmethod
    def _from_legacy(double_sparse_legacy: DoubleSparseLegacy, device):
        a_sparse = double_sparse_legacy.a
        b_sparse = double_sparse_legacy.b
        non_zero_row_count = double_sparse_legacy.k
        mod = SparsifiedLinear(
            double_sparse_legacy.m,
            double_sparse_legacy.n,
            double_sparse_legacy.k,
            a_sparse.crow_indices().int(),
            merge_col_val(a_sparse.col_indices().short(), a_sparse.values().half()),
            b_sparse.crow_indices().int(),
            merge_col_val(b_sparse.col_indices().short(), b_sparse.values().half()),
            non_zero_row_count
        )
        return mod.to(device=device)

    @staticmethod
    def from_legacy(double_sparse_legacy: DoubleSparseLegacy, device):
        b_sparse = double_sparse_legacy.b

        if True:
            b_row_offsets = b_sparse.crow_indices()
            row_counts = torch.diff(b_row_offsets)
            non_zero_row_count = (row_counts != 0).sum().item()

            row_counts_sorted, row_ids = row_counts.sort(descending=False)

            non_zero_row_ids_sorted = row_ids[row_counts_sorted != 0]
            zero_row_ids_sorted = row_ids[row_counts_sorted == 0]

            a_dense = double_sparse_legacy.a.to_dense()

            reordered_row_ids = torch.concat((non_zero_row_ids_sorted, zero_row_ids_sorted))

            a_dense = a_dense[:, reordered_row_ids]
            a_dense[:, non_zero_row_ids_sorted.shape[0]:] = 0
            b_dense = b_sparse.to_dense()[reordered_row_ids, :]
            a_sparse = a_dense.to_sparse_csr()
            b_sparse = b_dense.to_sparse_csr()
        else:
            a_sparse = double_sparse_legacy.a.to_dense().to_sparse_csr()
            b_sparse = double_sparse_legacy.b.to_dense().to_sparse_csr()
            non_zero_row_count = double_sparse_legacy.k

        mod = SparsifiedLinear(
            double_sparse_legacy.m,
            double_sparse_legacy.n,
            double_sparse_legacy.k,
            a_sparse.crow_indices().int(),
            merge_col_val(a_sparse.col_indices().short(), a_sparse.values().half()),
            b_sparse.crow_indices()[:(non_zero_row_count + 1)].int(),
            merge_col_val(b_sparse.col_indices().short(), b_sparse.values().half()),
            non_zero_row_count
        )

        return mod.to(device=device)

    def dequantize_dense_only(self):
        """
        Dequantize the matrix using only the dense weight data.

        Possibly useful during debugging, testing or benchmarking.

        @return: Dense-only dequantized SpQR matrix.
        """
        return self._spqr_dequantize(0)

    def dequantize(self):
        """
        @return: Dequantized SpQR matrix.
        """
        return self._spqr_dequantize(self.col_vals.shape[0])

    @property
    def nnz(self):
        """
        @return: Number of non-zeros of the sparse matrix, optionally including zero-padding in the case of PTCSR.
        """
        return self.col_vals.shape[0]

    @property
    def density(self) -> float:
        """
        @return: Sparse matrix density.
        """
        return self.col_vals.shape[0] / (self.m * self.n)

    @property
    def sparsity(self) -> float:
        """
        @return: Sparsity.
        """
        return 1 - self.density

    @torch.no_grad()
    def forward(self, x: T, flag=FeatureFlags.CSR_ASYNC) -> T:
        """
        Forward matmul operation.
        @param x: Input tensor.
        @param flag: Feature flag.
        @return: A tensor resulting from a multiplication between the Dobuel Sparse tensor and input tensor x.
        """
        if not hasattr(self, 'K'): self.K = get_doublesparse_mul()
        batch_size = x.shape[1]
        y = torch.empty((1, batch_size, self.m), dtype=torch.float16, device=x.device)
        workspace = self.workspace
        if workspace is None or batch_size != 1: workspace = torch.empty(self.k * batch_size, dtype=torch.float32, device=x.device).contiguous()
        self.K(
            self.m,
            self.n,
            self.k,
            self.a_row_offsets,
            self.a_col_vals,
            self.b_row_offsets,
            self.b_col_vals,
            self.non_zero_rows,
            batch_size,
            x,
            flag,
            workspace,
            y,
            y
        )
        return y



class SparsifiedLinear(torch.nn.Module):
    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 a_row_offsets: torch.Tensor,
                 a_col_vals: torch.Tensor,
                 b_row_offsets: torch.Tensor,
                 b_col_vals: torch.Tensor,
                 non_zero_rows: int):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k

        self.a_row_offsets = nn.Parameter(a_row_offsets, requires_grad=False)
        self.a_col_vals = nn.Parameter(a_col_vals, requires_grad=False)
        self.b_row_offsets = nn.Parameter(b_row_offsets, requires_grad=False)
        self.b_col_vals = nn.Parameter(b_col_vals, requires_grad=False)
        self.non_zero_rows = non_zero_rows
        self.workspace = None

    @staticmethod
    def _from_legacy(double_sparse_legacy: DoubleSparseLegacy, device):
        a_sparse = double_sparse_legacy.a
        b_sparse = double_sparse_legacy.b
        non_zero_row_count = double_sparse_legacy.k
        mod = SparsifiedLinear(
            double_sparse_legacy.m,
            double_sparse_legacy.n,
            double_sparse_legacy.k,
            a_sparse.crow_indices().int(),
            merge_col_val(a_sparse.col_indices().short(), a_sparse.values().half()),
            b_sparse.crow_indices().int(),
            merge_col_val(b_sparse.col_indices().short(), b_sparse.values().half()),
            non_zero_row_count
        )
        return mod.to(device=device)

    @staticmethod
    def from_legacy(double_sparse_legacy: DoubleSparseLegacy, device):
        b_sparse = double_sparse_legacy.b

        if True:
            b_row_offsets = b_sparse.crow_indices()
            row_counts = torch.diff(b_row_offsets)
            non_zero_row_count = (row_counts != 0).sum().item()

            row_counts_sorted, row_ids = row_counts.sort(descending=False)

            non_zero_row_ids_sorted = row_ids[row_counts_sorted != 0]
            zero_row_ids_sorted = row_ids[row_counts_sorted == 0]

            a_dense = double_sparse_legacy.a.to_dense()

            reordered_row_ids = torch.concat((non_zero_row_ids_sorted, zero_row_ids_sorted))

            a_dense = a_dense[:, reordered_row_ids]
            a_dense[:, non_zero_row_ids_sorted.shape[0]:] = 0
            b_dense = b_sparse.to_dense()[reordered_row_ids, :]
            a_sparse = a_dense.to_sparse_csr()
            b_sparse = b_dense.to_sparse_csr()
        else:
            a_sparse = double_sparse_legacy.a.to_dense().to_sparse_csr()
            b_sparse = double_sparse_legacy.b.to_dense().to_sparse_csr()
            non_zero_row_count = double_sparse_legacy.k

        mod = SparsifiedLinear(
            double_sparse_legacy.m,
            double_sparse_legacy.n,
            double_sparse_legacy.k,
            a_sparse.crow_indices().int(),
            merge_col_val(a_sparse.col_indices().short(), a_sparse.values().half()),
            b_sparse.crow_indices()[:(non_zero_row_count + 1)].int(),
            merge_col_val(b_sparse.col_indices().short(), b_sparse.values().half()),
            non_zero_row_count
        )

        return mod.to(device=device)

    def dequantize_dense_only(self):
        """
        Dequantize the matrix using only the dense weight data.

        Possibly useful during debugging, testing or benchmarking.

        @return: Dense-only dequantized SpQR matrix.
        """
        return self._spqr_dequantize(0)

    def dequantize(self):
        """
        @return: Dequantized SpQR matrix.
        """
        return self._spqr_dequantize(self.col_vals.shape[0])

    @property
    def nnz(self):
        """
        @return: Number of non-zeros of the sparse matrix, optionally including zero-padding in the case of PTCSR.
        """
        return self.col_vals.shape[0]

    @property
    def density(self) -> float:
        """
        @return: Sparse matrix density.
        """
        return self.col_vals.shape[0] / (self.m * self.n)

    @property
    def sparsity(self) -> float:
        """
        @return: Sparsity.
        """
        return 1 - self.density

    @torch.no_grad()
    def forward(self, x: T, flag=FeatureFlags.CSR_ASYNC) -> T:
        """
        Forward matmul operation.
        @param x: Input tensor.
        @param flag: Feature flag.
        @return: A tensor resulting from a multiplication between the Dobuel Sparse tensor and input tensor x.
        """
        if not hasattr(self, 'K'): self.K = get_doublesparse_mul()
        batch_size = x.shape[1]
        y = torch.empty((1, batch_size, self.m), dtype=torch.float16, device=x.device)
        workspace = self.workspace
        if workspace is None or batch_size != 1: workspace = torch.empty(self.k * batch_size, dtype=torch.float32, device=x.device).contiguous()
        self.K(
            self.m,
            self.n,
            self.k,
            self.a_row_offsets,
            self.a_col_vals,
            self.b_row_offsets,
            self.b_col_vals,
            self.non_zero_rows,
            batch_size,
            x,
            flag,
            workspace,
            y,
            y
        )
        return y


def updiv(x, y):
    """
    @return: Utility method: updivision between x and y.
    """
    return (x + y - 1) // y
