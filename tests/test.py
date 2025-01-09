import unittest
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from double_sparse_compression import SparsifiedLinear
from double_sparse_compression.inference import FeatureFlags, SparseStorageConfiguration, DoubleSparseLegacy, updiv
from double_sparse_compression.inference_kernels.kernel_selector import *

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def generate_x_fp32(n, upper_bound=2):
    x_fp32 = ((torch.rand(n) - 0.5) * 2).int()
    return x_fp32.float()


def create_x_random(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half()


def create_x_ones(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half() * 0 + 1


def create_x_zeros(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half() * 0


def random_like(v: torch.Tensor, upper_bound=1):
    return generate_x_fp32(v.shape[0], upper_bound).to(dtype=v.dtype, device=v.device)


def generate_3bit(n):
    upper_bound = 3
    x_char = (torch.rand(n) * upper_bound).char()
    return x_char


def generate_x_int32(n):
    upper_bound = 3
    x_int32 = ((torch.rand(n) - 0.5) * upper_bound).int()
    return x_int32


def random_csr_host(m, n, density):
    r = ((torch.rand(m, n) <= density) * (torch.ones(m, n) * 1).int()).to_sparse_csr()
    return r


def random_doublesparse(m, n, k, density) -> DoubleSparseLegacy:
    a = random_csr_host(m, k, density)
    b = random_csr_host(k, n, density)
    return DoubleSparseLegacy(m, n, k, a, b)


def to_dense(double_sparse: DoubleSparseLegacy):
    return torch.matmul(double_sparse.a.to_dense().half().to(device='cuda'), double_sparse.b.to_dense().half().to(device='cuda'))


def doublesparse_mul(sparsified_linear: SparsifiedLinear, x, y, featurea_flag, batch_size):
    return get_doublesparse_mul()(
        sparsified_linear.m,
        sparsified_linear.n,
        sparsified_linear.k,
        sparsified_linear.a_row_offsets,
        sparsified_linear.a_col_vals,
        sparsified_linear.b_row_offsets,
        sparsified_linear.b_col_vals,
        sparsified_linear.non_zero_rows,
        batch_size,
        x, featurea_flag, y, y
    )


class TestSparseFp16Easy(unittest.TestCase):
    def test_sparse_ones(self):
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # for m in [4096// 2, 11008// 2]:
        #     for n in [4096// 2, 11008// 2]:
        #         for k in [4096// 2, 11008// 2]:

        for m in [4096, 11008]:
            for n in [4096, 11008]:
                for k in [11008, 4096]:
                    for batch_size in [1, 2]:
                        for density in [0.1]:
                            for flag in [
                                FeatureFlags.CSR,
                            ]:
                                print(f"Running m = {m} n = {n} k = {k} batch_size = {batch_size}")
                                # Generate test case
                                x_fp32 = generate_x_fp32(n * batch_size)
                                x_fp16_device = x_fp32.cuda(device=device).half()

                                ds = random_doublesparse(m, n, k, 1.0)

                                dense = to_dense(ds).to(dtype=torch.half, device=device)

                                y_true = torch.matmul(dense, x_fp16_device.reshape(n, batch_size)).flatten()

                                y = torch.zeros(m * batch_size, dtype=torch.half, device=device)

                                sparsified_linear = SparsifiedLinear.from_legacy(ds, device)
                                doublesparse_mul(sparsified_linear, x_fp16_device, y, flag, batch_size)

                                passed = torch.equal(y, y_true)

                                self.assertTrue(
                                    passed,
                                    msg=f"Failed for m = {m} n = {n} k = {k} batch_size = {batch_size} density = {density}\ny={y}\ny_true={y_true}"
                                )


if __name__ == "__main__":

    unittest.main()
