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
    return ((torch.rand(n) - 0.5) * upper_bound).round().float()


def create_x_random(n, upper_bound=2):
    return generate_x_fp32(n, upper_bound).half()


def create_x_ones(n, upper_bound=2):
    return generate_x_fp32(n, upper_bound).half() * 0 + 1


def create_x_zeros(n):
    return generate_x_fp32(n, 1).half() * 0


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
    r = ((torch.rand(m, n) <= density) * (create_x_random(m * n).reshape(m, n)).float()).to_sparse_csr()
    return r


def random_doublesparse(m, n, k, density) -> DoubleSparseLegacy:
    a = random_csr_host(m, k, density)
    b = random_csr_host(k, n, density)
    return DoubleSparseLegacy(m, n, k, a, b)


def to_dense(double_sparse: DoubleSparseLegacy):
    return torch.matmul(double_sparse.a.to_dense().float().to(device='cuda'), double_sparse.b.to_dense().float().to(device='cuda')).half()
#
#
# class TestSparseFp16(unittest.TestCase):
#     def test_sparse_random(self):
#         # Call this once just to trigger the annoying torch sparse warning.
#         device = torch.device("cuda:0")
#
#         seed = 1
#         np.random.seed(seed)
#         torch.random.manual_seed(seed)
#
#         for m in [4096, 11008]:
#             for n in [4096, 11008]:
#                 for k in [11008, 4096]:
#                     for batch_size in [1, 2, 4, 8]:
#                         for density in [0.05, 0.1, 0.3]:
#                             print(f"Running m = {m} n = {n} k = {k} batch_size = {batch_size}")
#                             # Generate test case
#                             x = create_x_random(n * batch_size).cuda(device=device).half()
#                             ds = random_doublesparse(m, n, k, density)
#                             sparsified_linear = SparsifiedLinear.from_legacy(ds, device)
#                             y = sparsified_linear.forward(x.reshape((n, batch_size)).contiguous(), FeatureFlags.CSR).squeeze()
#                             dense_mat = to_dense(ds)
#                             x_reshaped = x.reshape((batch_size, n)).T.contiguous()
#                             y_matmul = (dense_mat.float() @ x_reshaped.float()).T.half().squeeze()
#                             passed = torch.equal(y, y_matmul)
#                             print(passed)
#                             self.assertTrue(
#                                 passed,
#                                 msg=f"Failed for m = {m} n = {n} k = {k} batch_size = {batch_size} density = {density}\ny={y}\nmatmul = {y_matmul}"
#                             )
#


class TestSparseFp8(unittest.TestCase):
    def test_sparse_random(self):
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        for m in [16]:
            for n in [4096, 11008]:
                for k in [4096, 11008]:
                    for batch_size in [1, 8]:
                        for density in [0.05, 0.3]:
                            print(f"Running m = {m} n = {n} k = {k} batch_size = {batch_size}")
                            # Generate test case
                            x = create_x_random(n * batch_size).cuda(device=device).half()
                            ds = random_doublesparse(m, n, k, density)
                            sparsified_linear = SparsifiedLinear.from_legacy(ds, device)
                            y = sparsified_linear.forward(x.reshape((n, batch_size)).contiguous(), FeatureFlags.CSR).squeeze()
                            dense_mat = to_dense(ds)
                            x_reshaped = x.reshape((batch_size, n)).T.contiguous()
                            y_matmul = (dense_mat.float() @ x_reshaped.float()).T.half().squeeze()
                            passed = torch.equal(y, y_matmul)
                            print(passed)
                            self.assertTrue(
                                passed,
                                msg=f"Failed for m = {m} n = {n} k = {k} batch_size = {batch_size} density = {density}\ny={y}\nmatmul = {y_matmul}"
                            )

if __name__ == "__main__":
    unittest.main()
