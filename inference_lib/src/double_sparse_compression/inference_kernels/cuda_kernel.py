import os

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
SPQR_CUDA = load(
    name="doublesparse_cuda",
    sources=[os.path.join(CUDA_FOLDER, "doublesparse_cuda.cpp"),
             os.path.join(CUDA_FOLDER, "doublesparse_cuda_kernel.cu")],
    extra_cflags=[""],
    extra_cuda_cflags=["-lineinfo -O3 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89"]
)

torch.library.define(
    "doublesparse_cuda"
    "::doublesparse_mul",
    "(int m, int n, int k, Tensor a_row_offsets, Tensor a_col_vals, Tensor b_row_offsets, Tensor b_col_vals, int non_zero_rows, int batch_size, Tensor x, int f, Tensor workspace, Tensor Y, Tensor(Y!) out) -> ()",
)
torch.library.define(
    "doublesparse_cuda::doublesparse_mul_timer",
    "(int m, int n, int k, Tensor a_row_offsets, Tensor a_col_vals, Tensor b_row_offsets, Tensor b_col_vals, int non_zero_rows, int batch_size, Tensor x, Tensor Y, Tensor measurements, int f, Tensor workspace) -> ()",
)

torch.library.impl("doublesparse_cuda::doublesparse_mul_timer", "default", SPQR_CUDA.doublesparse_mul_timer)
torch.library.impl("doublesparse_cuda::doublesparse_mul", "default", SPQR_CUDA.doublesparse_mul)


def call_doublesparse_mul(*args):
    return torch.ops.doublesparse_cuda.doublesparse_mul(*args)


def call_doublesparse_mul_timer(*args):
    return torch.ops.doublesparse_cuda.doublesparse_mul_timer(*args)


@torch.library.register_fake("doublesparse_cuda::doublesparse_mul")
def spqr_mul_meta(m, n, k, a_row_offsets, a_col_vals, b_row_offsets, b_col_vals, non_zero_rows, batch_size, x, f, workspace, Y, out):
    return
