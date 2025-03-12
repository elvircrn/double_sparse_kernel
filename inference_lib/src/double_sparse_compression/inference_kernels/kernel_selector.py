import torch

def get_doublesparse_sputnik():
    from .sputnik_kernels import CUDA_FOLDER
    return CUDA_FOLDER


def get_doublesparse_mul_timer():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.doublesparse_cuda.doublesparse_mul_timer



def get_doublesparse_mul():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.doublesparse_cuda.doublesparse_mul


