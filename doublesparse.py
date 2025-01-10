import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from double_sparse_compression.inference import SparsifiedLinear, DoubleSparseLegacy


def find_other2(A, W, nnz, Z, U, reg=0, rho_start=0.03, rho=1, iters=5, prune_iters=2, fixmask=None, debug=False, print_sc=None):
    XX = (A.T @ A).div_(torch.linalg.norm(A, dim=0) + 1e-8).T @ A
    diag_mean = XX.diagonal().mean()
    XX.diagonal().add_(diag_mean * (1 + reg))
    eye = torch.eye(XX.size(0), device=XX.device, dtype=XX.dtype)
    XXinv = torch.linalg.inv(XX + eye * rho)
    XXinv2 = torch.linalg.inv(XX + eye * rho_start)

    norm2 = torch.linalg.norm(A, dim=0) + 1e-8
    U *= norm2.unsqueeze(1)
    Z *= norm2.unsqueeze(1)
    XY = (A / norm2).T @ W
    B = XXinv2 @ (XY + rho_start * (Z - U))
    bsparsity = min(0.99, 1 - nnz / B.numel())

    for itt in range(iters):
        if itt < prune_iters and fixmask is None:
            thres = (B + U).abs().kthvalue(int(B.numel() * bsparsity)).values
            mask = (B + U).abs() > thres
        elif fixmask is not None:
            mask = fixmask

        Z = (B + U) * mask
        U += B - Z
        B = XXinv @ (XY + rho * (Z - U))

        if debug:
            print(itt, bsparsity, (Z != 0).sum().item() / Z.numel())
            if print_sc:
                print_sc(A @ (B / norm2.unsqueeze(1)))
                print_sc(A @ (Z / norm2.unsqueeze(1)))
            print(((A != 0).sum() + (Z != 0).sum()) / W.numel())
            print("-------")

    return Z / norm2.unsqueeze(1), U / norm2.unsqueeze(1)


def mag_prune(W, sp=0.6):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return W * mask


def ent(p):
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def factorizeT(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    if fixmask is None:
        nza = int(W.shape[0] ** 2 * asp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)

    Az = torch.eye(W.shape[0], device=W.device)
    Au = torch.zeros_like(Az)
    norm = XX.diagonal().sqrt_().unsqueeze(1) + 1e-8
    Wn = W * norm
    Bz = mag_prune(Wn, (1 - nzb / 2 / W.numel()))
    Bu = torch.zeros_like(Bz)
    for itt in range(iters):
        rho_start = min(1.0, itt / (iters - 3)) ** 3
        Az, Au = map(torch.t, find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, rho_start=rho_start, fixmask=fixmask))
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, rho_start=rho_start)

    Az_nnz = (Az != 0).sum().item()
    Bz_nnz = (Bz != 0).sum().item()
    total_nnz = Az_nnz + Bz_nnz
    ent_az = ent(Az_nnz / Az.numel())
    ent_bz = ent(Bz_nnz / Bz.numel())
    total_ent = (Az.numel() * ent_az + Bz.numel() * ent_bz) / W.numel()

    print(total_nnz / W.numel(), Az_nnz / Az.numel(), Bz_nnz / Bz.numel(), Az.shape, Bz.shape, total_ent, ent(0.4), ent(0.5))
    return (Az / norm) @ Bz, Bz.T, (Az / norm).T


def factorizef(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    if W.shape[0] >= W.shape[1]:
        return factorizeT(W.T, XX, asp, sp, fixmask)

    nza = (fixmask != 0).sum().item() if fixmask is not None else int(W.shape[0] ** 2 * asp)
    nzb = int(W.numel() * sp - nza)
    if nzb < 0:
        raise ValueError(f"Sparsity set too high: {sp}")
    norm = XX.diagonal().sqrt_() + 1e-8
    Wn = W * norm

    Az = torch.eye(W.shape[0], device=W.device)
    Au = torch.zeros_like(Az)
    Bz = mag_prune(Wn, 1 - nzb / (2 * W.numel()))
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        rho_start = min(1.0, itt / (iters - 3)) ** 3
        Az, Au = map(torch.t, find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, rho_start=rho_start, fixmask=fixmask))
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, rho_start=rho_start)

    Az_nnz = (Az != 0).sum().item()
    Bz_nnz = (Bz != 0).sum().item()
    total_nnz = Az_nnz + Bz_nnz
    ent_az = ent(Az_nnz / Az.numel())
    ent_bz = ent(Bz_nnz / Bz.numel())
    total_ent = (Az.numel() * ent_az + Bz.numel() * ent_bz) / W.numel()

    print(total_nnz / W.numel(), Az_nnz / Az.numel(), Bz_nnz / Bz.numel(), Az.shape, Bz.shape, total_ent, ent(0.4), ent(0.5))
    return Az @ (Bz / norm), Az, Bz / norm



def finalize(XXb, W, Ab, Bb):
    fsparsity = 1 - (Ab != 0).sum() / Ab.numel()
    mask = (Ab != 0).T

    XX = Bb.matmul(XXb).matmul(Bb.T)
    XY = Bb.matmul(XXb).matmul(W.T)

    norm2 = torch.diag(XX).sqrt() + 1e-8
    XX = XX / norm2 / norm2.unsqueeze(1)
    XY = XY / norm2.unsqueeze(1)
    Ax = (Ab * norm2).T.clone()
    # Ax = torch.linalg.solve(XX, XY)

    rho = 1
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device) * rho)
    U = torch.zeros_like(Ax)
    for itt in range(20):
        Z = (Ax + U) * mask

        U = U + (Ax - Z)

        Ax = XXinv.matmul(XY + rho * (Z - U))

    Ac = Z.T / norm2
    return Ac


def factorize(W, XX, sparsity, nofinal=False, fixmask=None):
    if W.shape[0] == W.shape[1]:
        asp = 0.16
    else:
        asp = 0.25
    W2, Ab, Bb = factorizef(W, XX, asp=asp, sp=1 - sparsity, fixmask=fixmask)
    print("err_prefin", (W2 - W).matmul(XX).matmul((W2 - W).T).diag().sum().item())
    if nofinal:
        return W2, Ab.cpu(), Bb.cpu()
    Ac = finalize(XX, W, Ab, Bb)
    W3 = Ac.matmul(Bb)
    assert W3.shape == W.shape
    print("err_fin   ", (W3 - W).matmul(XX).matmul((W3 - W).T).diag().sum().item())
    print("sparsity check", ((Ac != 0).sum() + (Bb != 0).sum()).item() / W.numel())
    return W3, Ac.cpu(), Bb.cpu()


class DoubleSparse:
    def __init__(self, layer, nofinal=False, fixmask=None):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.nofinal = nofinal
        self.fixmask = fixmask

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
            self, sparsity,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            raise AttributeError("Conv not supported")
        if isinstance(self.layer, transformers.Conv1D):
            raise AttributeError("Conv not supported")
        W = W.float()
        tick = time.time()

        W2, A, B = factorize(W, self.H, sparsity, nofinal=self.nofinal, fixmask=self.fixmask)
        self.A = A.to_sparse_csr()
        self.B = B.to_sparse_csr()

        m, k = A.shape
        k, n = B.shape

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        sparsified_linear = SparsifiedLinear.from_legacy(
            DoubleSparseLegacy(m, n, k, self.A, self.B),
            self.layer.weight.device
        )
        self.layer = sparsified_linear
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        return sparsified_linear

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
