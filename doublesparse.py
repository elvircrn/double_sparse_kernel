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

def find_other2(A, W, nnz, Z, U, print_sc=None, debug=False, reg=0, rho_start=0.03, iters=5, prune_iters=2,
                fixmask=None):
    XX = A.T @ A
    norm2 = torch.linalg.norm(A, dim=0) + 1e-8
    An = A / norm2
    XX = An.T @ An
    XX.diagonal().add_(XX.diagonal().mean() * reg)

    eye = torch.eye(XX.shape[1], device=XX.device)
    XXinv = torch.linalg.inv(XX + eye * rho)
    XXinv2 = torch.linalg.inv(XX + eye * rho_start)

    Wnn = W
    XY = An.T @ Wnn

    U *= norm2.unsqueeze(1)
    Z *= norm2.unsqueeze(1)

    B = XXinv2 @ (XY + rho_start * (Z - U))

    bsparsity = min(0.99, 1 - nnz / B.numel())

    for itt in range(iters):
        if itt < prune_iters and fixmask is None:
            BU = (B + U).abs()
            thres = BU.flatten().kthvalue(int(B.numel() * bsparsity)).values
            mask = BU.abs() > thres
        elif fixmask is not None:
            mask = fixmask

        Z = (B + U) * mask
        U += B - Z
        B = L_inv_rho @ (XY + rho * (Z - U))

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
    # W = lx.weight.detach().T.float()
    if fixmask is None:
        nza = int(W.shape[0] ** 2 * asp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)

    Az = torch.eye(W.shape[0], device=W.device)
    Au = torch.zeros_like(Az)
    norm = XX.diag().sqrt().unsqueeze(1) + 1e-8

    Wn = W * norm

    Bz = mag_prune(Wn, (1 - nzb / 2 / W.numel()))
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        # if itt < 10:
        #    rho_start = 0.0
        # elif itt < 15:
        #    rho_start = 0.00
        # else:
        #    rho_start = 0.1
        rho_start = min(1.0, itt / (iters - 3)) ** 3
        Az, Au = (x.T for x in
                  find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))

        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, debug=False, rho_start=rho_start)

    print(((Az != 0).sum() + (Bz != 0).sum()).item() / W.numel(), (Az != 0).sum().item() / Az.numel(),
          (Bz != 0).sum().item() / Bz.numel(), Az.shape, Bz.shape,
          (Az.numel() * ent((Az != 0).sum().item() / Az.numel()) + Bz.numel() * ent(
              (Bz != 0).sum().item() / Bz.numel())) / W.numel(),
          ent(0.4), ent(0.5))
    return ((Az / norm).matmul(Bz)).T, Bz.T, (Az / norm).T


def factorizef(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    s_time = time.time()
    if W.shape[0] >= W.shape[1]:
        return factorizeT(W.T, XX, asp, sp=sp, fixmask=fixmask)

    if fixmask is None:
        nza = int(W.shape[0] ** 2 * asp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)
    if nzb < 0:
        raise f'Error: sparsity set to high at {sp}'
    norm = XX.diag().sqrt() + 1e-8

    Wn = W * norm

    Az = torch.eye(W.shape[0], device=W.device)
    Au = torch.zeros_like(Az)

    Bz = mag_prune(Wn, (1 - nzb / 2 / W.numel()))
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        # if itt < 10:
        #    rho_start = 0.0
        # elif itt < 15:
        #    rho_start = 0.00
        # else:
        #    rho_start = 0.1

        rho_start = min(1.0, itt / (iters - 3)) ** 3
        Az, Au = (x.T for x in
                  find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))

        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, debug=False, rho_start=rho_start)

        # print(itt, time.time() - s_time, end =" ")
        # print_scores(Az.matmul(Bz / norm))

    print(((Az != 0).sum() + (Bz != 0).sum()).item() / W.numel(), (Az != 0).sum().item() / Az.numel(),
          (Bz != 0).sum().item() / Bz.numel(), Az.shape, Bz.shape,
          (Az.numel() * ent((Az != 0).sum().item() / Az.numel()) + Bz.numel() * ent(
              (Bz != 0).sum().item() / Bz.numel())) / W.numel(),
          ent(0.4), ent(0.5))
    return Az.matmul(Bz / norm), Az, Bz / norm


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
