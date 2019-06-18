import torch
from torch import Tensor


@torch.jit.script
def down_shift(x: Tensor) -> Tensor:
    shape = x.shape
    zeros = torch.zeros_like(x[:, :, :1, :])
    return torch.cat([zeros, x[:, :, :shape[2] - 1, :]], dim=2)


@torch.jit.script
def right_shift(x: Tensor) -> Tensor:
    shape = x.shape
    zeros = torch.zeros_like(x[:, :, :, :1])
    return torch.cat([zeros, x[:, :, :, :shape[3] - 1]], dim=3)


@torch.jit.script
def down_cut(x: Tensor, hk: int, wk: int) -> Tensor:
    _, _, h, w = x.shape
    return x[:, :, :h - hk + 1, (wk - 1) // 2:w - (wk - 1) // 2]


@torch.jit.script
def right_cut(x: Tensor, hk: int, wk: int) -> Tensor:
    _, _, h, w = x.shape
    return x[:, :, :h - hk + 1, :w - wk + 1]
