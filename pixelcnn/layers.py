import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch._jit_internal import weak_module, weak_script_method
from typing import Any, Optional, Tuple


def identity(x: Any) -> Any:
    return x


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


@weak_module
class ConcatElu(nn.Module):
    __constants__ = ['alpha', 'inplace']

    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    @weak_script_method
    def forward(self, x: Tensor) -> Tensor:
        axis = len(x.shape) - 1
        return F.elu(torch.cat((x, -x), dim=axis), self.alpha, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


class DownShiftedConv2d(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel: Tuple[int, int],
            stride: Tuple[int, int],
    ) -> None:
        super().__init__()
        side_pad = (kernel[1] - 1) // 2
        self.pad = nn.ZeroPad2d((side_pad, side_pad, kernel[0] - 1, 0))
        self.conv = weight_norm(nn.Conv2d(in_channel, out_channel, kernel, stride))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class DownShiftedDeconv2d(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel: Tuple[int, int],
            stride: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.deconv = weight_norm(nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
