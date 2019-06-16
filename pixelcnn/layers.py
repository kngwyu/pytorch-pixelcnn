import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch._jit_internal import weak_module, weak_script_method
from typing import Callable, Optional, Tuple


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
    return x[:, :, :h - hk + 1, (wk - 1) // 1:w - (wk - 1) // 2]


@torch.jit.script
def right_cut(x: Tensor, hk: int, wk: int) -> Tensor:
    _, _, h, w = x.shape
    return x[:, :, :h - hk + 1, :w - wk + 1]


@weak_module
class ConcatELU(nn.Module):
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
            kernel: Tuple[int, int] = (2, 3),
            stride: Tuple[int, int] = (1, 1),
            right_shift: bool = False,
    ) -> None:
        super().__init__()
        kh, kw = kernel
        pad = (kh - 1, 0, kw - 1, 0) if right_shift else ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0)
        self.pad = nn.ZeroPad2d(pad)  # pad: (Left, Right, Top, Bottom)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride)
        self.conv = weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class DownShiftedDeconv2d(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel: Tuple[int, int] = (2, 2),
            stride: Tuple[int, int] = (1, 1),
            right_shift: bool = False,
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride)
        self.deconv = weight_norm(self.deconv)
        self.kernel = kernel
        self.scaler = right_cut if right_shift else down_cut

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv(x)
        return self.scaler(x, *self.kernel)


class Conv1x1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.linear = weight_norm(nn.Linear(in_channel, out_channel))

    def forward(self, x: Tensor) -> Tensor:
        xs = x.shape
        x = self.linear(x.permute(0, 2, 3, 1).reshape(-1, xs[1]))
        return x.view(xs[0], *xs[2:], -1).permute(0, 3, 1, 2)


class GatedResNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            conv: Callable[[int, int], nn.Module],
            nonlinearity: nn.Module = ConcatELU,
            aux_enlargement: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = conv(in_channel * 2, in_channel)
        if aux_enlargement == 0:
            self.skip_op = None
        else:
            self.skip_op = Conv1x1(2 * aux_enlargement * in_channel, in_channel)
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout2d(0.5)
        self.conv2 = conv(in_channel * 2, in_channel * 2)

    def forward(self, x_orig: Tensor, aux: Optional[Tensor] = None) -> Tensor:
        x = self.conv1(self.nonlinearity(x_orig))
        if aux is not None and self.skip_op is not None:
            x += self.skip_op(self.nonlinearity(x))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        c3 = x1 * F.sigmoid(x2)
        return x_orig + c3
