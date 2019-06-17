import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from typing import Callable, Optional, Tuple
from .utils import down_cut, right_cut


class ConcatELU(nn.Module):
    __constants__ = ['alpha']

    def __init__(self, alpha: float = 1.) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(torch.cat((x, -x), dim=1), self.alpha, inplace=True)

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
        # pad: (Left, Right, Top, Bottom)
        pad = (kh - 1, 0, kw - 1, 0) if right_shift else ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0)
        self.pad = nn.ZeroPad2d(pad)
        self.conv = weight_norm(nn.Conv2d(in_channel, out_channel, kernel, stride))

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
        self.deconv = weight_norm(nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.kernel = kernel
        self.scaler = right_cut if right_shift else down_cut

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv(x)
        return self.scaler(x, *self.kernel)


class Conv1x1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class GatedResNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            conv: Callable[[int, int], nn.Module],
            nonlinearity: nn.Module = ConcatELU(),
            aux_enlargement: int = 0,
    ) -> None:
        super().__init__()
        nl_enlargement = 2 if isinstance(nonlinearity, ConcatELU) else 1
        self.conv1 = conv(in_channel * nl_enlargement, in_channel)
        if aux_enlargement == 0:
            self.skip_op = None
        else:
            self.skip_op = Conv1x1(nl_enlargement * aux_enlargement * in_channel, in_channel)
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout2d(0.5)
        self.conv2 = conv(nl_enlargement * in_channel, nl_enlargement * in_channel)

    def forward(self, x_orig: Tensor, aux: Optional[Tensor] = None) -> Tensor:
        x = self.conv1(self.nonlinearity(x_orig))
        if aux is not None and self.skip_op is not None:
            x += self.skip_op(self.nonlinearity(aux))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        c3 = x1 * torch.sigmoid(x2)
        return x_orig + c3
