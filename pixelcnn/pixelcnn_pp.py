from functools import partial
import torch
from torch import nn, Tensor
from typing import Callable, List, Tuple
from .layers import ConcatELU, DownShiftedConv2d, DownShiftedDeconv2d, GatedResNet
from .layers import down_cut, down_shift, right_cut, right_shift


class UpLayer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_channel: int,
            nonlinearity: Callable[[], nn.Module] = ConcatELU,
    ) -> None:
        super().__init__()
        self.down_pass = nn.ModuleList([
            GatedResNet(in_channel, DownShiftedConv2d, nonlinearity=nonlinearity)
            for _ in range(num_layers)
        ])
        self.down_right_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                partial(DownShiftedConv2d, kernel=(2, 2), right_shift=True),
                nonlinearity=nonlinearity,
                aux_enlargement=1,
            ) for _ in range(num_layers)
        ])

    def forward(
            self,
            dx: Tensor,
            drx: Tensor,
            dx_cache: List[Tensor],
            drx_cache: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        for down, down_right in zip(self.down_pass, self.down_right_pass):
            dx = down(dx)
            drx = down_right(drx, aux=dx)
            dx_cache.append(dx)
            drx_cache.append(drx)


class DownLayer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_channel: int,
            nonlinearity: Callable[[], nn.Module] = ConcatELU,
    ) -> None:
        super().__init__()
        self.down_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                DownShiftedDeconv2d,
                nonlinearity=nonlinearity,
            )
            for _ in range(num_layers)
        ])
        self.down_right_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                partial(DownShiftedDeconv2d, kernel=(2, 2), right_shift=True),
                nonlinearity=nonlinearity,
                aux_enlargement=2,
            ) for _ in range(num_layers)
        ])

        def forward(
            self,
            dx: Tensor,
            drx: Tensor,
            dx_skipped: List[Tensor],
            drx_skipped: List[Tensor],
        ) -> Tuple[Tensor, Tensor]:
            for down, down_right in zip(self.down_pass, self.down_right_pass):
                dx = down(dx, aux=dx_skipped.pop())
                drx = down(dx, aux=torch.cat(dx, drx_skipped.pop(), dim=1))
            return dx, drx


class PixelCNNpp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
