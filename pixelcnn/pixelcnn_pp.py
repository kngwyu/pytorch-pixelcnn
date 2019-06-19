from functools import partial
import torch
from torch import nn, Tensor
from typing import Callable, List, Tuple
from .layers import ConcatELU, Conv1x1, DownShiftedConv2d, DownShiftedDeconv2d, GatedResNet
from .utils import down_shift, right_shift


class UpLayer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_channel: int,
            nonlinearity: Callable[[], nn.Module] = ConcatELU,
    ) -> None:
        super().__init__()
        self.up_pass = nn.ModuleList([
            GatedResNet(in_channel, DownShiftedConv2d, nonlinearity=nonlinearity())
            for _ in range(num_layers)
        ])
        self.up_left_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                partial(DownShiftedConv2d, kernel=(2, 2), right_shift=True),
                nonlinearity=nonlinearity(),
                aux_enlargement=1,
            ) for _ in range(num_layers)
        ])

    def forward(self, ux_cache: List[Tensor], ulx_cache: List[Tensor]) -> None:
        for up, up_left in zip(self.up_pass, self.up_left_pass):
            ux = up(ux_cache[-1])
            ulx = up_left(ulx_cache[-1], aux=ux)
            ux_cache.append(ux)
            ulx_cache.append(ulx)


class DownLayer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_channel: int,
            nonlinearity: Callable[[], nn.Module] = ConcatELU,
    ) -> None:
        super().__init__()
        self.up_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                DownShiftedConv2d,
                nonlinearity=nonlinearity(),
                aux_enlargement=1,
            ) for _ in range(num_layers)
        ])
        self.up_left_pass = nn.ModuleList([
            GatedResNet(
                in_channel,
                partial(DownShiftedConv2d, kernel=(2, 2), right_shift=True),
                nonlinearity=nonlinearity(),
                aux_enlargement=2,
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        ux: Tensor,
        ulx: Tensor,
        ux_skipped: List[Tensor],
        ulx_skipped: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        for up, up_left in zip(self.up_pass, self.up_left_pass):
            ux = up(ux, aux=ux_skipped.pop())
            ulx = up_left(ulx, aux=torch.cat((ux, ulx_skipped.pop()), dim=1))
        return ux, ulx


class PixelCNNpp(nn.Module):
    def __init__(
            self,
            input_channel: int,
            num_groups: int = 3,
            num_layers: int = 5,
            hidden_channel: int = 80,
            downsize_stride: int = 2,
            num_logistic_mix: int = 10,
            device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
            nonlinearity: Callable[[], nn.Module] = ConcatELU,
    ) -> None:
        super().__init__()
        # INITIALIZE
        self.init_u = DownShiftedConv2d(input_channel + 1, hidden_channel, kernel=(2, 3))
        self.init_ul = nn.ModuleList([
            DownShiftedConv2d(input_channel + 1, hidden_channel, kernel=(1, 3)),
            DownShiftedConv2d(input_channel + 1, hidden_channel, kernel=(2, 1), right_shift=True)
        ])

        # UP PASS
        hidden_channels = hidden_channel, hidden_channel
        self.up_pass = nn.ModuleList([
            UpLayer(num_layers, hidden_channel, nonlinearity)
            for _ in range(num_groups)
        ])
        self.downsize_u = nn.ModuleList([
            DownShiftedConv2d(*hidden_channels, stride=downsize_stride)
            for _ in range(num_groups - 1)
        ])
        self.downsize_ul = nn.ModuleList([
            DownShiftedConv2d(
                *hidden_channels,
                kernel=(2, 2),
                stride=downsize_stride,
                right_shift=True
            ) for _ in range(num_groups - 1)
        ])

        # DOWN PASS
        downpass_layers = [num_layers] + [num_layers + 1] * (num_groups - 1)
        self.down_pass = nn.ModuleList([
            DownLayer(resnet_layer, hidden_channel, nonlinearity)
            for resnet_layer in downpass_layers
        ])
        self.upsize_u = nn.ModuleList([
            DownShiftedDeconv2d(*hidden_channels, stride=downsize_stride)
            for _ in range(num_groups - 1)
        ])
        self.upsize_ul = nn.ModuleList([
            DownShiftedDeconv2d(
                *hidden_channels,
                kernel=(2, 2),
                stride=downsize_stride,
                right_shift=True
            )
            for _ in range(num_groups - 1)
        ])
        self.out = nn.Sequential(
            nn.ELU(inplace=True),
            Conv1x1(hidden_channel, num_logistic_mix * 10)
        )
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat((x, torch.ones_like(x[:, :1, ...])), dim=1)
        # UP PASS
        ux_cache = [down_shift(self.init_u(x))]
        ulx_cache = [down_shift(self.init_ul[0](x)) + right_shift(self.init_ul[1](x))]
        for up, downsize_u, downsize_ul in zip(self.up_pass, self.downsize_u, self.downsize_ul):
            up(ux_cache, ulx_cache)
            ux_cache.append(downsize_u(ux_cache[-1]))
            ulx_cache.append(downsize_ul(ulx_cache[-1]))
        self.up_pass[-1](ux_cache, ulx_cache)

        # DOWN PASS
        ux, ulx = ux_cache.pop(), ulx_cache.pop()
        for down, upsize_u, upsize_ul in zip(self.down_pass, self.upsize_u, self.upsize_ul):
            ux, ulx = down(ux, ulx, ux_cache, ulx_cache)
            ux = upsize_u(ux)
            ulx = upsize_ul(ulx)
        _, ulx = down(ux, ulx, ux_cache, ulx_cache)
        assert len(ux_cache) == 0 and len(ulx_cache) == 0
        return self.out(ulx)


if __name__ == '__main__':
    ''' testing loss compatibility '''
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    model = PixelCNNpp(3, num_layers=3)
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.constant_(param, 0.05)
        if 'bias' in name:
            nn.init.constant_(param, 1.0)
    out = model(x)
    print('out_mean: %s' % out.mean().item())
