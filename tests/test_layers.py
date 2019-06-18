from functools import partial
from pixelcnn.layers import *
import pytest
import torch


def test_concat_elu():
    x = torch.randn(2, 3, 4, 5)
    celu = ConcatELU()
    assert tuple(celu(x).shape) == (2, 6, 4, 5)


@pytest.mark.parametrize('net, out_shape', [
    (Conv1x1(3, 4), (5, 4, 6, 6)),
    (DownShiftedConv2d(3, 4), (5, 4, 6, 6)),
    (DownShiftedConv2d(3, 4, kernel=(2, 2), right_shift=True), (5, 4, 6, 6)),
    (DownShiftedDeconv2d(3, 4), (5, 4, 6, 6)),
    (DownShiftedDeconv2d(3, 3, stride=2), (5, 3, 12, 12)),
    (DownShiftedDeconv2d(3, 4, right_shift=True), (5, 4, 6, 6)),
])
def test_conv(net, out_shape):
    x = torch.randn(5, 3, 6, 6)
    out = net(x)
    assert tuple(out.shape) == out_shape


@pytest.mark.parametrize('net, out_shape, aux', [
    (GatedResNet(3, DownShiftedConv2d), (5, 3, 6, 6), None),
    (
        GatedResNet(
            3,
            partial(DownShiftedConv2d, kernel=(2, 2), right_shift=True),
            aux_enlargement=1,
        ),
        (5, 3, 6, 6),
        torch.randn(5, 3, 6, 6)
    ),
    (
        GatedResNet(
            3,
            partial(DownShiftedDeconv2d, kernel=(2, 2), right_shift=True),
            aux_enlargement=2,
        ),
        (5, 3, 6, 6),
        torch.randn(5, 6, 6, 6)
    )
])
def test_gated_resnet(net, out_shape, aux):
    x = torch.randn(5, 3, 6, 6)
    out = net(x, aux=aux)
    assert tuple(out.shape) == out_shape
