from pixelcnn.layers import *
import pytest
import torch


def test_downshift() -> None:
    x = torch.randn(2, 3, 4, 5)
    ds = down_shift(x)
    assert ds.shape == x.shape


def test_rightshift() -> None:
    x = torch.randn(2, 3, 4, 5)
    rs = right_shift(x)
    assert rs.shape == x.shape


def test_downcut() -> None:
    x = torch.randn(2, 3, 4, 5)
    dc = down_cut(x, 2, 2)
    assert tuple(dc.shape) == (2, 3, 3, 4)


def test_rightcut() -> None:
    x = torch.randn(2, 3, 4, 5)
    rc = right_cut(x, 2, 2)
    assert tuple(rc.shape) == (2, 3, 3, 4)


@pytest.mark.parametrize('net, out_shape', [
    (Conv1x1(3, 4), (5, 4, 6, 6)),
    (DownShiftedConv2d(3, 4), (5, 4, 6, 6)),
    (DownShiftedConv2d(3, 4, kernel=(2, 2), right_shift=True), (5, 4, 6, 6)),
    (DownShiftedDeconv2d(3, 4), (5, 4, 6, 6)),
    (DownShiftedDeconv2d(3, 4, right_shift=True), (5, 4, 6, 6)),
])
def test_conv(net: torch.nn.Module, out_shape: tuple) -> None:
    x = torch.randn(5, 3, 6, 6)
    out = net(x)
    assert tuple(out.shape) == out_shape
