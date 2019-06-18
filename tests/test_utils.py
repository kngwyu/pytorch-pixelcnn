from pixelcnn.utils import *
import torch


def test_downshift():
    x = torch.randn(2, 3, 4, 5)
    ds = down_shift(x)
    assert ds.shape == x.shape


def test_rightshift():
    x = torch.randn(2, 3, 4, 5)
    rs = right_shift(x)
    assert rs.shape == x.shape


def test_downcut():
    x = torch.randn(2, 3, 4, 5)
    dc = down_cut(x, 2, 2)
    assert tuple(dc.shape) == (2, 3, 3, 5)


def test_rightcut():
    x = torch.randn(2, 3, 4, 5)
    rc = right_cut(x, 2, 2)
    assert tuple(rc.shape) == (2, 3, 3, 4)
