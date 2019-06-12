from pixelcnn.model import down_shift, right_shift
import torch


def test_downshift():
    x = torch.randn(2, 3, 4, 5)
    ds = down_shift(x)
    assert ds.shape == x.shape


def test_rightshift():
    x = torch.randn(2, 3, 4, 5)
    rs = right_shift(x)
    assert rs.shape == x.shape
