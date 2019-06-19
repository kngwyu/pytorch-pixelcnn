
from pixelcnn.pixelcnn_pp import PixelCNNpp
import torch


def test_pixelcnn_pp():
    nn = PixelCNNpp(3)
    x = torch.randn(10, 3, 8, 8).to(nn.device)
    assert tuple(nn(x).shape) == (10, 100, 8, 8)
