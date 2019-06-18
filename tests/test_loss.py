from numpy.testing import assert_array_almost_equal
from pixelcnn.loss import *
import torch


def test_siga_minus_sigb():
    a, b = torch.randn(10, 10), torch.randn(10, 10)
    c1 = torch.sigmoid(a) - torch.sigmoid(b)
    c2 = siga_minus_sigb(a, b)
    assert_array_almost_equal(c1.cpu().numpy(), c2.cpu().numpy())
