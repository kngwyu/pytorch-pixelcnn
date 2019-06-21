from pixelcnn.loss import *
import torch


def test_dmix_loss():
    a, b = torch.randn(10, 3, 6, 6), torch.randn(10, 100, 6, 6)
    loss = discretized_mix_logistic_loss(a, b)
    assert loss.shape == torch.Size([])
