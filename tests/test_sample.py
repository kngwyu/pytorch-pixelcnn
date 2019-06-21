from pixelcnn import sample
import torch


def test_dmix_sample():
    s = sample.from_discretized_mix_logistic(torch.randn(10, 100, 6, 6), 10)
    assert s.shape == torch.Size([10, 3, 6, 6])
