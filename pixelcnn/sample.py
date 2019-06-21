import torch
from torch.nn import functional as F
from torch import Tensor


@torch.jit.script
def from_discretized_mix_logistic(logits: Tensor, n_mix: int, out_channel: int = 3) -> Tensor:
    logits = logits.permute(0, 2, 3, 1)
    bhw = logits.shape[:-1]
    logit_probs = logits[:, :, :, :n_mix]
    tmp = logits[:, :, :, n_mix:].view(bhw + (out_channel, -1))
    sample = torch.zeros_like(logit_probs).uniform_(1e-5, 1.0 - 1e-5)
    sample = logit_probs - sample.log_().neg_().log_()
    sample = F.one_hot(sample.max(dim=-1)[1], num_classes=n_mix).float().view(bhw + (1, -1))
    means = tmp[:, :, :, :, :n_mix].mul(sample).sum(dim=-1)
    log_scales = tmp[:, :, :, :, n_mix:n_mix * 2].mul(sample).sum(dim=-1).clamp_(min=-7.0)
    coeffs = tmp[:, :, :, :, n_mix * 2:n_mix * 3].mul(sample).tanh_().sum(dim=-1)
    u = torch.zeros_like(means).uniform_(1e-5, 1.0 - 1e-5)
    x = means + log_scales.exp().mul_(u.log() - (1.0 - u).log_())
    x0 = x[:, :, :, 0].clamp_(-1.0, 1.0)
    x1 = x[:, :, :, 1].add_(coeffs[:, :, :, 0] * x0).clamp_(-1.0, 1.0)
    x2 = x[:, :, :, 2].add_(coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1).clamp_(-1.0, 1.0)
    return torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
