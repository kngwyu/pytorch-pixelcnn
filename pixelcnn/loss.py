import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


def split_output(output: Tensor, n_mix: int) -> Tuple[Tensor, Tensor, Tensor]:
    means = output[:, :, :, :, :n_mix]
    log_scales = torch.clamp(output[:, :, :, :, n_mix: n_mix * 2], min=-7.0)
    coeffs = torch.tanh(output[:, :, :, :, n_mix * 2: n_mix * 3])
    return means, log_scales, coeffs


def log_prob_from_logits_(logits: Tensor) -> Tensor:
    m = logits.max(dim=-1, keepdim=True)[0]
    return logits.sub(m).sub_(torch.exp(logits - m).sum(-1, keepdim=True).log_())


@torch.jit.script
def discretized_mix_logistic_loss(
        target: Tensor,
        output: Tensor,
        color_depth: float = 255.0
) -> Tensor:
    """Log-likelihood for mixture of discretized logistics,
    assumes the data has been rescaled to [-1,1] interval
    """
    n_mix = output.size(1) // 10
    output = output.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)
    logit_probs = output[:, :, :, :n_mix]
    means, log_scales, coeffs = \
        split_output(output[:, :, :, n_mix:].view(target.shape + (-1,)), n_mix)
    target = target.unsqueeze(-1) + torch.zeros_like(means)
    m = means[:, :, :, 0]
    m2 = means[:, :, :, 1] + coeffs[:, :, :, 0] * target[:, :, :, 0]
    m3 = means[:, :, :, 2] + coeffs[:, :, :, 1] * target[:, :, :, 0] \
                           + coeffs[:, :, :, 2] * target[:, :, :, 1]
    means = torch.cat((m.unsqueeze(-2), m2.unsqueeze(-2), m3.unsqueeze(-2)), dim=3)
    centered_t = target - means
    inv_stdv = log_scales.neg().exp()
    plus_in = inv_stdv * (centered_t + 1.0 / color_depth)
    min_in = inv_stdv * (centered_t - 1.0 / color_depth)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = plus_in.sigmoid() - min_in.sigmoid()
    mid_in = inv_stdv * centered_t
    log_pdf_mid = mid_in - log_scales.add(F.softplus(mid_in).mul(2.0))
    log_probs = torch.where(
        target < -0.999,
        log_cdf_plus,
        torch.where(
            target > 0.999,
            log_one_minus_cdf_min,
            torch.where(
                cdf_delta > 1e-5,
                cdf_delta.clamp(min=1e-12).log(),
                log_pdf_mid - torch.tensor(color_depth / 2., device=log_pdf_mid.device).log()
            )
        )
    )
    log_probs = log_probs.sum(dim=3).add_(log_prob_from_logits_(logit_probs))
    return log_probs.logsumexp(dim=-1).sum().neg()
