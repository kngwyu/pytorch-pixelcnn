import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


@torch.jit.script
def split_output(output: Tensor, n_mix: int) -> Tuple[Tensor, Tensor, Tensor]:
    means = output[:, :, :, :, :n_mix]
    log_scales = torch.clamp(output[:, :, :, :, n_mix: n_mix * 2], min=-7.0)
    coeffs = torch.tanh(output[:, :, :, :, n_mix * 2: n_mix * 3])
    return means, log_scales, coeffs


@torch._jit_internal.weak_script
def siga_minus_sigb(a: Tensor, b: Tensor) -> Tensor:
    exp_a = torch.exp(-a)
    exp_b = torch.exp(-b)
    return exp_b.sub(exp_a).div_(exp_a.add(1.0).mul_(exp_b.add(1.0)))


@torch._jit_internal.weak_script
def log_prob_from_logits_(logits: Tensor) -> Tensor:
    m = logits.max(dim=-1, keepdim=True)[0]
    return logits.sub(m).sub_(torch.exp(logits - m).sum(-1, keepdim=True).log_())


@torch.jit.script
def _dmixloss_impl(
        target: Tensor,
        centered_t: Tensor,
        log_scales: Tensor,
        logit_probs: Tensor,
        color_depth: float
) -> Tensor:
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
        split_output(output[:, :, :, n_mix:].view(*target.shape, -1), n_mix)
    target = target.unsqueeze(-1) + torch.zeros_like(means)
    m2 = means[:, :, :, 1] + coeffs[:, :, :, 0] * target[:, :, :, 0]
    m3 = means[:, :, :, 2] + coeffs[:, :, :, 1] * target[:, :, :, 0] \
                           + coeffs[:, :, :, 2] * target[:, :, :, 1]
    means = torch.cat(tuple(m.unsqueeze(-2) for m in (means[:, :, :, 0], m2, m3)), dim=3)
    centered_t = target - means
    return _dmixloss_impl(target, centered_t, log_scales, logit_probs, color_depth)
