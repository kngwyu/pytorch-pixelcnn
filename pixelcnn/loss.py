import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


@torch._jit_internal.weak_script
def siga_minus_sigb(a: Tensor, b: Tensor) -> Tensor:
    exp_a = torch.exp(-a, out=a)
    exp_b = torch.exp(-b, out=b)
    return exp_b.sub(exp_a).div_(exp_a.add_(1.0).mul_(exp_b.add(1.0)))


@torch.jit.script
def split_output(output: Tensor, n_mix: int) -> Tuple[Tensor, Tensor, Tensor]:
    means = output[:, :, :, :, :n_mix]
    log_scales = torch.clamp(output[:, :, :, :, n_mix: n_mix * 2], min=-7.0)
    coeffs = torch.tanh(output[:, :, :, :, n_mix * 2: n_mix * 3])
    return means, log_scales, coeffs


@torch.jit.script
def log_prob_from_logits_(logits: Tensor) -> Tensor:
    m = logits.max(-1, keepdim=True)[0]
    return logits.sub_(m).sub_(torch.exp(logits - m).sum(-1, keepdim=True).log())


@torch.jit.script
def _dmixloss_impl(
        target: Tensor,
        centered_t: Tensor,
        log_scales: Tensor,
        color_depth: float
) -> Tensor:
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_t + color_depth)
    min_in = inv_stdv * (centered_t - color_depth)
    log_cdf_plus = plus_in.sub_(F.softplus(plus_in))
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = siga_minus_sigb(plus_in, min_in)
    mid_in = inv_stdv * centered_t
    log_pdf_mid = mid_in.sub_(log_scales).sub_(F.softplus(mid_in).mul_(2.0))
    return torch.where(
        target < -0.999,
        log_cdf_plus,
        torch.where(
            target > 0.999,
            log_one_minus_cdf_min,
            torch.where(
                cdf_delta > 1e-5,
                cdf_delta.clamp(1e-12),
                log_pdf_mid - torch.tensor(color_depth / 2., device=log_pdf_mid.device).log()
            )
        )
    )


def discritized_mix_logistic_loss(
        target: Tensor,
        output: Tensor,
        color_depth: float = 255.
) -> Tensor:
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
    log_probs = _dmixloss_impl(target, centered_t, log_scales, color_depth)
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits_(logit_probs)
    return log_probs.logsumexp(dim=-1).sum()


if __name__ == '__main__':
    t = torch.randn(2, 3, 6, 6)
    o = torch.randn(2, 100, 6, 6)
    discritized_mix_logistic_loss(t, o)
