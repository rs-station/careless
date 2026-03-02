"""
Custom optimizers for careless.
"""
import torch
from torch.optim import Adam


class AdamEpsInsideSqrt(Adam):
    """
    Adam optimizer that places epsilon *inside* the square root of the second
    moment estimate, matching the TF-Keras convention:

        theta_t = theta_{t-1} - alpha * m_hat / sqrt(v_hat + eps)

    PyTorch's default Adam uses:

        theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)

    The inside-sqrt placement provides a larger effective denominator floor when
    gradient variance is small (sqrt(eps) ≈ 3e-4 for eps=1e-7 vs eps=1e-7),
    making the optimizer more conservative during noisy early training — especially
    important for scale-sensitive likelihoods like Student-T.

    All arguments are identical to ``torch.optim.Adam``.  The ``amsgrad``,
    ``fused``, ``foreach``, ``capturable``, and ``differentiable`` options are
    not supported (raises RuntimeError if non-default values are passed).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7,
                 weight_decay=0, **kwargs):
        for unsupported in ('amsgrad', 'fused', 'foreach', 'capturable', 'differentiable'):
            if kwargs.get(unsupported, False):
                raise RuntimeError(
                    f"AdamEpsInsideSqrt does not support {unsupported}=True"
                )
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step'].item()

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Optional L2 / weight decay (coupled, same as Adam default)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Moment updates
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Bias-corrected second moment (v_hat)
                v_hat = exp_avg_sq / bias_correction2

                # TF-Keras style: eps inside the sqrt
                denom = (v_hat + eps).sqrt_()

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
