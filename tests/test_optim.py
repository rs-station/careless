"""Tests for careless.optim — custom Adam variant."""
import pytest
import torch
from careless.optim import AdamEpsInsideSqrt


def _make_param_and_opt(lr=1e-3, eps=1e-7, betas=(0.9, 0.999)):
    p = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    opt = AdamEpsInsideSqrt([p], lr=lr, eps=eps, betas=betas)
    return p, opt


def _manual_adam_eps_inside(grad, m, v, step, lr, beta1, beta2, eps):
    """Reference implementation: eps inside sqrt."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    update = lr * m_hat / (v_hat + eps) ** 0.5
    return m, v, update


def test_update_matches_reference():
    """AdamEpsInsideSqrt update must match the manual reference formula."""
    torch.manual_seed(0)
    lr, eps = 1e-2, 1e-7
    betas = (0.9, 0.999)

    p, opt = _make_param_and_opt(lr=lr, eps=eps, betas=betas)
    p_ref = p.detach().clone()

    # Reference state
    m_ref = torch.zeros_like(p_ref)
    v_ref = torch.zeros_like(p_ref)

    for step in range(1, 4):
        grad = torch.randn(3)

        # Reference step
        m_ref, v_ref, upd = _manual_adam_eps_inside(
            grad, m_ref, v_ref, step, lr, *betas, eps
        )
        p_ref = p_ref - upd

        # Optimizer step
        opt.zero_grad()
        p.grad = grad.clone()
        opt.step()

        assert torch.allclose(p.detach(), p_ref, atol=1e-6), \
            f"Step {step}: p={p.detach()} vs ref={p_ref}"


def test_eps_inside_differs_from_standard_adam():
    """The two eps placements must produce different updates for typical inputs."""
    torch.manual_seed(42)
    lr, eps = 1e-2, 1e-4   # large eps amplifies the difference

    p_inside = torch.tensor([1.0], requires_grad=True)
    p_outside = torch.tensor([1.0], requires_grad=True)

    opt_inside  = AdamEpsInsideSqrt([p_inside],  lr=lr, eps=eps)
    opt_outside = torch.optim.Adam( [p_outside], lr=lr, eps=eps)

    grad = torch.tensor([0.1])
    for _ in range(5):
        for opt, p in [(opt_inside, p_inside), (opt_outside, p_outside)]:
            opt.zero_grad()
            p.grad = grad.clone()
            opt.step()

    assert not torch.allclose(p_inside.detach(), p_outside.detach()), \
        "eps-inside and eps-outside Adam must differ with large eps"


def test_unsupported_flags_raise():
    """Passing unsupported flags must raise RuntimeError."""
    p = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        AdamEpsInsideSqrt([p], amsgrad=True)


def test_converges_on_quadratic():
    """Optimizer must reduce loss on a simple quadratic."""
    target = torch.tensor([3.0, -1.0, 2.0])
    p = torch.zeros(3, requires_grad=True)
    opt = AdamEpsInsideSqrt([p], lr=1e-2)

    for _ in range(1000):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()

    assert loss.item() < 1e-3, f"Did not converge: loss={loss.item()}"
