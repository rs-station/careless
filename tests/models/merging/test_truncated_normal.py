import pytest
import torch
import numpy as np
from careless.distributions import TruncatedNormal


def test_truncated_normal_sample_and_log_prob():
    loc   = np.ones(100, dtype='float32')
    scale = np.ones(100, dtype='float32')
    q = TruncatedNormal.from_loc_and_scale(loc, scale)

    z  = q.rsample()
    ll = q.log_prob(z)

    assert torch.all(torch.isfinite(z))
    assert torch.all(torch.isfinite(ll))


def test_truncated_normal_has_two_param_groups():
    loc   = np.ones(100, dtype='float32')
    scale = np.ones(100, dtype='float32')
    q = TruncatedNormal.from_loc_and_scale(loc, scale)

    params = list(q.parameters())
    assert len(params) == 2
    for p in params:
        assert p.shape == torch.Size([100])


def test_truncated_normal_gradients():
    loc   = np.ones(100, dtype='float32')
    scale = np.ones(100, dtype='float32')
    q = TruncatedNormal.from_loc_and_scale(loc, scale)

    z  = q.rsample()
    ll = q.log_prob(z).sum()
    ll.backward()

    for p in q.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_truncated_normal_samples_in_support():
    """rsample must always produce values in [low, high] even for large scale."""
    rng   = np.random.default_rng(0)
    loc   = (rng.random(500) * 5 + 0.1).astype('float32')
    scale = (rng.random(500) * 3 + 0.1).astype('float32')  # large scale stress test
    q = TruncatedNormal.from_loc_and_scale(loc, scale, low=0.0)
    for _ in range(50):
        z = q.rsample((5,))
        assert torch.all(z >= 0.0), \
            f"rsample produced {(z < 0).sum().item()} negative values"


def test_truncated_normal_irg_gradients_finite():
    """IRG backward must produce finite gradients."""
    rng   = np.random.default_rng(1)
    loc   = (rng.random(100) * 3 + 0.5).astype('float32')
    scale = (rng.random(100) * 0.5 + 0.1).astype('float32')
    q = TruncatedNormal.from_loc_and_scale(loc, scale, low=0.0)
    z  = q.rsample((3,))
    z.sum().backward()
    for p in q.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad)), \
            f"Non-finite IRG gradient: {p.grad}"


def test_stable_studentt_log_prob_matches_standard():
    """log_prob should agree with torch.distributions.StudentT for small residuals."""
    from torch.distributions import StudentT as TorchStudentT
    from careless.distributions import StudentT as StableStudentT
    rng = np.random.default_rng(42)
    loc   = torch.tensor(rng.random(200).astype('float32') * 5)
    scale = torch.tensor((rng.random(200).astype('float32') + 0.1))
    value = loc + scale * torch.tensor(rng.standard_normal(200).astype('float32'))
    ref   = TorchStudentT(16., loc, scale).log_prob(value)
    ours  = StableStudentT(16., loc, scale).log_prob(value)
    assert torch.allclose(ref, ours, atol=1e-5), \
        f"Max deviation: {(ref - ours).abs().max()}"


def test_stable_studentt_log_prob_finite_for_large_residuals():
    """log_prob and its gradient must be finite for large (>4 sigma) outliers."""
    loc   = torch.zeros(100)
    scale = torch.ones(100)
    value = torch.full((100,), 1000.0)  # extreme outlier
    from careless.distributions import StudentT
    lp = StudentT(16., loc, scale).log_prob(value)
    assert torch.all(torch.isfinite(lp))


def test_stable_studentt_gradient_finite():
    """Gradient of log_prob w.r.t. value must be finite for all residual magnitudes."""
    from careless.distributions import StudentT
    value = torch.linspace(-100., 100., 500, requires_grad=True)
    lp = StudentT(torch.tensor(16.), torch.zeros(500), torch.ones(500)).log_prob(value)
    lp.sum().backward()
    assert torch.all(torch.isfinite(value.grad))


def test_moment_4():
    """Test truncated normal 4th moment against scipy.stats.truncnorm.moment."""
    from scipy.stats import truncnorm

    npoints = 50
    eps = 1e-3
    rng = np.random.default_rng(42)
    loc_np, scale_np = rng.random((2, npoints)).astype('float32')
    scale_np = scale_np + eps

    q = TruncatedNormal.from_loc_and_scale(loc_np, scale_np)
    mom4 = q.moment_4(method='scipy')

    low, high = 0.0, np.inf
    a = (low - loc_np) / scale_np
    b = (high - loc_np) / scale_np
    mom4_ref = truncnorm.moment(4, a, b, loc_np, scale_np)

    assert np.all(np.isclose(mom4, mom4_ref, rtol=1e-5))
