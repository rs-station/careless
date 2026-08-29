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


@pytest.mark.parametrize("loc0,scale0", [
    (0.9, 0.5),   # ~2 sigma from the bound: truncation active
    (0.3, 0.5),   # mass piled against the bound
    (0.05, 0.5),  # loc essentially at the bound
    (1.2, 0.3),   # truncation inactive (sanity)
    (0.6, 0.7),
])
def test_truncated_normal_irg_gradient_matches_finite_difference(loc0, scale0):
    """
    The reparameterization gradient from _TruncNormIRG.backward must match the
    pathwise gradient of E[g(z)] w.r.t. loc/scale, checked against a finite
    difference of the exact truncated-normal moments. A previous version dropped
    the moving-lower-bound terms and was wrong (sign flip on d/dscale) whenever
    the truncation was active -- the regime a structure-factor posterior lives in.
    """
    from scipy.stats import truncnorm

    torch.manual_seed(0)
    n_samples = 4_000_000
    low, high = 0.0, 1e10

    q = TruncatedNormal(loc=torch.tensor([loc0]), scale=torch.tensor([scale0]),
                        low=low, high=high)
    z = q.rsample((n_samples,))
    # non-trivial test function so both moments contribute
    (z ** 2 + z).mean().backward()

    # chain the raw-parameter grads through the exp / exp+shift bijectors
    d_loc = q._loc._value.grad.item() / loc0
    d_scale = q._scale._value.grad.item() / (scale0 - 1e-7)

    def expected(mu, sig):
        a, b = (low - mu) / sig, (high - mu) / sig
        m1 = truncnorm.moment(1, a, b, loc=mu, scale=sig)
        m2 = truncnorm.moment(2, a, b, loc=mu, scale=sig)
        return m1 + m2

    h = 1e-4
    fd_loc = (expected(loc0 + h, scale0) - expected(loc0 - h, scale0)) / (2 * h)
    fd_scale = (expected(loc0, scale0 + h) - expected(loc0, scale0 - h)) / (2 * h)

    assert d_loc == pytest.approx(fd_loc, abs=5e-3), \
        f"d/dloc: IRG {d_loc:.4f} vs finite-diff {fd_loc:.4f}"
    assert d_scale == pytest.approx(fd_scale, abs=5e-3), \
        f"d/dscale: IRG {d_scale:.4f} vs finite-diff {fd_scale:.4f}"


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
