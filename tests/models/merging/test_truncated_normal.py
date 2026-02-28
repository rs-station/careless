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
