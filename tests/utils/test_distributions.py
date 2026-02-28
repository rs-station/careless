"""
Tests for distributions available in careless.utils.distributions,
which re-exports FoldedNormal and Rice from rs_distributions.
"""

import pytest
import torch
import numpy as np
from careless.utils.distributions import FoldedNormal, Rice


def test_FoldedNormal_log_prob():
    """FoldedNormal log_prob should be finite for positive values."""
    loc   = torch.rand(50) + 0.1
    scale = torch.rand(50) + 0.1
    x     = torch.rand(50) + 0.01

    dist = FoldedNormal(loc, scale)
    log_p = dist.log_prob(x)
    assert torch.all(torch.isfinite(log_p))


def test_FoldedNormal_sample():
    """FoldedNormal samples should be non-negative."""
    dist = FoldedNormal(torch.ones(100), torch.ones(100))
    z = dist.sample((10,))
    assert torch.all(z >= 0)


def test_Rice_log_prob():
    """Rice log_prob should be finite for positive values."""
    loc   = torch.rand(50).abs() + 0.01
    scale = torch.rand(50).abs() + 0.1
    x     = torch.rand(50).abs() + 0.01

    dist = Rice(loc, scale)
    log_p = dist.log_prob(x)
    assert torch.all(torch.isfinite(log_p))


def test_Rice_sample():
    """Rice samples should be finite."""
    dist = Rice(torch.ones(50), torch.ones(50))
    z = dist.sample((5,))
    assert torch.all(torch.isfinite(z))
