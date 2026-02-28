import pytest
import torch
import math
import numpy as np
from torch.distributions import Laplace, Normal, StudentT
from careless.models.priors.empirical import (
    LaplaceReferencePrior,
    NormalReferencePrior,
    StudentTReferencePrior,
    RiceWoolfsonReferencePrior,
)
from careless.distributions import TruncatedNormal


rng = np.random.default_rng(0)
observed = rng.choice([True, False], 100)
observed[0] = True
observed[1] = False
Fobs, SigFobs = rng.random((2, 100)).astype(np.float32)
Fobs[~observed] = 1.0
SigFobs[~observed] = 1.0


def _reference_prior_test(prior, ref_dist, mc_samples):
    """
    Test that:
    1. log_prob is finite everywhere.
    2. log_prob is zero for unobserved indices.
    3. log_prob matches reference distribution for observed indices.
    4. Gradients through a TruncatedNormal surrogate posterior are finite.
    """
    q = TruncatedNormal.from_loc_and_scale(
        Fobs, SigFobs, np.full(len(Fobs), 1e-5, dtype='float32')
    )

    z = q.rsample((mc_samples,))  # (mc_samples, n_refls)
    log_probs = prior.log_prob(z)

    assert torch.all(torch.isfinite(log_probs)), "log_prob contains non-finite values"
    assert torch.all(log_probs[..., ~observed] == 0.0), "unobserved indices should have log_prob 0"

    loss = log_probs.sum()
    loss.backward()
    for p in q.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), "non-finite gradient"

    # Values for observed indices should match the reference distribution
    q.zero_grad()
    z_ref = ref_dist.sample((mc_samples,))
    expected = ref_dist.log_prob(z_ref)[..., observed].detach()
    result   = prior.log_prob(z_ref)[..., observed].detach()
    assert torch.allclose(expected, result, atol=1e-5)


@pytest.mark.parametrize('mc_samples', [3, 1])
def test_LaplaceReferencePrior(mc_samples):
    prior   = LaplaceReferencePrior(Fobs[observed], SigFobs[observed], observed)
    ref     = Laplace(
        torch.as_tensor(Fobs),
        torch.as_tensor(SigFobs) / math.sqrt(2.0),
    )
    _reference_prior_test(prior, ref, mc_samples)


@pytest.mark.parametrize('mc_samples', [3, 1])
def test_NormalReferencePrior(mc_samples):
    prior   = NormalReferencePrior(Fobs[observed], SigFobs[observed], observed)
    ref     = Normal(torch.as_tensor(Fobs), torch.as_tensor(SigFobs))
    _reference_prior_test(prior, ref, mc_samples)


@pytest.mark.parametrize('mc_samples', [3, 1])
def test_StudentTReferencePrior(mc_samples):
    prior   = StudentTReferencePrior(Fobs[observed], SigFobs[observed], 4.0, observed)
    ref     = StudentT(4.0, torch.as_tensor(Fobs), torch.as_tensor(SigFobs))
    _reference_prior_test(prior, ref, mc_samples)
