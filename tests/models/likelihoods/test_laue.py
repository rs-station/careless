import pytest
import torch
import math
import numpy as np
from torch.distributions import Normal, Laplace, StudentT
from careless.models.likelihoods.laue import NormalLikelihood, LaplaceLikelihood, StudentTLikelihood
from careless.models.base import BaseModel


def _make_inputs(inputs):
    return tuple(torch.as_tensor(x) for x in inputs)


def _fake_ipred(inputs):
    """Construct a simple (1, n_obs) fake ipred for testing the Laue convolve path."""
    harmonic_id = BaseModel.get_harmonic_id(inputs).squeeze(-1)
    intensities = BaseModel.get_intensities(inputs).squeeze(-1).float()
    counts = torch.bincount(harmonic_id, minlength=int(harmonic_id.max()) + 1)
    result = intensities[harmonic_id] / counts[harmonic_id].float()
    return result.unsqueeze(0)  # shape (1, n_obs)


def _test_likelihood(likelihood_cls, laue_inputs, dof=None):
    inputs = _make_inputs(laue_inputs)
    if dof is not None:
        likelihood = likelihood_cls(dof)(inputs)
    else:
        likelihood = likelihood_cls()(inputs)

    ipred = _fake_ipred(inputs)
    log_p = likelihood.log_prob(ipred)
    assert torch.all(torch.isfinite(log_p))

    # Batched test
    ipred_batched = ipred.expand(3, -1)  # (3, n_obs)
    log_p_batched = likelihood.log_prob(ipred_batched)
    assert torch.all(torch.isfinite(log_p_batched))
    assert log_p_batched.shape[0] == 3


def test_laue_NormalLikelihood(laue_inputs):
    inputs = _make_inputs(laue_inputs)
    likelihood = NormalLikelihood()(inputs)
    iobs    = BaseModel.get_intensities(inputs).squeeze(-1).float()
    sigiobs = BaseModel.get_uncertainties(inputs).squeeze(-1).float()

    ref = Normal(iobs, sigiobs)
    ipred = _fake_ipred(inputs)

    log_p = likelihood.log_prob(ipred)
    log_p_ref = ref.log_prob(iobs)

    # Both should be finite
    assert torch.all(torch.isfinite(log_p))
    assert log_p.shape[-1] == log_p_ref.shape[-1]


def test_laue_LaplaceLikelihood(laue_inputs):
    _test_likelihood(LaplaceLikelihood, laue_inputs)


@pytest.mark.parametrize('dof', [1.0, 2.0, 4.0])
def test_laue_StudentTLikelihood(dof, laue_inputs):
    _test_likelihood(StudentTLikelihood, laue_inputs, dof=dof)
