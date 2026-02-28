import pytest
import torch
import math
import numpy as np
from torch.distributions import Normal, Laplace, StudentT
from careless.models.likelihoods.mono import NormalLikelihood, LaplaceLikelihood, StudentTLikelihood
from careless.models.base import BaseModel


def _make_inputs(inputs):
    return tuple(torch.as_tensor(x) for x in inputs)


def test_mono_NormalLikelihood(mono_inputs):
    inputs = _make_inputs(mono_inputs)
    likelihood = NormalLikelihood()(inputs)
    iobs = BaseModel.get_intensities(inputs).squeeze(-1).float()
    sigiobs = BaseModel.get_uncertainties(inputs).squeeze(-1).float()

    ref = Normal(iobs, sigiobs)
    z = ref.sample()
    assert torch.allclose(likelihood.log_prob(z), ref.log_prob(z))


def test_mono_LaplaceLikelihood(mono_inputs):
    inputs = _make_inputs(mono_inputs)
    likelihood = LaplaceLikelihood()(inputs)
    iobs = BaseModel.get_intensities(inputs).squeeze(-1).float()
    sigiobs = BaseModel.get_uncertainties(inputs).squeeze(-1).float()

    ref = Laplace(iobs, sigiobs / math.sqrt(2.0))
    z = ref.sample()
    assert torch.allclose(likelihood.log_prob(z), ref.log_prob(z))


@pytest.mark.parametrize('dof', [1.0, 2.0, 4.0])
def test_mono_StudentTLikelihood(dof, mono_inputs):
    inputs = _make_inputs(mono_inputs)
    likelihood = StudentTLikelihood(dof)(inputs)
    iobs = BaseModel.get_intensities(inputs).squeeze(-1).float()
    sigiobs = BaseModel.get_uncertainties(inputs).squeeze(-1).float()

    ref = StudentT(dof, iobs, sigiobs)
    z = ref.sample()
    assert torch.allclose(likelihood.log_prob(z), ref.log_prob(z))
