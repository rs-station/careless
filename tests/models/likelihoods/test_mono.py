import pytest
from careless.models.likelihoods.mono import *
from careless.models.base import BaseModel
from tensorflow_probability import distributions as tfd

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status


def test_mono_NormalLikelihood(mono_inputs):
    likelihood = NormalLikelihood()(mono_inputs)
    iobs = BaseModel.get_intensities(mono_inputs)
    sigiobs = BaseModel.get_uncertainties(mono_inputs)

    l_true = tfd.Normal(iobs, sigiobs)
    z = l_true.sample()

    assert np.allclose(likelihood.log_prob(z), l_true.log_prob(z))

def test_mono_LaplaceLikelihood(mono_inputs):
    likelihood = LaplaceLikelihood()(mono_inputs)
    iobs = BaseModel.get_intensities(mono_inputs)
    sigiobs = BaseModel.get_uncertainties(mono_inputs)

    l_true = tfd.Laplace(iobs, sigiobs)
    z = l_true.sample()

    assert np.allclose(likelihood.log_prob(z), l_true.log_prob(z))

@pytest.mark.parametrize('dof', [1., 2., 4.])
def test_mono_StudentTLikelihood(dof, mono_inputs):
    likelihood = StudentTLikelihood(dof)(mono_inputs)
    iobs = BaseModel.get_intensities(mono_inputs)
    sigiobs = BaseModel.get_uncertainties(mono_inputs)

    l_true = tfd.StudentT(dof, iobs, sigiobs)
    z = l_true.sample()

    assert np.allclose(likelihood.log_prob(z), l_true.log_prob(z))


