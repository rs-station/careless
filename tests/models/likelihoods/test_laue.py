import pytest
from careless.models.likelihoods.laue import *
from careless.models.base import BaseModel
from tensorflow_probability import distributions as tfd

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status

def fake_ipred(inputs):
    harmonic_id = BaseModel.get_harmonic_id(inputs)
    intensities = BaseModel.get_intensities(inputs)
    counts = np.bincount(harmonic_id.flatten())[:,None].astype('float32') 
    i  = intensities / counts
    return i[harmonic_id.flatten()]

def test_laue_NormalLikelihood(laue_inputs):
    likelihood = NormalLikelihood()(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.Normal(iobs, sigiobs)

    test = likelihood.log_prob(ipred)
    expected = l_true.log_prob(iobs)
    assert np.allclose(likelihood.log_prob(ipred), l_true.log_prob(iobs))

def test_laue_LaplaceLikelihood(laue_inputs):
    likelihood = LaplaceLikelihood()(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.Laplace(iobs, sigiobs)

    assert np.allclose(likelihood.log_prob(ipred), l_true.log_prob(iobs))

@pytest.mark.parametrize('dof', [1., 2., 4.])
def test_laue_StudentTLikelihood(dof, laue_inputs):
    likelihood = StudentTLikelihood(dof)(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.StudentT(dof, iobs, sigiobs)

    assert np.allclose(likelihood.log_prob(ipred), l_true.log_prob(iobs))


