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

    test = likelihood.log_prob(ipred.T).numpy()
    expected = l_true.log_prob(iobs).numpy()
    #from IPython import embed
    #embed()
    #XX
    assert np.array_equal(expected.shape, test.T.shape)
    assert np.allclose(expected, test.T)

def test_laue_LaplaceLikelihood(laue_inputs):
    likelihood = LaplaceLikelihood()(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.Laplace(iobs, sigiobs/np.sqrt(2.))

    test = likelihood.log_prob(ipred.T).numpy()
    expected = l_true.log_prob(iobs).numpy()
    assert np.array_equal(expected.shape, test.T.shape)
    assert np.allclose(expected, test.T)

@pytest.mark.parametrize('dof', [1., 2., 4.])
def test_laue_StudentTLikelihood(dof, laue_inputs):
    likelihood = StudentTLikelihood(dof)(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.StudentT(dof, iobs, sigiobs)

    test = likelihood.log_prob(ipred.T).numpy()
    expected = l_true.log_prob(iobs).numpy()
    assert np.array_equal(expected.shape, test.T.shape)
    assert np.allclose(expected, test.T)
