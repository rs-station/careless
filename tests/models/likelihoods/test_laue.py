import pytest
from careless.models.likelihoods.laue import *
from careless.models.base import BaseModel
from tensorflow_probability import distributions as tfd

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status

def fake_ipred(inputs):
    harmonic_id = BaseModel.get_harmonic_id(inputs).flatten()
    intensities = BaseModel.get_intensities(inputs).flatten()
    result = intensities[harmonic_id] / np.bincount(harmonic_id)[harmonic_id]
    return result[None,:].astype('float32')

def test_laue_NormalLikelihood(laue_inputs):
    likelihood = NormalLikelihood()(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.Normal(iobs, sigiobs)

    iconv = likelihood.convolve(ipred)

    test = likelihood.log_prob(ipred).numpy()
    expected = l_true.log_prob(iobs).numpy()
    assert np.array_equal(expected.shape, test.T.shape)
    assert np.allclose(expected, test.T)

    #Test batches larger than 1
    ipred = np.concatenate((ipred, ipred, ipred), axis=0)
    likelihood.convolve(ipred).numpy()
    test = likelihood.log_prob(ipred).numpy()
    assert np.array_equiv(expected, test.T)

def test_laue_LaplaceLikelihood(laue_inputs):
    likelihood = LaplaceLikelihood()(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.Laplace(iobs, sigiobs/np.sqrt(2.))

    iconv = likelihood.convolve(ipred)

    test = likelihood.log_prob(ipred).numpy()
    expected = l_true.log_prob(iobs).numpy()

    nobs = BaseModel.get_harmonic_id(laue_inputs).max() + 1

    test = likelihood.log_prob(ipred).numpy()
    expected = l_true.log_prob(iobs).numpy().T

    #The zero padded entries at the end of the input will disagree
    #with the expected values. This is fine, because they will not
    #contribute to the gradient
    test = test[:,:nobs]
    expected = expected[:,:nobs]

    assert np.array_equal(expected.shape, test.shape)
    assert np.allclose(expected, test)

    #Test batches larger than 1
    ipred = np.concatenate((ipred, ipred, ipred), axis=0)
    likelihood.convolve(ipred).numpy()
    test = likelihood.log_prob(ipred).numpy()
    test = test[:,:nobs]
    assert np.array_equiv(expected, test)



@pytest.mark.parametrize('dof', [1., 2., 4.])
def test_laue_StudentTLikelihood(dof, laue_inputs):
    likelihood = StudentTLikelihood(dof)(laue_inputs)
    iobs = BaseModel.get_intensities(laue_inputs)
    sigiobs = BaseModel.get_uncertainties(laue_inputs)
    ipred = fake_ipred(laue_inputs)

    l_true = tfd.StudentT(dof, iobs, sigiobs)

    iconv = likelihood.convolve(ipred)

    test = likelihood.log_prob(ipred).numpy()
    expected = l_true.log_prob(iobs).numpy()

    nobs = BaseModel.get_harmonic_id(laue_inputs).max() + 1

    test = likelihood.log_prob(ipred).numpy()
    expected = l_true.log_prob(iobs).numpy().T

    #The zero padded entries at the end of the input will disagree
    #with the expected values. This is fine, because they will not
    #contribute to the gradient
    test = test[:,:nobs]
    expected = expected[:,:nobs]

    assert np.array_equal(expected.shape, test.shape)
    assert np.allclose(expected, test)

    #Test batches larger than 1
    ipred = np.concatenate((ipred, ipred, ipred), axis=0)
    likelihood.convolve(ipred).numpy()
    test = likelihood.log_prob(ipred).numpy()
    test = test[:,:nobs]
    assert np.array_equiv(expected, test)


