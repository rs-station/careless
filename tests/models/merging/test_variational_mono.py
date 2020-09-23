from careless.models.priors.empirical import LaplaceReferencePrior,NormalReferencePrior,StudentTReferencePrior
from careless.models.priors.wilson import WilsonPrior
from careless.models.merging.variational import VariationalMergingModel
from careless.models.scaling.nn import SequentialScaler
import pytest
import tensorflow as tf
import numpy as np 

from careless.utils.tf import disable_gpu
status = disable_gpu()
assert status




# I'm sleepy. So we will write the monochromatic tests first
n = 1000 #Reflection observations
h = 200 #Unique miller ids
d = 6 #Number of metadata columns
max_mult = 6 #Maximum epsilon multiplicity factor

fobs, sigfobs = np.random.random((2, h)).astype(np.float32)
iobs,sigiobs = np.random.random((2, n)).astype(np.float32)
miller_ids = np.concatenate((np.arange(h), np.random.randint(0, h, n-h))).astype(np.int32)
metadata = np.random.random((n, d)).astype(np.float32)
epsilon = np.random.randint(1, max_mult, h).astype(np.float32)
centric = np.random.randint(1, 2, h).astype(np.float32)


from careless.models.likelihoods.mono import NormalLikelihood,LaplaceLikelihood,StudentTLikelihood
@pytest.mark.parametrize('likelihood_model', [NormalLikelihood, LaplaceLikelihood, StudentTLikelihood])
@pytest.mark.parametrize('prior_model', [LaplaceReferencePrior, NormalReferencePrior, StudentTReferencePrior, WilsonPrior])
@pytest.mark.parametrize('scaling_model', [SequentialScaler])
def test_mono(likelihood_model, prior_model, scaling_model):
    dof = 4. #For the students
    if likelihood_model == StudentTLikelihood:
        likelihood = likelihood_model(iobs, sigiobs, dof)
    else:
        likelihood = likelihood_model(iobs, sigiobs)
    
    if prior_model == WilsonPrior:
        prior = prior_model(centric, epsilon)
    elif prior_model == StudentTReferencePrior:
        prior = prior_model(fobs, sigfobs, dof)
    else:
        prior = prior_model(fobs, sigfobs)

    scaler = scaling_model(metadata)
    merger = VariationalMergingModel(miller_ids, scaler, prior, likelihood, surrogate_posterior=None)
    loss = merger()
    #Test eager
    optimizer = tf.keras.optimizers.Adam(0.1)
    loss = merger._train_step(optimizer, 1)
    #Test graph
    loss = merger.train_step(optimizer, 1)
    merger.sample()
    samples = merger.sample(sample_shape=10)
    assert samples.shape == (10, n) #TODO: uncomment this when safe
    v = merger.surrogate_posterior.trainable_variables[0]

    # Randomly change posterior loc to -100
    # This will cause the probability to underflow
    idx = np.random.randint(0, 2, v.shape).astype(bool)
    idx[0] = True #Make sure there is at least one True
    v.assign(tf.where(
        idx, 
        v, 
        -100.*np.ones(v.shape).astype(np.float32)
    ))
    sample = merger.surrogate_posterior.sample()
    probs  = merger.surrogate_posterior.prob(sample)
    isfinite = tf.reduce_all(tf.math.is_finite(probs))
    assert not isfinite #This is really a test test

    # This should reset the variational distributions to their original values
    merger.rescue_variational_distributions()

    sample = merger.surrogate_posterior.sample()
    probs  = merger.surrogate_posterior.prob(sample)
    isfinite = tf.reduce_all(tf.math.is_finite(probs))
    isfinite = tf.reduce_all(tf.math.is_finite(merger.surrogate_posterior.sample()))
    assert isfinite 

