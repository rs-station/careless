import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from careless.models.merging.surrogate_posteriors import RiceWoolfson
from careless.models.priors.empirical import *
import numpy as np

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status


observed = np.random.choice([True, False], 100)
observed[0] = True  #just in case
observed[1] = False #just in case
Fobs,SigFobs = np.random.random((2, 100)).astype(np.float32)
Fobs[~observed] = 1.
SigFobs[~observed] = 1.

def ReferencePrior_test(p, ref, mc_samples):
    #This part checks indexing and gradient numerics
    q = tfd.TruncatedNormal( #<-- use this dist because RW has positive support
        tf.Variable(Fobs), 
        tfp.util.TransformedVariable( 
            SigFobs,
            tfp.bijectors.Softplus(),
        ),
        low=1e-5,
        high=1e10,
    )
    with tf.GradientTape() as tape:
        z = q.sample(mc_samples)
        log_probs = p.log_prob(z)
    grads = tape.gradient(log_probs, q.trainable_variables)

    assert np.all(np.isfinite(log_probs))
    for grad in grads:
        assert np.all(np.isfinite(grad))
    assert np.all(log_probs.numpy()[...,~observed] == 0.)

    #This tests that the observed values follow the correct distribution
    z = ref.sample(mc_samples)
    expected = ref.log_prob(z).numpy()[...,observed]
    result = p.log_prob(z).numpy()[...,observed]
    assert np.allclose(expected, result, atol=1e-5)

@pytest.mark.parametrize('mc_samples', [(), 3, 1])
def test_LaplaceReferencePrior(mc_samples):
    p = LaplaceReferencePrior(Fobs[observed], SigFobs[observed], observed)
    q = tfd.Laplace(Fobs, SigFobs/np.sqrt(2.))
    ReferencePrior_test(p, q, mc_samples)


@pytest.mark.parametrize('mc_samples', [(), 3, 1])
def test_NormalReferencePrior(mc_samples):
    p = NormalReferencePrior(Fobs[observed], SigFobs[observed], observed)
    q = tfd.Normal(Fobs, SigFobs)
    ReferencePrior_test(p, q, mc_samples)


@pytest.mark.parametrize('mc_samples', [(), 3, 1])
def test_StudentTReferencePrior(mc_samples):
    p = StudentTReferencePrior(Fobs[observed], SigFobs[observed], 4., observed)
    q = tfd.StudentT(4, Fobs, SigFobs)
    ReferencePrior_test(p, q, mc_samples)


@pytest.mark.xfail
@pytest.mark.parametrize('mc_samples', [(), 3, 1])
@pytest.mark.parametrize('centrics', ['all', 'none', 'some'])
def test_RiceWoolfsonReferencePrior(mc_samples, centrics):
    if centrics == 'all':
        centric = np.ones(len(Fobs), dtype=bool)
    elif centrics == 'none':
        centric = np.zeros(len(Fobs), dtype=bool)
    elif centrics == 'some':
        centric = np.random.choice([True, False], len(Fobs))
        centric[observed][0] = True
        centric[observed][1] = False
    else:
        raise ValueError(f"received value, {centrics}, for parameter `centrics` which is not one of 'all', 'none', or 'some'")
    p = RiceWoolfsonReferencePrior(Fobs[observed], SigFobs[observed], centric[observed], observed)
    q = RiceWoolfson(Fobs, SigFobs, centric)
    ReferencePrior_test(p, q, mc_samples)

