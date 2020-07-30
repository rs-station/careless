from careless.models.priors.wilson import *
import numpy as np

from careless.utils.tensorflow import disable_gpu
status = disable_gpu()
assert status



def test_Centric():
    E = np.linspace(0.1, 3., 100)
    p = (2./np.pi)**0.5 *np.exp(-0.5*E**2.)
    centric = Centric()
    centric.mean()
    centric.stddev()
    assert np.all(np.isclose(p, centric.prob(E)))
    assert np.all(np.isclose(np.log(p), centric.log_prob(E)))


def test_Acentric():
    acentric = Acentric()
    E = np.linspace(0.1, 3., 100)
    p = 2.*E*np.exp(-E**2.)
    assert np.all(np.isclose(p, acentric.prob(E)))
    assert np.all(np.isclose(np.log(p), acentric.log_prob(E)))

def test_Wilson():
    centric = np.random.randint(0, 2, 100).astype(np.float32)
    epsilon = np.random.randint(1, 6, 100).astype(np.float32)
    prior = WilsonPrior(centric, epsilon)
    F = np.random.random(100).astype(np.float32)
    probs = prior.prob(F)
    log_probs = prior.log_prob(F)
    assert np.all(np.isfinite(probs))
    assert np.all(np.isfinite(log_probs))
