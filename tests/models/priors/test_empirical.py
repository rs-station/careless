from careless.models.priors.empirical import *
import numpy as np

from careless.utils.tensorflow import disable_gpu
status = disable_gpu()
assert status


Fobs,SigFobs = np.random.random((2, 100)).astype(np.float32)


def test_LaplaceReferencePrior():
    p_E = LaplaceReferencePrior(Fobs, SigFobs)
    mean = p_E.mean()
    std = p_E.stddev()
    samples = np.random.normal(Fobs, SigFobs)
    probs = p_E.log_prob(samples)
    log_probs = p_E.log_prob(samples)
    assert np.all(np.isfinite(probs))
    assert np.all(np.isfinite(log_probs))
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert len(mean) == len(Fobs)
    assert len(std) == len(Fobs)

def test_NormalReferencePrior():
    p_E = NormalReferencePrior(Fobs, SigFobs)
    mean = p_E.mean()
    std = p_E.stddev()
    samples = np.random.normal(Fobs, SigFobs)
    probs = p_E.log_prob(samples)
    log_probs = p_E.log_prob(samples)
    assert np.all(np.isfinite(probs))
    assert np.all(np.isfinite(log_probs))
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert len(mean) == len(Fobs)
    assert len(std) == len(Fobs)
