from careless.models.likelihoods.mono import *

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status


iobs,sigiobs = np.random.random((2, 100)).astype(np.float32)
samples = np.random.random(100).astype(np.float32)

def test_NormalLikelihood():
    likelihood = NormalLikelihood(iobs, sigiobs)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

def test_LaplaceLikelihood():
    likelihood = LaplaceLikelihood(iobs, sigiobs)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

def test_StudentTLikelihood():
    dof = 4.
    likelihood = StudentTLikelihood(iobs, sigiobs, dof)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

