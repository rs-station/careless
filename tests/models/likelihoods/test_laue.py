from careless.models.likelihoods.laue import *

iobs,sigiobs = np.random.random((2, 10)).astype(np.float32)
samples = np.random.random(100).astype(np.float32)
harmonic_index = np.random.randint(0, 10, 100)

def test_NormalLikelihood():
    likelihood = NormalLikelihood(iobs, sigiobs, harmonic_index)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

def test_LaplaceLikelihood():
    likelihood = LaplaceLikelihood(iobs, sigiobs, harmonic_index)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

def test_StudentTLikelihood():
    dof = 4.
    likelihood = StudentTLikelihood(iobs, sigiobs, harmonic_index, dof)
    likelihood.prob(samples)
    likelihood.log_prob(samples)

