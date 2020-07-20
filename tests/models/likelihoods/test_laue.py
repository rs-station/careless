from careless.models.likelihoods.laue import *

# This is a bit tricky, but we need to test that all the dimensions for harmonic deconvolutions work out
# We're going to have a certain number of observed reflections `n_refls`. We're going to have a larger number
# of harmonics that need predicting, `n_harmonics`. 


n_refls = 10
n_harmonics = 77
n_samples = 13

iobs, sigiobs = np.random.random((2, n_refls))

#Convenient way to make sure we have at least one of every refl_id
harmonic_index = np.concatenate((np.arange(n_refls), np.random.randint(0, n_refls, n_harmonics - n_refls))) 

predictions = np.random.random(n_harmonics).astype(np.float32)
n_predictions = np.random.random((n_samples, n_harmonics)).astype(np.float32)

def test_NormalLikelihood():
    likelihood = NormalLikelihood(iobs, sigiobs, harmonic_index)

    probs = likelihood.prob(predictions)
    assert probs.shape == n_refls
    log_probs = likelihood.log_prob(predictions)
    assert log_probs.shape == n_refls
    probs = likelihood.prob(n_predictions)
    assert probs.shape == (n_samples, n_refls)
    log_probs = likelihood.log_prob(n_predictions)
    assert log_probs.shape == (n_samples, n_refls)

def test_LaplaceLikelihood():
    likelihood = LaplaceLikelihood(iobs, sigiobs, harmonic_index)

    probs = likelihood.prob(predictions)
    assert probs.shape == n_refls
    log_probs = likelihood.log_prob(predictions)
    assert log_probs.shape == n_refls
    probs = likelihood.prob(n_predictions)
    assert probs.shape == (n_samples, n_refls)
    log_probs = likelihood.log_prob(n_predictions)
    assert log_probs.shape == (n_samples, n_refls)

def test_StudentTLikelihood():
    dof = 4.
    likelihood = StudentTLikelihood(iobs, sigiobs, harmonic_index, dof)

    probs = likelihood.prob(predictions)
    assert probs.shape == n_refls
    log_probs = likelihood.log_prob(predictions)
    assert log_probs.shape == n_refls
    probs = likelihood.prob(n_predictions)
    assert probs.shape == (n_samples, n_refls)
    log_probs = likelihood.log_prob(n_predictions)
    assert log_probs.shape == (n_samples, n_refls)
