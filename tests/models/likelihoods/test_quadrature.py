from careless.models.likelihoods.quadrature import mono
from careless.models.likelihoods.quadrature import laue
import numpy as np
import pytest


@pytest.mark.parametrize('likelihood', [mono.NormalLikelihood, mono.LaplaceLikelihood, mono.StudentTLikelihood])
def test_mono(likelihood):
    n = 1000

    I,SigI = np.random.random((2, n)).astype(np.float32)
    SigI += 0.1

    if likelihood == mono.StudentTLikelihood:
        dof = 4.
        likey = likelihood(I, SigI, dof)
    else:
        likey = likelihood(I, SigI)

    ll = likey.expected_log_likelihood(I, SigI).numpy()
    ll = np.squeeze(ll)

    assert np.all(np.isfinite(ll))
    assert len(ll) == n

@pytest.mark.parametrize('likelihood', [laue.NormalLikelihood, laue.LaplaceLikelihood, laue.StudentTLikelihood])
def test_laue(likelihood):
    n = 1000
    n_obs = 500

    Ipred,SigIpred = np.random.random((2, n)).astype(np.float32)
    SigIpred += 0.1

    harmonic_id = np.hstack((np.arange(n_obs) , np.random.randint(0, n_obs, n-n_obs)))
    I,SigI = np.zeros((2, n_obs))
    I[harmonic_id] += Ipred
    SigI[harmonic_id] += SigIpred**2.
    SigI = np.sqrt(SigI)

    if likelihood == laue.StudentTLikelihood:
        dof = 4.
        likey = likelihood(I, SigI, harmonic_id, dof)
    else:
        likey = likelihood(I, SigI, harmonic_id)

    ll = likey.expected_log_likelihood(Ipred, SigIpred).numpy()
    ll = np.squeeze(ll)

    assert np.all(np.isfinite(ll))
    assert len(ll) == n_obs

