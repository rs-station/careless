import pytest
import torch
import numpy as np
from careless.models.priors.wilson import WilsonPrior, Centric, Acentric
from careless.distributions import TruncatedNormal


def test_Centric():
    E = torch.linspace(0.1, 3.0, 100)
    p_expected = (2.0 / np.pi) ** 0.5 * torch.exp(-0.5 * E ** 2.0)
    centric = Centric(torch.ones(1))

    prob = torch.exp(centric.log_prob(E))
    assert torch.all(torch.isclose(p_expected, prob, atol=1e-5))


def test_Acentric():
    E = torch.linspace(0.1, 3.0, 100)
    p_expected = 2.0 * E * torch.exp(-(E ** 2.0))
    acentric = Acentric(torch.ones(1))

    prob = torch.exp(acentric.log_prob(E))
    assert torch.all(torch.isclose(p_expected, prob, atol=1e-5))


@pytest.mark.parametrize('mc_samples', [1, 3])
def test_Wilson(mc_samples):
    n_refls = 100
    centric = np.random.randint(0, 2, n_refls).astype(bool)
    epsilon = np.random.randint(1, 6, n_refls).astype(np.float32)
    prior = WilsonPrior(centric, epsilon)

    F = torch.from_numpy(np.random.random(n_refls).astype('float32'))
    log_probs = prior.log_prob(F)
    assert torch.all(torch.isfinite(log_probs))

    # Check gradient flow through the prior
    loc   = prior.mean.detach().numpy()
    scale = prior.stddev.detach().numpy()
    q = TruncatedNormal.from_loc_and_scale(loc, scale, np.zeros(n_refls, dtype='float32'))

    z = q.rsample((mc_samples,))
    log_p = prior.log_prob(z)
    assert torch.all(torch.isfinite(log_p))

    loss = log_p.sum()
    loss.backward()
    for p in q.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))
