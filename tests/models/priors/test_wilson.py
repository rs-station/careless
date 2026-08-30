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


def test_wilson_log_prob_handles_zero_samples():
    """WilsonPrior.log_prob must not raise when x=0 is passed for centric reflections.

    Previously, log_prob evaluated p_acentric (Weibull, support x>0) for ALL
    reflections before the torch.where selection, causing a support validation
    error when centric samples at exactly x=0 were passed in.
    """
    n_refls = 200
    centric = np.zeros(n_refls, dtype=bool)
    centric[:100] = True   # first half centric, second half acentric
    epsilon = np.ones(n_refls, dtype=np.float32)
    prior = WilsonPrior(centric, epsilon)

    # Explicitly include x=0 for centric positions (the previously failing case)
    x = torch.ones(n_refls)
    x[:100] = 0.0   # centric reflections at exactly zero

    log_p = prior.log_prob(x)   # must not raise
    assert torch.all(torch.isfinite(log_p[100:]))   # acentric are always finite for x>0


def test_truncated_normal_asymmetric_lower_bounds():
    """
    Surrogate posterior lower bounds must match TF v0.5.4 physics:
      centric  → low = 0.0   (HalfNormal support starts at 0; samples are >= 0)
      acentric → low = 1e-32 (Weibull support is strictly positive; samples are > 0)
    """
    n_refls = 100
    centric_mask = np.zeros(n_refls, dtype=bool)
    centric_mask[:50] = True   # first half centric

    low = np.where(centric_mask, 0.0, 1e-32).astype('float32')
    loc = np.ones(n_refls, dtype='float32')
    scale = np.ones(n_refls, dtype='float32') * 0.1

    q = TruncatedNormal.from_loc_and_scale(loc, scale, low=low)
    for _ in range(20):
        z = q.rsample((5,))
        # All samples must be non-negative (low >= 0 for all)
        assert torch.all(z >= 0), "Sampled structure factors must be non-negative"
        # Acentric samples must be strictly positive
        assert torch.all(z[..., 50:] > 0), \
            "Acentric structure factors must be strictly positive"


def test_build_model_uses_asymmetric_posterior_bounds(
    mono_inputs, mono_reciprocal_asu_collection
):
    """
    build_model() must set low=0 for centric and low=1e-32 for acentric reflections,
    matching TF v0.5.4 surrogate posterior initialization.
    """
    from types import SimpleNamespace
    import torch
    from careless.io.manager import DataManager

    dm = DataManager(mono_inputs, mono_reciprocal_asu_collection)
    parser = SimpleNamespace(
        type='mono', parents=None, dwr=None, wilson_prior_b=None,
        epsilon=1e-7, structure_factor_init_scale=1.0,
        studentt_likelihood_dof=None, refine_uncertainties=False,
        mlp_layers=2, mlp_width=None, scale_bijector='exp',
        image_layers=0, use_image_scales=False, mc_samples=1,
        spectral_file=None,
        kl_weight=None, learning_rate=1e-3, beta_1=0.9, beta_2=0.99,
        clipnorm=None, clipvalue=None, global_clipnorm=None,
        adam_epsilon=1e-7, filter_nan_gradients=True,
        reindexing_ops=None, optimize_double_wilson_r=False,
    )
    model = dm.build_model(parser=parser)
    q = model.surrogate_posterior

    centric = mono_reciprocal_asu_collection.centric  # boolean array

    low = q.low.cpu().numpy()
    assert np.all(low[centric] == 0.0), \
        "Centric reflections must have low=0.0"
    assert np.all(low[~centric] == pytest.approx(1e-32)), \
        "Acentric reflections must have low=1e-32"
