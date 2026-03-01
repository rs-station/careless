import pytest
import torch
import numpy as np

from careless.models.priors.empirical import (
    LaplaceReferencePrior,
    NormalReferencePrior,
    StudentTReferencePrior,
)
from careless.models.priors.wilson import WilsonPrior
from careless.models.merging.variational import VariationalMergingModel
from careless.models.scaling.nn import MLPScaler
from careless.models.scaling.image import HybridImageScaler, ImageScaler
from careless.models.base import BaseModel, reset_losses_and_metrics, get_accumulated_losses
from careless.distributions import TruncatedNormal
from careless.models.likelihoods.mono import NormalLikelihood, LaplaceLikelihood, StudentTLikelihood


def _make_inputs(mono_inputs):
    """Convert numpy arrays to torch tensors."""
    return tuple(torch.as_tensor(x) for x in mono_inputs)


@pytest.mark.parametrize('likelihood_cls', [NormalLikelihood, LaplaceLikelihood, StudentTLikelihood])
@pytest.mark.parametrize('prior_cls', [LaplaceReferencePrior, NormalReferencePrior, StudentTReferencePrior, WilsonPrior])
@pytest.mark.parametrize('scaling_cls', [HybridImageScaler, MLPScaler])
@pytest.mark.parametrize('mc_samples', [3, 1])
def test_mono(likelihood_cls, prior_cls, scaling_cls, mono_inputs, mc_samples):
    inputs = _make_inputs(mono_inputs)
    nrefls   = int(BaseModel.get_refl_id(inputs).max()) + 1
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1

    dof = 4.0
    if likelihood_cls is StudentTLikelihood:
        likelihood = likelihood_cls(dof)
    else:
        likelihood = likelihood_cls()

    if prior_cls is WilsonPrior:
        prior = prior_cls(
            np.random.choice([True, False], nrefls),
            np.ones(nrefls, dtype='float32'),
        )
    elif prior_cls is StudentTReferencePrior:
        prior = prior_cls(
            np.ones(nrefls, dtype='float32'),
            np.ones(nrefls, dtype='float32'),
            dof,
        )
    else:
        prior = prior_cls(
            np.ones(nrefls, dtype='float32'),
            np.ones(nrefls, dtype='float32'),
        )

    loc   = prior.mean.detach().numpy()
    scale = prior.stddev.detach().numpy() / 10.0
    low   = np.zeros(nrefls, dtype='float32')
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, low)

    mlp_scaler = MLPScaler(2, 8)
    if scaling_cls is HybridImageScaler:
        image_scaler = ImageScaler(n_images)
        scaler = HybridImageScaler(mlp_scaler, image_scaler)
    else:
        scaler = mlp_scaler

    merger = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler, mc_samples)

    reset_losses_and_metrics()
    ipred = merger(inputs)
    losses = get_accumulated_losses()
    loss = sum(losses)

    assert torch.all(torch.isfinite(ipred)), "ipred contains non-finite values"
    assert torch.isfinite(loss), "total ELBO loss is non-finite"

    # Verify backward pass works
    loss.backward()
    for p in merger.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"non-finite grad in {p.shape}"


def test_train_model_history_contains_total_loss(mono_inputs):
    """train_model history must include 'Loss' (the total -ELBO), not only its components."""
    inputs = _make_inputs(mono_inputs)
    nrefls   = int(BaseModel.get_refl_id(inputs).max()) + 1
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1

    prior = WilsonPrior(
        np.random.choice([True, False], nrefls),
        np.ones(nrefls, dtype='float32'),
    )
    loc   = prior.mean.detach().numpy()
    scale = prior.stddev.detach().numpy() / 10.0
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, np.zeros(nrefls, 'float32'))
    scaler  = MLPScaler(2, 8)
    merger  = VariationalMergingModel(surrogate_posterior, prior, NormalLikelihood(), scaler)

    history = merger.train_model(inputs, steps=3, progress=False)

    assert "Loss" in history, "train_model history is missing the 'Loss' key"
    assert len(history["Loss"]) == 3
    assert all(np.isfinite(v) for v in history["Loss"]), "Loss history contains non-finite values"
