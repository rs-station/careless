"""
Batched inference: evaluating the scaling model in chunks must not change the
predictions.

`--num-batches` only ever applied to the training loop. The prediction pass ran
the scaling model over the whole dataset in one call, so `ImageLayer`'s
O(n_obs * width**2) weight gather made *inference*, not training, the thing that
set the usable width ceiling once accumulation was in use.
"""
import copy

import numpy as np
import pytest
import torch

from careless.distributions import TruncatedNormal
from careless.models.base import BaseModel
from careless.models.likelihoods import laue as laue_likelihoods
from careless.models.likelihoods import mono as mono_likelihoods
from careless.models.merging.variational import VariationalMergingModel
from careless.models.priors.wilson import WilsonPrior
from careless.models.scaling.image import HybridImageScaler, ImageScaler
from careless.models.scaling.nn import MLPScaler

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
BATCHES = [2, 3, 7, 16]
SEED = 1234


def _make_inputs(raw):
    return tuple(torch.as_tensor(x) for x in raw)


def _build(inputs, laue=False, image_scaler=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    nrefls = int(BaseModel.get_refl_id(inputs).max()) + 1
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1
    prior = WilsonPrior(
        np.random.choice([True, False], nrefls), np.ones(nrefls, dtype='float32')
    )
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(
        prior.mean.detach().numpy(),
        prior.stddev.detach().numpy() / 10.0,
        np.zeros(nrefls, dtype='float32'),
    )
    scaler = MLPScaler(2, 8)
    if image_scaler:
        scaler = HybridImageScaler(scaler, ImageScaler(n_images))
    module = laue_likelihoods if laue else mono_likelihoods
    likelihood = module.StudentTLikelihood(8.0)
    return VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler)


#: Deviation is measured against the scale of the whole array, not per element.
#: These arrays contain entries arbitrarily close to zero -- a scale posterior may
#: predict ~6e-3 where the array's spread is ~1 -- and an elementwise relative
#: tolerance turns the float32 floor at such an entry into a 1e-5 "failure" that
#: says nothing about the code. Measured across eight random initializations, the
#: worst *absolute* deviation between the whole-dataset and chunked paths is
#: 1.5e-7 to 4.8e-7 on arrays whose maximum is 0.6 to 2.2, and it does not grow
#: with num_batches. That is float32 reassociation, so the bound below sits about
#: a factor of three above it.
TOL = 1e-6


def _assert_close(reference, test, label):
    for name, (a, b) in {"mean": (reference[0], test[0]), "stddev": (reference[1], test[1])}.items():
        assert a.shape == b.shape, f"{label}: {name} shape changed, {a.shape} vs {b.shape}"
        denom = max(float(np.abs(a).max()), 1e-30)
        worst = float(np.max(np.abs(a - b)) / denom)
        assert worst < TOL, (
            f"{label}: {name} deviates by {worst:.3e} of the array scale "
            f"(bound {TOL:.0e})"
        )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_batches', BATCHES)
@pytest.mark.parametrize('image_scaler', [False, True])
def test_mono_batched_inference_matches_whole_dataset(mono_inputs, num_batches, device, image_scaler):
    inputs = tuple(t.to(device) for t in _make_inputs(mono_inputs))
    model = _build(inputs, image_scaler=image_scaler).to(device)
    model.eval()

    _assert_close(
        model.scale_mean_stddev(inputs),
        model.scale_mean_stddev(inputs, num_batches),
        f"scale, nb={num_batches}",
    )
    _assert_close(
        model.prediction_mean_stddev(inputs),
        model.prediction_mean_stddev(inputs, num_batches),
        f"prediction, nb={num_batches}",
    )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_batches', BATCHES)
def test_laue_batched_inference_matches_whole_dataset(laue_inputs, num_batches, device):
    """
    The Laue path is the one that could go wrong quietly: the likelihood convolves
    within a harmonic group, and the convolution indexes by a global harmonic_id.
    Chunking only the scaling model and convolving the assembled full-length array
    keeps that intact.
    """
    inputs = tuple(t.to(device) for t in _make_inputs(laue_inputs))
    model = _build(inputs, laue=True).to(device)
    model.eval()

    _assert_close(
        model.scale_mean_stddev(inputs),
        model.scale_mean_stddev(inputs, num_batches),
        f"laue scale, nb={num_batches}",
    )
    _assert_close(
        model.prediction_mean_stddev(inputs),
        model.prediction_mean_stddev(inputs, num_batches),
        f"laue prediction, nb={num_batches}",
    )


@pytest.mark.parametrize('num_batches', BATCHES)
def test_the_chunked_path_actually_runs(mono_inputs, num_batches):
    """
    An equivalence test whose fast path can silently fall back is not a test: if
    scale_moments ignored num_batches the numbers would agree perfectly. Count the
    scaling-model calls and require one per chunk.
    """
    inputs = _make_inputs(mono_inputs)
    model = _build(inputs)
    model.eval()

    expected = len(model._batch_boundaries(inputs, num_batches))
    assert expected > 1, "the fixture is too small to exercise chunking"

    calls = {"n": 0}
    original = model.scaling_model.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.scaling_model.forward = counting

    model.scale_moments(inputs, num_batches)
    assert calls["n"] == expected, (
        f"expected {expected} scaling-model calls for num_batches={num_batches}, "
        f"got {calls['n']}"
    )

    calls["n"] = 0
    model.scale_moments(inputs, 1)
    assert calls["n"] == 1, "num_batches=1 should be a single whole-dataset call"


def test_every_observation_is_written_exactly_once(mono_inputs):
    """A partition, not a sample: no row may be left at its uninitialized value."""
    inputs = _make_inputs(mono_inputs)
    model = _build(inputs)
    model.eval()

    sentinel = float('nan')
    n_obs = int(BaseModel.get_refl_id(inputs).shape[0])
    mean, stddev = model.scale_moments(inputs, 7)
    assert mean.shape == (n_obs,) and stddev.shape == (n_obs,)
    assert torch.isfinite(mean).all(), "some rows were never written"
    assert torch.isfinite(stddev).all(), "some rows were never written"
    assert (stddev > 0).all()


def test_get_predictions_forwards_num_batches(mono_inputs):
    """The DataManager must actually pass the setting down."""
    from careless.io.manager import DataManager

    seen = {}
    model = _build(_make_inputs(mono_inputs))

    def fake_pred(inputs, num_batches=1):
        seen["prediction"] = num_batches
        n = int(BaseModel.get_refl_id(inputs).shape[0])
        return np.zeros(n), np.zeros(n)

    def fake_scale(inputs, num_batches=1):
        seen["scale"] = num_batches
        n = int(BaseModel.get_refl_id(inputs).shape[0])
        return np.zeros(n), np.zeros(n)

    model.prediction_mean_stddev = fake_pred
    model.scale_mean_stddev = fake_scale

    sig = DataManager.get_predictions.__code__.co_varnames[:DataManager.get_predictions.__code__.co_argcount]
    assert "num_batches" in sig, "get_predictions does not accept num_batches"


@pytest.mark.parametrize('device', DEVICES)
def test_the_equivalence_check_can_fail(mono_inputs, device, monkeypatch):
    """
    Negative control. Perturb one chunk by a hair and require _assert_close to
    notice, so a passing equivalence test means something.
    """
    inputs = tuple(t.to(device) for t in _make_inputs(mono_inputs))
    model = _build(inputs).to(device)
    model.eval()

    reference = model.scale_mean_stddev(inputs)

    original = model.scaling_model.forward
    seen = {"n": 0}

    def perturbed(*args, **kwargs):
        dist = original(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] == 2:  # corrupt the second chunk only
            return torch.distributions.Normal(dist.loc * 1.0001, dist.scale)
        return dist

    model.scaling_model.forward = perturbed
    broken = model.scale_mean_stddev(inputs, 4)

    with pytest.raises(AssertionError):
        _assert_close(reference, broken, "deliberately broken")
