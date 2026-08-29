"""
Gradient accumulation must not change the update.

These tests pin the two properties that make `train_model(num_batches=N)` a drop-in
replacement for the whole-dataset step:

  1. the structure factors are sampled exactly once per step, and
  2. the accumulated gradient equals the monolithic gradient, element for element.
"""
import copy

import numpy as np
import pytest
import torch

from careless.distributions import TruncatedNormal
from careless.models.base import (
    BaseModel,
    reset_losses_and_metrics,
    get_accumulated_losses,
)
from careless.models.likelihoods import mono as mono_likelihoods
from careless.models.likelihoods import laue as laue_likelihoods
from careless.models.merging.variational import VariationalMergingModel
from careless.models.priors.wilson import WilsonPrior
from careless.models.scaling.image import HybridImageScaler, ImageScaler
from careless.models.scaling.nn import MLPScaler


SEED = 1234

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _make_inputs(inputs):
    return tuple(torch.as_tensor(x) for x in inputs)


def _build_model(inputs, mc_samples=1, kl_weight=None, laue=False):
    """A small but representative merger: MLP + per-image scales, Wilson prior."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    nrefls = int(BaseModel.get_refl_id(inputs).max()) + 1
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1

    prior = WilsonPrior(
        np.random.choice([True, False], nrefls),
        np.ones(nrefls, dtype='float32'),
    )
    loc = prior.mean.detach().numpy()
    scale = prior.stddev.detach().numpy() / 10.0
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(
        loc, scale, np.zeros(nrefls, dtype='float32')
    )

    scaler = HybridImageScaler(MLPScaler(2, 8), ImageScaler(n_images))
    likelihood_module = laue_likelihoods if laue else mono_likelihoods
    likelihood = likelihood_module.StudentTLikelihood(8.0)

    model = VariationalMergingModel(
        surrogate_posterior,
        prior,
        likelihood,
        scaler,
        mc_sample_size=mc_samples,
        kl_weight=kl_weight,
    )

    # Materialize the LazyLinear layers so deepcopies share an identical shape.
    with torch.no_grad():
        reset_losses_and_metrics()
        model(inputs)
    model.zero_grad(set_to_none=True)
    return model


def _monolithic_grads(model, inputs, n_obs):
    """
    Whole-dataset forward/backward with no detaching and no batching: the reference
    the accumulated step has to reproduce.
    """
    device = next(model.parameters()).device
    torch.manual_seed(SEED)
    noise = torch.empty((model.mc_sample_size, n_obs), device=device).normal_()
    z_f = model.sample_structure_factors()

    reset_losses_and_metrics()
    model.add_structure_factor_kl(z_f)
    model(inputs, z_f=z_f, scale_noise=noise)
    loss = sum(get_accumulated_losses())

    model.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.detach()), _grad_dict(model)


def _accumulated_grads(model, inputs, num_batches):
    """One `train_model` step, with the gradients read back before they are consumed."""
    torch.manual_seed(SEED)
    captured = {}

    original_step = model.configure_optimizers

    def configure_optimizers():
        optimizer = original_step()
        step = optimizer.step

        def capture(*args, **kwargs):
            captured.update(_grad_dict(model))
            return step(*args, **kwargs)

        optimizer.step = capture
        return optimizer

    model.configure_optimizers = configure_optimizers
    history = model.train_model(
        inputs, steps=1, num_batches=num_batches, progress=False
    )
    model.configure_optimizers = original_step
    return history["Loss"][0], captured


def _grad_dict(model):
    return {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def _assert_grads_match(reference, test, label):
    assert set(reference) == set(test), f"{label}: different parameters received gradients"
    assert reference, f"{label}: no gradients were produced"
    worst = 0.0
    for name, ref in reference.items():
        got = test[name]
        denom = ref.abs().max().clamp(min=1e-30)
        worst = max(worst, float((got - ref).abs().max() / denom))
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-7), (
            f"{label}: gradient mismatch for '{name}'; "
            f"max abs diff {float((got - ref).abs().max()):.3e}"
        )
    return worst


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_batches', [1, 2, 3, 7])
@pytest.mark.parametrize('kl_weight', [None, 0.005])
@pytest.mark.parametrize('mc_samples', [1, 3])
def test_mono_accumulated_gradients_match_monolithic(
    mono_inputs, num_batches, kl_weight, mc_samples, device
):
    inputs = _make_inputs(mono_inputs)
    n_obs = int(BaseModel.get_refl_id(inputs).shape[0])

    model = _build_model(inputs, mc_samples=mc_samples, kl_weight=kl_weight).to(device)
    inputs = tuple(t.to(device) for t in inputs)
    reference_model = copy.deepcopy(model)

    ref_loss, ref_grads = _monolithic_grads(reference_model, inputs, n_obs)
    loss, grads = _accumulated_grads(model, inputs, num_batches)

    _assert_grads_match(ref_grads, grads, f"num_batches={num_batches}")
    assert np.isclose(loss, ref_loss, rtol=1e-5), (
        f"loss differs: monolithic {ref_loss!r} vs accumulated {loss!r}"
    )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_batches', [1, 3, 4, 16])
def test_laue_accumulated_gradients_match_monolithic(laue_inputs, num_batches, device):
    inputs = _make_inputs(laue_inputs)
    n_obs = int(BaseModel.get_refl_id(inputs).shape[0])

    model = _build_model(inputs, laue=True).to(device)
    inputs = tuple(t.to(device) for t in inputs)
    reference_model = copy.deepcopy(model)

    ref_loss, ref_grads = _monolithic_grads(reference_model, inputs, n_obs)
    loss, grads = _accumulated_grads(model, inputs, num_batches)

    _assert_grads_match(ref_grads, grads, f"laue num_batches={num_batches}")
    assert np.isclose(loss, ref_loss, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_batches', [2, 4, 16])
def test_laue_batches_do_not_split_harmonics(laue_inputs, num_batches, device):
    """A harmonic group spans several rows; convolution requires them in one batch."""
    inputs = tuple(t.to(device) for t in _make_inputs(laue_inputs))
    harmonic_id = BaseModel.get_harmonic_id(inputs).squeeze(-1).long()

    boundaries = VariationalMergingModel._batch_boundaries(inputs, num_batches)
    assert len(boundaries) > 1, "test data is too small to exercise batching"
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == len(harmonic_id)

    owner = {}
    for batch, (lo, hi) in enumerate(boundaries):
        assert hi > lo
        for h in harmonic_id[lo:hi].unique().tolist():
            assert owner.setdefault(h, batch) == batch, (
                f"harmonic group {h} is split across batches"
            )


@pytest.mark.parametrize('num_batches', [1, 5])
def test_structure_factors_are_sampled_once_per_step(mono_inputs, num_batches):
    inputs = _make_inputs(mono_inputs)
    model = _build_model(inputs)

    calls = []
    original = model.surrogate_posterior.rsample
    model.surrogate_posterior.rsample = lambda *a, **kw: (
        calls.append(1) or original(*a, **kw)
    )

    steps = 3
    model.train_model(inputs, steps=steps, num_batches=num_batches, progress=False)
    model.surrogate_posterior.rsample = original

    assert len(calls) == steps, (
        f"expected {steps} q(F) samples with num_batches={num_batches}, got {len(calls)}"
    )


def test_metrics_and_history_survive_batching(mono_inputs):
    inputs = _make_inputs(mono_inputs)
    n_obs = int(BaseModel.get_refl_id(inputs).shape[0])
    split = n_obs // 3

    model = _build_model(inputs)
    reference_model = copy.deepcopy(model)

    torch.manual_seed(SEED)
    single = reference_model.train_model(
        inputs, steps=3, num_batches=1, progress=False,
        validation_data=tuple(t[:split] for t in inputs), validation_frequency=1,
    )
    torch.manual_seed(SEED)
    batched = model.train_model(
        inputs, steps=3, num_batches=4, progress=False,
        validation_data=tuple(t[:split] for t in inputs), validation_frequency=1,
    )

    assert set(single) == set(batched)
    for key in single:
        np.testing.assert_allclose(
            batched[key], single[key], rtol=1e-4, atol=1e-6,
            err_msg=f"history key '{key}' diverged under batching",
        )


@pytest.mark.parametrize('num_batches', [1, 4])
def test_divergence_never_reaches_the_optimizer(mono_inputs, num_batches):
    """
    A non-finite step must be abandoned with the parameters untouched.

    The divergence check moved out of the batch loop -- it used to run per batch,
    which cost a host stall per batch -- and now runs once per step, from the same
    transfer that reads the metrics. It still sits before optimizer.step(), so the
    guarantee is unchanged: a batch that goes non-finite runs its backward and
    dirties .grad, but no optimizer step is taken and zero_grad() precedes the next
    one. This pins that, including that the surviving history is exactly the steps
    that completed.
    """
    inputs = _make_inputs(mono_inputs)
    diverge_at = 2

    clean = _build_model(inputs)
    poisoned = copy.deepcopy(clean)

    torch.manual_seed(SEED)
    clean_history = clean.train_model(
        inputs, steps=diverge_at, num_batches=num_batches, progress=False
    )
    reference = {n: p.detach().clone() for n, p in clean.named_parameters()}

    steps_seen = {"n": 0}
    original_sample = poisoned.sample_structure_factors
    original_forward = poisoned.forward

    def counting_sample(*args, **kwargs):
        steps_seen["n"] += 1
        return original_sample(*args, **kwargs)

    def poisoned_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if steps_seen["n"] > diverge_at:
            poisoned.add_loss(torch.tensor(float('nan')))
        return out

    poisoned.sample_structure_factors = counting_sample
    poisoned.forward = poisoned_forward

    torch.manual_seed(SEED)
    history = poisoned.train_model(
        inputs, steps=diverge_at + 5, num_batches=num_batches, progress=False
    )

    assert len(history["Loss"]) == diverge_at, (
        f"expected the run to stop after {diverge_at} good steps, "
        f"got {len(history['Loss'])}"
    )
    np.testing.assert_allclose(history["Loss"], clean_history["Loss"], rtol=1e-6)

    for name, p in poisoned.named_parameters():
        assert torch.equal(p.detach(), reference[name]), (
            f"'{name}' moved on the diverging step"
        )
