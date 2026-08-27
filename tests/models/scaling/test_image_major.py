"""
The image-major layout must be invisible from outside the scaling model.

`NeuralImageScaler` is a pure row-wise map, so permuting reflections into a padded
image-major layout and back cannot change anything -- including, critically, the Laue
harmonic convolution, which runs downstream on row-ordered tensors and never sees the
padded layout.
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
from careless.models.likelihoods import laue as laue_likelihoods
from careless.models.likelihoods import mono as mono_likelihoods
from careless.models.merging.variational import VariationalMergingModel
from careless.models.priors.wilson import WilsonPrior
from careless.models.scaling.image import (
    NeuralImageScaler,
    build_image_major_plan,
    choose_slots,
)

SEED = 4321
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _tensors(inputs):
    return tuple(torch.as_tensor(x) for x in inputs)


def _build_scaler(inputs, width=8, image_layers=2, image_major=True, slots=None):
    torch.manual_seed(SEED)
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1
    scaler = NeuralImageScaler(
        image_layers, n_images, 3, width,
        image_major=image_major, slots=slots,
    )
    with torch.no_grad():
        scaler(inputs)          # materialize the lazy layers
    scaler.zero_grad(set_to_none=True)
    return scaler


def _grads(module):
    return {n: p.grad.detach().clone()
            for n, p in module.named_parameters() if p.grad is not None}


# ----------------------------------------------------------------------
# The plan itself
# ----------------------------------------------------------------------

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('slots', [8, 16, 64])
def test_plan_round_trips_every_reflection(laue_inputs, slots, device):
    """inv must select exactly the real rows out of the padded array, in order."""
    inputs = tuple(t.to(device) for t in _tensors(laue_inputs))
    image_id = BaseModel.get_image_id(inputs)
    n_images = int(image_id.max()) + 1
    plan = build_image_major_plan(image_id, n_images, slots=slots)

    n = image_id.reshape(-1).numel()
    marker = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(-1)
    assert torch.equal(marker[plan.pad_index][plan.inv], marker), \
        "gather-in then gather-out is not the identity"

    # Every padded slot belongs to the image it is filed under.
    gid = image_id.reshape(-1)[plan.pad_index]
    rank_of_image = torch.empty_like(plan.img_order)
    rank_of_image[plan.img_order] = torch.arange(n_images, device=device)
    for off, live in zip(plan.offsets, plan.alive):
        block = rank_of_image[gid[off:off + live * plan.slots]].view(live, plan.slots)
        assert torch.equal(block, block[:, :1].expand_as(block)), \
            "a padded slot was filled from a different image"

    assert plan.n_padded >= n
    assert plan.padding_fraction >= 0.0


def test_choose_slots_respects_the_padding_budget():
    counts = torch.tensor([26, 181, 265, 388, 893, 2073], dtype=torch.long)
    for budget in (0.05, 0.15, 0.25, 0.60):
        slots = choose_slots(counts, max_padding=budget)
        padded = int((((counts + slots - 1) // slots) * slots).sum())
        overhead = padded / int(counts.sum()) - 1.0
        assert overhead <= budget + 1e-9, f"slots={slots} overspends the budget"


# ----------------------------------------------------------------------
# Scaler equivalence
# ----------------------------------------------------------------------

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fixture', ['mono_inputs', 'laue_inputs'])
@pytest.mark.parametrize('image_layers', [1, 3])
@pytest.mark.parametrize('width', [8, 16])
def test_scaler_matches_row_major(request, fixture, image_layers, width, device):
    cpu_inputs = _tensors(request.getfixturevalue(fixture))
    inputs = tuple(t.to(device) for t in cpu_inputs)

    reference = _build_scaler(cpu_inputs, width, image_layers, image_major=False).to(device)
    tiled = copy.deepcopy(reference)
    tiled.image_major = True

    q_ref = reference(inputs)
    q_new = tiled(inputs)

    for name in ("loc", "scale"):
        a, b = getattr(q_new, name), getattr(q_ref, name)
        assert a.shape == b.shape
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (
            f"{name} differs; max |Δ| = {float((a - b).abs().max()):.3e}"
        )

    g = torch.randn(q_ref.loc.shape, device=device)
    (q_ref.loc * g).sum().backward()
    (q_new.loc * g).sum().backward()
    ref_g, new_g = _grads(reference), _grads(tiled)
    assert set(ref_g) == set(new_g) and ref_g
    for name, expected in ref_g.items():
        assert torch.allclose(new_g[name], expected, rtol=1e-4, atol=1e-6), (
            f"gradient differs for '{name}'; "
            f"max |Δ| = {float((new_g[name] - expected).abs().max()):.3e}"
        )


def test_no_per_reflection_weight_gather(mono_inputs):
    """The whole point: the weight tensor must never be indexed per reflection."""
    inputs = _tensors(mono_inputs)
    scaler = _build_scaler(inputs, width=8, image_layers=2, image_major=True)
    n_obs = int(BaseModel.get_image_id(inputs).reshape(-1).numel())

    seen = []
    original = torch.Tensor.__getitem__

    def spy(self, item):
        out = original(self, item)
        if out.dim() == 3 and out.shape[0] == n_obs:
            seen.append(tuple(out.shape))
        return out

    torch.Tensor.__getitem__ = spy
    try:
        scaler(inputs)
    finally:
        torch.Tensor.__getitem__ = original

    assert not seen, f"a per-reflection weight tensor was still materialized: {seen}"


# ----------------------------------------------------------------------
# End to end, through the Laue likelihood
# ----------------------------------------------------------------------

@pytest.mark.parametrize('device', DEVICES)
def test_laue_merging_model_is_unchanged(laue_inputs, device):
    """
    Full ELBO on Laue data, harmonic convolution included. If the padded layout leaked
    into refl_id or harmonic_id this is where it would show.
    """
    inputs = tuple(t.to(device) for t in _tensors(laue_inputs))
    nrefls = int(BaseModel.get_refl_id(inputs).max()) + 1

    def build(image_major):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        prior = WilsonPrior(
            np.random.choice([True, False], nrefls),
            np.ones(nrefls, dtype='float32'),
        )
        surrogate = TruncatedNormal.from_loc_and_scale(
            prior.mean.detach().numpy(),
            prior.stddev.detach().numpy() / 10.0,
            np.zeros(nrefls, dtype='float32'),
        )
        scaler = _build_scaler(tuple(t.cpu() for t in inputs),
                               width=8, image_layers=2, image_major=image_major)
        model = VariationalMergingModel(
            surrogate, prior, laue_likelihoods.StudentTLikelihood(8.0), scaler,
            mc_sample_size=1, kl_weight=0.005,
        )
        return model.to(device)

    reference = build(False)
    tiled = build(True)
    tiled.load_state_dict(reference.state_dict())

    losses = []
    grads = []
    for model in (reference, tiled):
        torch.manual_seed(SEED)
        reset_losses_and_metrics()
        model(inputs)
        loss = sum(get_accumulated_losses())
        model.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.detach()))
        grads.append(_grads(model))

    assert np.isclose(losses[0], losses[1], rtol=1e-5), (
        f"Laue ELBO changed: {losses[0]!r} vs {losses[1]!r}"
    )
    assert set(grads[0]) == set(grads[1]) and grads[0]
    for name, expected in grads[0].items():
        assert torch.allclose(grads[1][name], expected, rtol=1e-4, atol=1e-6), (
            f"Laue gradient differs for '{name}'; "
            f"max |Δ| = {float((grads[1][name] - expected).abs().max()):.3e}"
        )
