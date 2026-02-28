import pytest
import torch
import numpy as np
from careless.models.scaling.image import ImageScaler, HybridImageScaler
from careless.models.scaling.nn import MLPScaler
from careless.models.base import BaseModel


def _make_inputs(inputs):
    return tuple(torch.as_tensor(x) for x in inputs)


def imagescaler_test(inputs):
    inputs = _make_inputs(inputs)
    n_obs = BaseModel.get_metadata(inputs).shape[0]
    image_scaler = ImageScaler(20)  # larger than actual image count
    scales = image_scaler(inputs)

    assert scales.shape[0] == n_obs
    assert torch.all(torch.isfinite(scales))


def hybridscaler_test(mc_samples, inputs):
    inputs = _make_inputs(inputs)
    n_obs = BaseModel.get_metadata(inputs).shape[0]
    mlp_scaler   = MLPScaler(2, 8)
    image_scaler = ImageScaler(20)
    scaler = HybridImageScaler(mlp_scaler, image_scaler)

    q = scaler(inputs)
    z = q.rsample((mc_samples,))
    assert z.shape == (mc_samples, n_obs)
    assert torch.all(torch.isfinite(z))


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_ImageScaler_laue(mc_samples, laue_inputs):
    imagescaler_test(laue_inputs)


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_ImageScaler_mono(mc_samples, mono_inputs):
    imagescaler_test(mono_inputs)


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_HybridImageScaler_laue(mc_samples, laue_inputs):
    hybridscaler_test(mc_samples, laue_inputs)


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_HybridImageScaler_mono(mc_samples, mono_inputs):
    hybridscaler_test(mc_samples, mono_inputs)
