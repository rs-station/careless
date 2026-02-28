import pytest
import torch
import numpy as np
from careless.models.scaling.nn import MLPScaler


def _mlpscaler_test(mc_samples, inputs):
    inputs = tuple(torch.as_tensor(x) for x in inputs)
    mlp = MLPScaler(2, 8)
    q = mlp(inputs)

    mean = q.mean
    stddev = q.stddev
    z = q.rsample((mc_samples,))
    log_p = q.log_prob(z)

    assert z.shape[0] == mc_samples
    assert torch.all(torch.isfinite(z))
    assert torch.all(torch.isfinite(log_p))


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_MLPScaler_laue(mc_samples, laue_inputs):
    _mlpscaler_test(mc_samples, laue_inputs)


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_MLPScaler_mono(mc_samples, mono_inputs):
    _mlpscaler_test(mc_samples, mono_inputs)
