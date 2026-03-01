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


@pytest.mark.parametrize("scale_bijector", ["exp", "softplus"])
def test_scale_bijector_choices(mono_inputs, scale_bijector):
    """Both bijector choices must produce a valid Normal with positive scale."""
    inputs = tuple(torch.as_tensor(x) for x in mono_inputs)
    mlp = MLPScaler(2, 8, scale_bijector=scale_bijector)
    q = mlp(inputs)
    assert torch.all(q.scale > 0), f"{scale_bijector} produced non-positive scale"
    assert torch.all(torch.isfinite(q.scale))
    assert torch.all(torch.isfinite(q.mean))


def test_default_scale_bijector_is_exp(mono_inputs):
    """MLPScaler default bijector must be 'exp', not 'softplus'."""
    inputs = tuple(torch.as_tensor(x) for x in mono_inputs)
    mlp_default = MLPScaler(2, 8)
    mlp_exp     = MLPScaler(2, 8, scale_bijector='exp')

    # Force the same weights by copying state after a forward pass initialises LazyLinear
    _ = mlp_default(inputs)
    _ = mlp_exp(inputs)
    mlp_exp.load_state_dict(mlp_default.state_dict())

    q_default = mlp_default(inputs)
    q_exp     = mlp_exp(inputs)
    assert torch.allclose(q_default.scale, q_exp.scale), \
        "Default bijector does not match 'exp'"
