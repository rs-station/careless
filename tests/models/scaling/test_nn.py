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


def _apply_identity_init(scaler):
    """Apply TF v0.5.4-style identity init to all Linear layers (mirrors careless.py)."""
    with torch.no_grad():
        for m in scaler.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight)
                k = min(m.weight.shape)
                m.weight.data[:k, :k] = torch.eye(k)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def test_identity_init_weights_and_biases(mono_inputs):
    """After identity init, every Linear weight must have an identity block and zero biases."""
    inputs = tuple(torch.as_tensor(x) for x in mono_inputs)
    mlp = MLPScaler(2, 8)
    _ = mlp(inputs)  # materialize LazyLinear
    _apply_identity_init(mlp)

    for m in mlp.modules():
        if isinstance(m, torch.nn.Linear):
            k = min(m.weight.shape)
            assert torch.allclose(m.weight[:k, :k], torch.eye(k)), \
                f"Top-left {k}×{k} block is not identity for Linear{tuple(m.weight.shape)}"
            assert torch.all(m.weight[:, k:] == 0) or m.weight.shape[1] <= k, \
                f"Columns beyond identity block are non-zero for Linear{tuple(m.weight.shape)}"
            if m.bias is not None:
                assert torch.all(m.bias == 0), \
                    f"Bias not zero for Linear{tuple(m.weight.shape)}"


def test_identity_init_gives_unit_scale_for_zero_metadata(mono_inputs):
    """With zero-mean metadata, identity-initialized MLPScaler must output scale ≈ 1."""
    inputs = list(torch.as_tensor(x) for x in mono_inputs)
    mlp = MLPScaler(2, 8)
    _ = mlp(inputs)  # materialize
    _apply_identity_init(mlp)

    # Replace metadata with zeros (simulates perfectly standardized inputs at mean)
    from careless.models.base import BaseModel
    inputs_zeroed = list(inputs)
    meta_idx = 3  # metadata is index 3 in the input tuple
    inputs_zeroed[meta_idx] = torch.zeros_like(inputs[meta_idx])

    with torch.no_grad():
        q = mlp(inputs_zeroed)

    # scale = exp(0) + epsilon ≈ 1.0 + 1e-7 ≈ 1.0
    expected_scale = 1.0 + mlp.epsilon
    assert torch.allclose(q.scale, torch.full_like(q.scale, expected_scale), atol=1e-5), \
        f"Expected scale ≈ {expected_scale}, got range [{q.scale.min():.4f}, {q.scale.max():.4f}]"
