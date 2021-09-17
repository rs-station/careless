from careless.models.scaling.nn import MLPScaler
import pytest
import tensorflow as tf
import numpy as np

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status


def MLPScaler_test(mc_samples, inputs):
    mlp = MLPScaler(2, 3)
    q = mlp(inputs)
    q.mean()
    q.stddev()
    z = q.sample(mc_samples)
    assert z.shape[0] == mc_samples
    q.log_prob(z)


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_MLPScaler_laue(mc_samples, laue_inputs):
    MLPScaler_test(mc_samples, laue_inputs)

@pytest.mark.parametrize("mc_samples", [1, 3])
def test_MLPScaler_mono(mc_samples, mono_inputs):
    MLPScaler_test(mc_samples, mono_inputs)
