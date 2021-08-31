import pytest
import tensorflow as tf
import numpy as np
from careless.models.merging.surrogate_posteriors import TruncatedNormal

def test_truncated_normal():
    loc   = np.ones(100, dtype='float32')
    scale = np.ones(100, dtype='float32')
    q = TruncatedNormal.from_location_and_scale(loc, scale)
    z = q.sample()
    ll = q.log_prob(z)
    assert np.all(np.isfinite( z))
    assert np.all(np.isfinite(ll))

    assert len(q.trainable_variables) == 2
    assert q.trainable_variables[0].shape == 100
    assert q.trainable_variables[1].shape == 100

    with tf.GradientTape() as tape:
        z = q.sample()
        ll = tf.reduce_sum(q.log_prob(z))
    grads = tape.gradient(ll, q.trainable_variables)
    assert grads[0] is not None
    assert grads[1] is not None

    assert np.all(np.isfinite(grads[0]))
    assert np.all(np.isfinite(grads[1]))
