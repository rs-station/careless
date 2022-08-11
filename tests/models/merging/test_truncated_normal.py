import pytest
import tensorflow as tf
import numpy as np
from careless.models.merging.surrogate_posteriors import TruncatedNormal

def test_truncated_normal():
    loc   = np.ones(100, dtype='float32')
    scale = np.ones(100, dtype='float32')
    q = TruncatedNormal.from_loc_and_scale(loc, scale)
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

@pytest.mark.parametrize("method", ["scipy", "tf"])
def test_moment_4(method, npoints=100, eps=1e-3, rtol=1e-5):
    """ Test truncated normal 4th moment against scipy.stats.truncnorm.moment """
    loc,scale = np.random.random((2, npoints)).astype('float32')
    scale = scale + eps

    q = TruncatedNormal.from_loc_and_scale(loc, scale)
    mom4 = q.moment_4(method=method)

    from scipy.stats import truncnorm
    low,high = 0., np.inf
    a, b = (low - loc) / scale, (high - loc) / scale
    mom4s = truncnorm.moment(4, a, b, loc, scale)
    assert np.all(np.isclose(mom4, mom4s, rtol=rtol))

