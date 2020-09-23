from careless.models.scaling.nn import *
import pytest
import tensorflow as tf
import numpy as np

from careless.utils.tf import disable_gpu
status = disable_gpu()
assert status


@pytest.mark.parametrize("nsamples", [(), 10])
@pytest.mark.parametrize("return_kl_term", [False, True])
def test_SequentialScaler(nsamples, return_kl_term):
    X = np.random.random((100, 5))
    n = SequentialScaler(X)
    with tf.GradientTape() as tape:
        if return_kl_term:
            sample, kl_loss = n.sample(True, nsamples)
        else:
            sample = n.sample(False, nsamples)
            kl_loss = 0.
        loss = tf.reduce_sum(sample) + kl_loss
    tf.convert_to_tensor(sample)
    tf.convert_to_tensor(kl_loss)
    n.trainable_variables

    #Test Gradients
    grads = tape.gradient(loss, n.trainable_variables)
    grads = tf.nest.flatten(grads)
    assert tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads])
    assert None not in grads

