import tensorflow as tf
from tensorflow_probability import distributions as tfd
from careless.models.scaling.base import Scaler
import numpy as np


class SequentialScaler(tf.keras.models.Sequential, Scaler):
    """
    Neural network based scaler with simple dense layers.
    """
    def __init__(self, metadata, layers=5, prior=None):
        """
        Parameters
        ----------
        metadata : array 
            m x d array of reflection metadata.
        """
        super().__init__()

        self.prior = prior

        self.metadata = np.array(metadata, dtype=np.float32)
        n,d = metadata.shape

        self.add(tf.keras.Input(shape=d))
        for i in range(layers):
            self.add(tf.keras.layers.Dense(d, activation=tf.keras.layers.LeakyReLU(0.01), use_bias=True, kernel_initializer='identity'))
            #self.add(tf.keras.layers.Dense(d, activation='softplus', use_bias=True))

        #self.add(tf.keras.layers.Dense(2, activation='linear', use_bias=True))
        self.add(tf.keras.layers.Dense(2, activation=tf.keras.layers.LeakyReLU(0.01), use_bias=True, kernel_initializer='identity'))

    @property
    def loc(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        return loc

    @property
    def scale(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        return scale

    def loc_and_scale(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        return loc, scale

    def __call__(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        return tfd.Normal(loc, scale).sample()

    def sample(self, return_kl_term=False, sample_shape=(), seed=None, name='sample', **kwargs):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        dist = tfd.Normal(loc, scale)
        sample = dist.sample(sample_shape, seed, name, **kwargs)
        if return_kl_term:
            eps = 1e-12
            q = dist.prob(sample)
            if self.prior is None:
                p = 1.
            else:
                p = self.prior.prob(sample)
            kl_div = q * (tf.math.log(q + eps) - tf.math.log(p + eps))
            return sample, tf.reduce_sum(kl_div, axis=-1)
        else:
            return sample

