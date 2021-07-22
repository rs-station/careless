import tensorflow as tf
from tensorflow_probability import distributions as tfd
from careless.models.scaling.base import Scaler
import numpy as np


class SequentialScaler(tf.keras.models.Sequential, Scaler):
    """
    Neural network based scaler with simple dense layers.
    """
    def __init__(self, metadata, layers=20, prior=None, width=None):
        """
        Parameters
        ----------
        metadata : array 
            m x d array of reflection metadata.
        layers : int (optional)
            How many dense layers to add. The default is 20 layers.
        prior : tfd.Distribution (optional)
            An optional prior distirubtion on the scaler output.
        width : int (optional)
            Optionally set the width of the hidden layers to be different than the dimensions of the
            metadata. 
        """
        super().__init__()

        self.prior = prior

        self.metadata = np.array(metadata, dtype=np.float32)
        n,d = metadata.shape
        if width is None:
            width = d

        for i in range(layers):
            self.add(tf.keras.layers.Dense(width, activation=tf.keras.layers.LeakyReLU(0.01), use_bias=True, kernel_initializer='identity'))
            #self.add(tf.keras.layers.Dense(d, activation='softplus', use_bias=True))
        #self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Dense(2, activation='linear', use_bias=True, kernel_initializer='identity')) #TODO: <<<=====Is this better or worse??!?
        #self.add(tf.keras.layers.Dense(2, activation=tf.keras.layers.LeakyReLU(0.01), use_bias=True, kernel_initializer='identity'))
        #This is a hack to initialize the weights for this module
        super().__call__(self.metadata[None,0])

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
            if self.prior is None:
                kl_div =  0.
            else:
                log_q = dist.log_prob(sample)
                log_p = self.prior.log_prob(sample)
                kl_div =  tf.reduce_sum(log_q - log_p, axis=-1)
            return sample, kl_div
        else:
            return sample


