import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


class SequentialScaler(tf.keras.models.Sequential):
    """
    Neural network based scaler with simple dense layers.
    """
    def __init__(self, metadata, layers=5):
        """
        Parameters
        ----------
        metadata : array 
            m x d array of reflection metadata.
        """
        super().__init__()

        self.metadata = np.array(metadata, dtype=np.float32)
        n,d = metadata.shape

        self.add(tf.keras.Input(shape=d))
        for i in range(layers):
            self.add(tf.keras.layers.Dense(d, activation=tf.keras.layers.LeakyReLU()))

        self.add(tf.keras.layers.Dense(2, activation='linear'))

    @property
    def loc(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        return loc

    @property
    def scale(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        return scale

    @property
    def nn_vals(self):
        return super().__call__(self.metadata)

    def __call__(self):
        loc, scale = tf.unstack(super().__call__(self.metadata), axis=1)
        scale = tf.math.softplus(scale)
        return tfd.Normal(loc, scale).sample()

