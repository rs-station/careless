import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
from careless.models.scaling.base import Scaler
import numpy as np


class NormalLayer(tfk.layers.Layer):
    def __init__(self, scale_bijector=None, epsilon=1e-7, **kwargs): 
        super().__init__(**kwargs)
        self.epsilon = epsilon
        if scale_bijector is None:
            self.scale_bijector = tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Softplus(),
            ])
        else:
            self.scale_bijector = scale_bijector

    def call(self, x, **kwargs):
        loc, scale = tf.unstack(x, axis=-1)
        scale = self.scale_bijector(scale)
        return tfd.Normal(loc, scale)

class MetadataScaler(Scaler):
    """
    Neural network based scaler with simple dense layers.
    This neural network outputs a normal distribution.
    """
    def __init__(self, n_layers, width, leakiness=0.01, epsilon=1e-7, scale_bijector=None, scale_multiplier=None):
        """
        Parameters
        ----------
        n_layers : int 
            Number of layers
        width : int
            Width of layers
        leakiness : float or None
            If float, use LeakyReLU activation with provided parameter. Otherwise 
            use a simple ReLU
        epsilon : float
            A small constant for numerical stability. This is passed to the distribution layer.
        scale_bijector : tfp.bijectors.Bijector
            Optional scale bijector for the ouptut distibution
        scale_multiplier : float
            Optional constant to multiply the output location and scale by. This can increase
            numerical stability. 
        """
        super().__init__()

        mlp_layers = []

        for i in range(n_layers):
            if leakiness is None:
                activation = tfk.layers.ReLU()
            else:
                activation = tfk.layers.LeakyReLU(leakiness)

            mlp_layers.append(
                tfk.layers.Dense(
                    width, 
                    activation=activation, 
                    use_bias=True, 
                    kernel_initializer='identity',
                    )
                )

        #The last layer is linear and generates location/scale params
        tfp_layers = []
        tfp_layers.append(
            tfk.layers.Dense(
                2, 
                activation='linear', 
                use_bias=True, 
                kernel_initializer='identity',
            )
        )

        #The final layer converts the output to a Normal distribution
        #tfp_layers.append(tfp.layers.IndependentNormal())
        tfp_layers.append(NormalLayer(epsilon=epsilon, scale_bijector=scale_bijector))
        if scale_multiplier is not None:
            tfp_layers.append(
                tfk.layers.Lambda(lambda x: tfb.Shift(scale_multiplier)(x))
            )

        self.network = tfk.Sequential(mlp_layers)
        self.distribution = tfk.Sequential(tfp_layers)

    def call(self, metadata):
        """
        Parameters
        ----------
        metadata : tf.Tensor(float32)

        Returns
        -------
        dist : tfp.distributions.Distribution
            A tfp distribution instance.
        """
        return self.distribution(self.network(metadata))


class MLPScaler(MetadataScaler):
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.Tensor(float32)
            An arbitrarily batched input tensor

        Returns
        -------
        dist : tfp.distributions.Distribution
            A tfp distribution instance.
        """
        metadata = self.get_metadata(inputs)
        return super().call(metadata)

