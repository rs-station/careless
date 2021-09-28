import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from careless.models.base import BaseModel
import numpy as np



class MLPScaler(BaseModel):
    """
    Neural network based scaler with simple dense layers.
    This neural network outputs a normal distribution.
    """
    def __init__(self, n_layers, width, leakiness=0.01):
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
        """
        super().__init__()

        layers = []

        for i in range(n_layers):
            if leakiness is None:
                activation = tf.keras.layers.ReLU()
            else:
                activation = tf.keras.layers.LeakyReLU(0.01)

            layers.append(
                tf.keras.layers.Dense(
                    width, 
                    activation=activation, 
                    use_bias=True, 
                    kernel_initializer='identity'
                    )
                )

        #The last layer is linear and generates location/scale params
        layers.append(
            tf.keras.layers.Dense(
                tfp.layers.IndependentNormal.params_size(), 
                activation='linear', 
                use_bias=True, 
                kernel_initializer='identity'
            )
        )

        #The final layer converts the output to a Normal distribution
        layers.append(tfp.layers.IndependentNormal())

        self.network = tf.keras.Sequential(layers)

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
        return self.network(metadata)


