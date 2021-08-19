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
    def __init__(self, layers, width):
        """
        Parameters
        ----------
        layers : int 
            Number of layers
        width : int
            Width of layers
        """
        super().__init__()

        self.metadata = np.array(metadata, dtype=np.float32)
        layers = []
        for i in range(layers):
            layers.append(
                tf.keras.layers.Dense(
                    width, 
                    activation=tf.keras.layers.LeakyReLU(), 
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

    def call(sefl, inputs):
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


