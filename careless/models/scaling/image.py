import tensorflow as tf
from tensorflow import keras as tfk
from careless.models.base import BaseModel
from careless.models.scaling.base import Scaler
import tensorflow_probability as tfp
import numpy as np


class ImageScaler(Scaler):
    """
    Simple linear image scales. Average value pegged at 1.
    """
    def __init__(self, max_images):
        """
        Parameters
        ----------
        max_images : int
            The maximum number of image variables to be learned
        """
        super().__init__()
        self._scales = tf.Variable(tf.ones(max_images - 1))

    @property
    def scales(self):
        return tf.concat(([1.], self._scales), axis=-1)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list or tf.data.DataSet
            A list of tensor inputs or a DataSet in the standard 
            careless format.

        Returns
        -------
        scales : tf.Tensor(float32)
            A tensor the same shape as image_ids.
        """
        image_ids = self.get_image_id(inputs)
        w = self.scales
        return tf.squeeze(tf.gather(w, image_ids))

class HybridImageScaler(Scaler):
    """
    A scaler that combines an `ImageScaler` with an `MLPScaler`
    """
    def __init__(self, mlp_scaler, image_scaler):
        super().__init__()
        self.mlp_scaler = mlp_scaler
        self.image_scaler = image_scaler

    def call(self, inputs):
        """ Parameters
        ----------
        """
        q = self.mlp_scaler(inputs)
        a = self.image_scaler(inputs)
        return tfp.distributions.TransformedDistribution(
            q,
            tfp.bijectors.Scale(scale=a),
        )


class ImageLayer(Scaler):
    def __init__(self, units, max_images, activation='ReLU', kernel_initializer='identity', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.activation_1 = tf.keras.activations.get(activation)
        self.activation_2 = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.units = units

        self.max_images = max_images

    def build(self, input_shape):
        if self.kernel_initializer == 'identity':
            def kernel_initializer(shape, dtype=tf.float32, **kwargs):
                return tf.eye(shape[1], shape[2], (shape[0],), dtype=dtype)
        else:
            kernel_initializer = self.kernel_initializer

        self.w_1 = self.add_weight(
            name='kernel_1',
            shape=(self.max_images, self.units, input_shape[0][-1]),
            initializer=kernel_initializer,
            trainable=True,
        )
        self.b_1 = self.add_weight(
            name='bias_1', 
            shape=(self.max_images, self.units),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.w_2 = self.add_weight(
            name='kernel_2',
            shape=(self.max_images, input_shape[0][-1], self.units),
            initializer=kernel_initializer,
            trainable=True,
        )
        self.b_2 = self.add_weight(
            name='bias_2', 
            shape=(self.max_images, input_shape[0][-1]),
            initializer=self.bias_initializer,
            trainable=True,
        )

    def call(self, metadata_and_image_id, *args, **kwargs):
        data,image_id = metadata_and_image_id
        image_id = tf.squeeze(image_id)

        out = data

        out = self.activation_1(out)
        w_1 = tf.gather(self.w_1, image_id, axis=0)
        b_1 = tf.gather(self.b_1, image_id, axis=0)
        out = tf.squeeze(tf.matmul(w_1, out[...,None]), axis=-1) + b_1

        out = self.activation_2(out)
        w_2 = tf.gather(self.w_2, image_id, axis=0)
        b_2 = tf.gather(self.b_2, image_id, axis=0)
        out = tf.squeeze(tf.matmul(w_2, out[...,None]), axis=-1) + b_2

        return out + data

class NeuralImageScaler(Scaler):
    def __init__(self, image_layers, max_images, mlp_layers, mlp_width, leakiness=0.01):
        super().__init__()
        layers = []

        def kernel_initializer(shape, dtype=None, **kwargs):
            fan_mean = 0.5*shape[-1] + 0.5*shape[-2]
            scale = tf.sqrt(1. / 5. / mlp_layers / fan_mean)
            tnorm = tfk.initializers.TruncatedNormal(0., scale)
            return tnorm(shape, dtype=dtype, **kwargs)

        for i in range(image_layers):
            layers.append(
                ImageLayer(mlp_width, max_images, kernel_initializer=kernel_initializer)
            )

        self.image_layers = layers
        from careless.models.scaling.nn import MetadataScaler
        self.metadata_scaler = MetadataScaler(mlp_layers, mlp_width, leakiness)

    def call(self, inputs):
        result = self.get_metadata(inputs)
        image_id = self.get_image_id(inputs),

        result = self.metadata_scaler.network(result)
        # One could use this line to add a skip connection here
        #result = result + self.get_metadata(inputs)

        for layer in self.image_layers:
            result = layer((result, image_id))
        result = self.metadata_scaler.distribution(result)
        return result

