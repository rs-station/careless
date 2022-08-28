import tensorflow as tf
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
        """
        Parameters
        ----------
        """
        q = self.mlp_scaler(inputs)
        a = self.image_scaler(inputs)
        return tfp.distributions.TransformedDistribution(
            q,
            tfp.bijectors.Scale(scale=a),
        )


class ImageLayer(Scaler):
    def __init__(self, units, max_images, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.units = units
        self.max_images = max_images

    def build(self, input_shape):
        def initializer(shape, dtype=tf.float32, **kwargs):
            return tf.eye(shape[1], shape[2], (shape[0],), dtype=dtype)

        self.w = self.add_weight(
            name='kernel',
            shape=(self.max_images, self.units, input_shape[0][-1]),
            initializer=initializer,
            trainable=True,
        )
        self.b = self.add_weight(
            name='bias', 
            shape=(self.max_images, self.units),
            initializer='zeros',
            trainable=True,
        )

    def call(self, metadata_and_image_id, *args, **kwargs):
        data,image_id = metadata_and_image_id
        image_id = tf.squeeze(image_id)
        w = tf.gather(self.w, image_id, axis=0)
        b = tf.gather(self.b, image_id, axis=0)
        result = self.activation(tf.squeeze(tf.matmul(w, data[...,None]), axis=-1) + b)
        return result

class NeuralImageScaler(Scaler):
    def __init__(self, image_layers, max_images, mlp_layers, mlp_width, leakiness=0.01, epsilon=1e-7):
        super().__init__()
        layers = []
        if leakiness is None:
            activation = 'ReLU'
        else:
            activation = tf.keras.layers.LeakyReLU(leakiness)

        for i in range(image_layers):
            layers.append(
                ImageLayer(mlp_width, max_images, activation)
            )

        self.image_layers = layers
        from careless.models.scaling.nn import MetadataScaler
        self.metadata_scaler = MetadataScaler(mlp_layers, mlp_width, leakiness, epsilon=epsilon)

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

