import tensorflow as tf
from careless.models.base import BaseModel
from careless.models.scaling.base import Scaler
import tensorflow_probability as tfp
import numpy as np


class DeltaFunction(object):
    def __init__(self, x, bijector=None):
        self.x = tf.squeeze(x, -1)
        self.bijector = bijector
        if self.bijector is None:
            self.bijector = tfp.bijectors.Exp()

    def sample(self, size):
        return self.x[None,:]

    def stddev(self):
        return tf.zeros_like(self.x)

    def mean(self):
        return self.x

class DeltaFunctionLayer(tf.keras.layers.Layer):
    def call(self, x):
        return DeltaFunction(x)


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

    def call(self, inputs, **kwargs):
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

    def call(self, inputs, **kwargs):
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

    def call(self, metadata_and_image_id, **kwargs):
        data,image_id = metadata_and_image_id
        image_id = tf.squeeze(image_id)
        w = tf.gather(self.w, image_id, axis=0)
        b = tf.gather(self.b, image_id, axis=0)
        result = self.activation(tf.squeeze(tf.matmul(w, data[...,None]), axis=-1) + b)
        return result

class NeuralImageScaler(Scaler):
    def __init__(self, image_layers, max_images, mlp_layers, mlp_width, leakiness=0.01, epsilon=1e-7):
        super().__init__()
        self.image_layers = image_layers

        from careless.models.scaling.nn import MLP
        self.mlp_1 = MLP(mlp_layers, mlp_width, leakiness)
        self.mlp_2 = MLP(mlp_layers, mlp_width, leakiness)
        self.distribution = tf.keras.Sequential([
            tf.keras.layers.Dense(1),
            DeltaFunctionLayer(),
        ])

    def _image_rep(self, imodel, isigi, metadata, image_id, samples=None):
        if samples is None:
            samples = self.image_layers

        n = tf.reduce_max(image_id) + 1
        n = tf.cast(n, 'int32')
        c = tf.math.bincount(tf.squeeze(tf.cast(image_id, 'int32'), axis=-1))
        c = tf.math.bincount(tf.squeeze(tf.cast(image_id, 'int32'), axis=-1))
        p = 1 / tf.cast(c, 'float32')
        p = tf.repeat(p, c)
        l = tf.keras.backend.shape(p)[-1]
        idx = tf.where(samples * p >= tf.random.uniform((l,)))
        idx = tf.squeeze(idx, -1)

        image_id = tf.gather(image_id, idx)
        out = tf.concat((
            imodel,
            isigi,
            metadata,
        ), axis=-1)
        out = tf.gather(out, idx)
        out = self.mlp_1(out)
        d = tf.keras.backend.shape(out)[-1]
        out = tf.scatter_nd(image_id, out, (n, d)) / samples
        return out


    def _call_laue(self, inputs, surrogate_posterior):
        metadata = self.get_metadata(inputs)
        image_id = self.get_image_id(inputs)
        refl_id = self.get_refl_id(inputs)
        harmonic_id = self.get_harmonic_id(inputs)
        isigi = tf.concat((
            self.get_intensities(inputs),
            self.get_uncertainties(inputs),
        ), axis=-1)
        isigi = tf.gather(isigi, tf.squeeze(harmonic_id, axis=-1))
        f = surrogate_posterior.mean()[:,None]
        sigf = surrogate_posterior.stddev()[:,None]
        imodel = tf.concat((
            f,
            sigf,
            tf.square(f) + tf.square(sigf)
        ), axis=-1)
        imodel = tf.gather(imodel, tf.squeeze(refl_id, -1))
        image_rep = self._image_rep(imodel, isigi, metadata, image_id)
        out = self.mlp_2(metadata)
        out = out + tf.gather(image_rep, tf.squeeze(image_id, axis=-1))
        return self.distribution(out)


    def _call_mono(self, inputs, surrogate_posterior):
        metadata = self.get_metadata(inputs)
        image_id = self.get_image_id(inputs)
        refl_id = self.get_refl_id(inputs)
        isigi = tf.concat((
            self.get_intensities(inputs),
            self.get_uncertainties(inputs),
        ), axis=-1)
        f = surrogate_posterior.mean()[:,None]
        sigf = surrogate_posterior.stddev()[:,None]
        imodel = tf.concat((
            f,
            sigf,
            tf.square(f) + tf.square(sigf)
        ), axis=-1)
        imodel = tf.gather(imodel, tf.squeeze(refl_id, -1))
        image_rep = self._image_rep(imodel, isigi, metadata, image_id)
        out = self.mlp_2(metadata)
        out = out + tf.gather(out, tf.squeeze(image_id, axis=-1))
        out = out + tf.gather(image_rep, tf.squeeze(image_id, axis=-1))
        return self.distribution(out)

    def call(self, inputs, surrogate_posterior, **kwargs):
        if self.is_laue(inputs):
            return self._call_laue(inputs, surrogate_posterior)
        else:
            return self._call_mono(inputs, surrogate_posterior)

