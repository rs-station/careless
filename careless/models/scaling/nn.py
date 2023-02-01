import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
from careless.models.scaling.base import Scaler
from .layers import ConvexCombination,ResNetDense,NormalLayer
import numpy as np

class MLP(tf.keras.models.Sequential):
    def __init__(self, n_layers, width=None, kernel_initializer='glorot_normal', dropout=None, **kwargs):
        layers = []
        if width is not None:
            layers.append(tf.keras.layers.Dense(width, kernel_initializer=kernel_initializer))
        for i in range(n_layers):
            layers.append(ResNetDense(dropout=dropout, kernel_initializer=kernel_initializer))
        super().__init__(layers, **kwargs)

class ImageEncoder(tf.keras.models.Sequential):
    def __init__(self, n_layers, width, kernel_initializer='glorot_normal', dropout=None, **kwargs):
        layers = []
        if width is not None:
            layers.append(tf.keras.layers.Dense(width, kernel_initializer=kernel_initializer, use_bias=False))
        for i in range(n_layers):
            layers.append(ResNetDense(dropout=dropout, kernel_initializer=kernel_initializer))
        layers.append(ConvexCombination(kernel_initializer=kernel_initializer, dropout=dropout))
        super().__init__(layers, **kwargs)

class ScaleModel(Scaler):
    """
    Neural network that learns a latent image representation in order to scale. 
    """
    def __init__(self, n_layers, width, epsilon=1e-7, seed=1234, dropout=0.1, momentum=0.99):
        """
        Parameters
        ----------
        n_layers : int 
            Number of layers
        width : int
            Width of layers
        """
        super().__init__()
        kinit = tf.keras.initializers.VarianceScaling(1. / 10., mode="fan_avg", seed=seed)

        self.image_encoder = ImageEncoder(n_layers, width, kernel_initializer=kinit, dropout=dropout)
        self.dense = tf.keras.layers.Dense(width, kernel_initializer=kinit, use_bias=False)
        self.mlp_scaler = MLP(n_layers, dropout=dropout, kernel_initializer=kinit)

        mlp_layers = []

        #The last layer is linear and generates location/scale params
        tfp_layers = []
        tfp_layers.append(
            tf.keras.layers.Dense(
                2, 
                use_bias=True, 
                kernel_initializer=kinit,
            )
        )

        #The final layer converts the output to a Normal distribution
        #tfp_layers.append(tfp.layers.IndependentNormal())
        tfp_layers.append(NormalLayer(epsilon=epsilon))

        self.scale = self.add_weight(name='global_scale', shape=(), initializer='ones', trainable=False)
        self.distribution = tf.keras.Sequential(tfp_layers)
        self.momentum = momentum

    def call(self, inputs, surrogate_posterior, **kwargs):
        """
        Parameters
        ----------
        metadata : tf.Tensor(float32)
        surrogate_posterior : careless.models.merging.surrogate_posteriors

        Returns
        -------
        dist : tfp.distributions.Distribution
            A tfp distribution instance.
        """
        refl_id  = self.get_refl_id(inputs)
        image_id = self.get_image_id(inputs)
        metadata = self.get_metadata(inputs)
        i = self.get_intensities(inputs)
        sigi = self.get_uncertainties(inputs)

        if kwargs.get('training', False):
            scale = tf.math.reciprocal(tf.math.reduce_std(i))
            if self.scale == 1.:
                self.scale.assign(scale)
            else:
                self.scale.assign(self.momentum * self.scale + (1. - self.momentum) * scale)

        i,sigi = self.scale*i, self.scale*sigi

        f,sigf = surrogate_posterior.mean(),surrogate_posterior.stddev()
        f,sigf = tf.stop_gradient(f),tf.stop_gradient(sigf)
        f = tf.gather(f, tf.squeeze(refl_id, axis=-1), axis=-1)
        sigf = tf.gather(sigf, tf.squeeze(refl_id, axis=-1), axis=-1)

        image_id = tf.squeeze(image_id, axis=-1)
        metadata = tf.RaggedTensor.from_value_rowids(metadata, image_id)
        i = tf.RaggedTensor.from_value_rowids(i, image_id)
        sigi = tf.RaggedTensor.from_value_rowids(sigi, image_id)
        f = tf.RaggedTensor.from_value_rowids(f[...,None], image_id)
        sigf = tf.RaggedTensor.from_value_rowids(sigf[...,None], image_id)

        image = tf.concat((f, sigf, i, sigi, metadata), axis=-1)
        out = self.image_encoder(image)
        out = self.dense(metadata) + out
        out = self.mlp_scaler(out)
        out = self.distribution(out.flat_values)
        return out

class LaueScaleModel(ScaleModel):
    def get_intensities(self, inputs):
        harmonic_id = self.get_harmonic_id(inputs)
        i = super().get_intensities(inputs)
        i = tf.gather(i, tf.squeeze(harmonic_id, axis=-1))
        return i

    def get_uncertainties(self, inputs):
        harmonic_id = self.get_harmonic_id(inputs)
        sigi = super().get_uncertainties(inputs)
        sigi = tf.gather(sigi, tf.squeeze(harmonic_id, axis=-1))
        return sigi


