import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
from careless.models.scaling.base import Scaler
import numpy as np
from careless.utils.distributions import FoldedNormal




class LocationScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_bijector=None, epsilon=1e-7, **kwargs): 
        super().__init__(**kwargs)
        self.epsilon = epsilon
        if scale_bijector is None:
            self.scale_bijector = tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ])
        else:
            self.scale_bijector = scale_bijector

    def call(self, x, **kwargs):
        loc, scale = tf.unstack(x, axis=-1)
        scale = self.scale_bijector(scale)
        return tfd.Normal(loc, scale)

class NormalLayer(LocationScaleLayer):
    def call(self, x, **kwargs):
        loc, scale = tf.unstack(x, axis=-1)
        scale = self.scale_bijector(scale)
        return tfd.Normal(loc, scale)

class FoldedNormalLayer(LocationScaleLayer):
    def call(self, x, **kwargs):
        loc, scale = tf.unstack(x, axis=-1)
        scale = self.scale_bijector(scale)
        return FoldedNormal(loc, scale)

class FeedForward(tfk.layers.Layer):
    """
    This is a ResNet version 2 style layer
    """
    def __init__(self, 
        dropout=None, 
        hidden_units=None, 
        activation='ReLU',
        kernel_initializer='glorot_normal', 
        normalize=False, 
        **kwargs
        ):
        """
        This is a ResNet version 2 style nonlinearity. It implements the following

        ```
        out = dropout(linear(activation(hidden_linear(activation(layer_norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional. 

        Parameters
        ----------
        dropout : float (optional)
            Apply dropout with this rate. Dropout occurs after the second linear layer. By default
            dropout is not used.
        hidden_units : int (optional)
            The size of the hidden layer. By default this will be 2 times the size of the input.
        activation : string or callable (optional)
            Either a string name of a keras activation or a callable function. The default is 'ReLU'.
        kernel_initializer : string or callable (optional)
            Either a string a keras intializer style function. The default is 'glorot_normal'. 
        normalize : bool (optional)
            Optionally apply layer normalization to the input. 
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer

        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None

        if normalize:
            self.normalize = tfk.layers.LayerNormalization(axis=-1)
        else:
            self.normalize = None

        self.activation = tfk.activations.get(activation)

    def build(self, shape, **kwargs):
        self.units = shape[-1]
        if self.hidden_units is None:
            self.hidden_units = 2 * self.units

        self.ff1 = tf.keras.layers.Dense(self.hidden_units, kernel_initializer=self.kernel_initializer, **kwargs)
        self.ff2 = tf.keras.layers.Dense(self.units, kernel_initializer=self.kernel_initializer, **kwargs)

    def call(self, X, **kwargs):
        out = X

        if self.normalize is not None:
            out = self.normalize(out)

        out = self.activation(out)
        out = self.ff1(out)
        out = self.activation(out)
        out = self.ff2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X
        return out

class MLPScaler(Scaler):
    def __init__(self, n_layers, width, epsilon=1e-7):
        """
        Parameters
        ----------
        n_layers : int 
            Number of layers
        width : int
            Width of layers
        """
        super().__init__()

        mlp_layers = []
        self.dmodel = width


        kinit =tfk.initializers.VarianceScaling(
            scale=1./20. / np.sqrt(n_layers), mode='fan_avg', distribution='truncated_normal', seed=1234
        )
            
        for i in range(n_layers):
            mlp_layers.append(FeedForward(hidden_units=width, kernel_initializer=kinit))

        #The last layer is linear and generates location/scale params
        tfp_layers = []
        tfp_layers.append(
            tf.keras.layers.Dense(2, kernel_initializer=kinit)
        )

        tfp_layers.append(FoldedNormalLayer(epsilon=epsilon))
        self.network = tf.keras.Sequential(mlp_layers)
        self.distribution = tf.keras.Sequential(tfp_layers)

        self.sample_input = tf.keras.layers.Dense(
            width-2, use_bias=True, kernel_initializer=kinit)
        self.metadata_input = tf.keras.layers.Dense(
            width-2, use_bias=True, kernel_initializer=kinit)


    @staticmethod
    def _append_one_hot(tensor, index, categories):
        suffix = tf.ones_like(tensor[...,:1]) * tf.one_hot(index, categories)
        out =tf.concat((
            tensor,
            suffix,
        ), axis=-1)
        return out

    def call(self, inputs, samples=32):
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
        image_id = self.get_image_id(inputs)
        metadata = self.get_metadata(inputs)
        intensities = self.get_intensities(inputs)
        uncertainties = self.get_uncertainties(inputs)
        if self.is_laue(inputs):
            harmonic_id = self.get_harmonic_id(inputs)
            intensities = tf.gather(intensities, tf.squeeze(harmonic_id, axis=-1))
            uncertainties = tf.gather(uncertainties, tf.squeeze(harmonic_id, axis=-1))

        y,i,c = tf.unique_with_counts(tf.squeeze(image_id, axis=-1))
        p = samples / tf.cast(tf.gather(c, image_id), 'float32')
        mask = tf.random.uniform(p.shape, 0., 1.) <= p
        mask = tf.squeeze(mask, axis=-1)

        samples = tf.concat((
            metadata[mask],
            intensities[mask],
            uncertainties[mask],
        ), axis=-1)
        samples = self.sample_input(samples)
        samples = self._append_one_hot(samples, 0, 2)
        samples = self.network(samples)
        num_images = tf.reduce_max(image_id) + 1
        _,_,c = tf.unique_with_counts(tf.squeeze(image_id[mask], axis=-1))
        image_rep = tf.scatter_nd(
            image_id[mask], samples, (num_images, self.dmodel)
        )
        image_rep /= tf.cast(c[...,None], 'float32')

        metadata = self.metadata_input(metadata) 
        metadata = self._append_one_hot(metadata, 1, 2)
        metadata = metadata + \
            tf.gather(image_rep, tf.squeeze(image_id, axis=-1))
        out = self.distribution(self.network(metadata))
        return out


