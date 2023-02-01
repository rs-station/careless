import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk


class ResNetDense(tfk.layers.Layer):
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
        self.kernel_initializer = tfk.initializers.get(kernel_initializer)

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
            out = self.dropout(out, **kwargs)

        out = out + X
        return out


class ConvexCombination(tfk.layers.Layer):
    def __init__(self, kernel_initializer='glorot_normal', dropout=None):
        super().__init__()
        self.linear = tfk.layers.Dense(1, kernel_initializer=kernel_initializer, use_bias=False)

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = tfk.layers.Dropout(dropout)

    def call(self, data, encoding=None, **kwargs):
        if encoding is None:
            encoding = data
        score = self.linear(data)

        score = tf.squeeze(score, axis=-1)
        if self.dropout is not None:
            mask = tf.ones_like(score)
            mask = self.dropout(mask, **kwargs)
            score = tf.where(mask >= 1., score, -np.inf)

        score = tf.nn.softmax(score, axis=-1)
        encoded = tf.reduce_sum(score[...,None] * encoding, axis=-2, keepdims=True)
        return encoded


class NormalLayer(tf.keras.layers.Layer):
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



