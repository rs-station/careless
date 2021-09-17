from careless.models.likelihoods.base import Likelihood
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

class LocationScaleLikelihood(Likelihood):
    def get_loc_and_scale(self, inputs):
        loc   = self.get_intensities(inputs)
        scale = self.get_uncertainties(inputs)
        return tf.squeeze(loc), tf.squeeze(scale)

class NormalLikelihood(LocationScaleLikelihood):
    def call(self, inputs):
        return tfd.Normal(*self.get_loc_and_scale(inputs))

class LaplaceLikelihood(LocationScaleLikelihood):
    def call(self, inputs):
        loc, scale = self.get_loc_and_scale(inputs)
        return tfd.Laplace(loc, scale/np.sqrt(2.))

class StudentTLikelihood(LocationScaleLikelihood):
    def __init__(self, dof):
        """
        Parameters
        ----------
        dof : float
            Degrees of freedom of the student t likelihood.
        """
        super().__init__()
        self.dof = dof

    def call(self, inputs):
        return tfd.StudentT(self.dof, *self.get_loc_and_scale(inputs))

class NeuralLikelihood(Likelihood):
    def __init__(self, mlp_layers, mlp_width):
        super().__init__()

        layers = []
        for _ in range(mlp_layers):
            layer = tf.keras.layers.Dense(
                mlp_width,
                activation=tf.keras.layers.LeakyReLU(),
            )
            layers.append(layer)

        layer = tf.keras.layers.Dense(
            1,
            activation='softplus',
        )
        layers.append(layer)
        self.network = tf.keras.models.Sequential(layers)

    def base_dist(self, loc, scale):
        raise NotImplementedError("extensions of this class must implement a base_dist(loc, scale) method")

    def call(self, inputs):
        iobs = self.get_intensities(inputs)
        #metadata = self.get_metadata(inputs)
        sigiobs = self.get_uncertainties(inputs)
        delta = self.network(tf.concat((iobs, sigiobs), axis=-1))
        sigpred = sigiobs * delta / tf.reduce_mean(delta)
        return self.base_dist(
            tf.squeeze(iobs), 
            tf.squeeze(sigpred),
        )

class NeuralNormalLikelihood(NeuralLikelihood):
    def base_dist(self, loc, scale):
        return tfd.Normal(loc, scale)
