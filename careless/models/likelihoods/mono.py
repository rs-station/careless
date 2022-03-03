from careless.models.likelihoods.base import Likelihood
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
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

class Ev11Likelihood(LocationScaleLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Sdfac = tfu.TransformedVariable(1., tfb.Softplus())
        self.Sdadd = tfu.TransformedVariable(1., tfb.Softplus())
        self.SdB = tfu.TransformedVariable(1., tfb.Softplus())
        self.loc = None
        self.scale = None

    def call(self, inputs):
        self.loc, self.scale = self.get_loc_and_scale(inputs)
        return self

    def corrected_sigiobs(self, ipred):
        ipred = tf.math.softplus(ipred)
        sigiobs = self.Sdfac * tf.math.sqrt(
            tf.square(self.scale) + \
            self.SdB * ipred + \
            self.Sdadd * tf.square(ipred)
        )
        return sigiobs

class NormalEv11Likelihood(Ev11Likelihood):
    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.Normal(self.loc, scale).log_prob(ipred)

class StudentTEv11Likelihood(Ev11Likelihood):
    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.StudentT(self.dof, self.loc, scale).log_prob(ipred)

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
