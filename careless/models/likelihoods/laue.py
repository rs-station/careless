from careless.models.likelihoods.mono import Likelihood
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class LaueBase(Likelihood):
    def dist(self, loc, scale):
        raise NotImplementedError(
            """ Extensions of this class must implement self.location_scale_distribution(loc, scale) """
            )

    def call(self, inputs):
        harmonic_id   = self.get_harmonic_id(inputs)
        intensities   = self.get_intensities(inputs)
        uncertainties = self.get_uncertainties(inputs)

        likelihood = self.dist(intensities, uncertainties)

        class ConvolvedLikelihood():
            """
            Convolved log probability object for Laue data.
            """
            def __init__(self, distribution, harmonic_id):
                self.harmonic_id = harmonic_id
                self.distribution = distribution

            def convolve(self, value):
                """
                Takes a set of sample points at which to compute the log prob. 
                values can either be a bare vector or it may have a batch
                dimension for mc samples, ie shape=(b, n_predictions). 
                """
                tv = tf.transpose(value)
                tr = tf.scatter_nd(self.harmonic_id, tv, tv.shape)
                return tf.transpose(tr)

            def mean(self, *args, **kwargs):
                return self.distribution.mean(*args, **kwargs)

            def stddev(self, *args, **kwargs):
                return self.distribution.stddev(*args, **kwargs)

            def log_prob(self, value):
                return self.distribution.log_prob(self.convolve(value))

        return ConvolvedLikelihood(likelihood, harmonic_id)

class NormalLikelihood(LaueBase):
    def dist(self, loc, scale):
        loc = tf.squeeze(loc)
        scale = tf.squeeze(scale)
        return tfd.Normal(loc, scale)

class LaplaceLikelihood(LaueBase):
    def dist(self, loc, scale):
        loc = tf.squeeze(loc)
        scale = tf.squeeze(scale)
        return tfd.Laplace(loc, scale/np.sqrt(2.))

class StudentTLikelihood(LaueBase):
    def __init__(self, dof):
        """
        Parameters
        ----------
        dof : float
            Degrees of freedom of the t-distributed error model.
        """
        super().__init__()
        self.dof = dof

    def dist(self, loc, scale):
        loc = tf.squeeze(loc)
        scale = tf.squeeze(scale)
        return tfd.StudentT(self.dof, loc, scale)

