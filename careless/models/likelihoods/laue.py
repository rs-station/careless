from careless.models.likelihoods.mono import Likelihood
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

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

class LaueBase(Likelihood):
    def dist(self, inputs):
        raise NotImplementedError(
            """ Extensions of this class must implement self.location_scale_distribution(loc, scale) """
            )

    def call(self, inputs):
        harmonic_id   = self.get_harmonic_id(inputs)

        likelihood = self.dist(inputs)

        return ConvolvedLikelihood(likelihood, harmonic_id)

from careless.models.likelihoods.mono import NormalEv11Likelihood as MonoNormalEv11Likelihood
class NormalEv11Likelihood(LaueBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono = MonoNormalEv11Likelihood()

    def dist(self, inputs):
        return self.mono(inputs)

from careless.models.likelihoods.mono import StudentTEv11Likelihood as MonoStudentTEv11Likelihood
class StudentTEv11Likelihood(LaueBase):
    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono = MonoStudentTEv11Likelihood(dof)

    def dist(self, inputs):
        return self.mono(inputs)


class NormalLikelihood(LaueBase):
    def dist(self, inputs):
        loc = self.get_intensities(inputs)
        scale = self.get_uncertainties(inputs)
        loc = tf.squeeze(loc)
        scale = tf.squeeze(scale)
        return tfd.Normal(loc, scale)

class LaplaceLikelihood(LaueBase):
    def dist(self, inputs):
        loc = self.get_intensities(inputs)
        scale = self.get_uncertainties(inputs)
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

    def dist(self, inputs):
        loc = self.get_intensities(inputs)
        scale = self.get_uncertainties(inputs)
        loc = tf.squeeze(loc)
        scale = tf.squeeze(scale)
        return tfd.StudentT(self.dof, loc, scale)

