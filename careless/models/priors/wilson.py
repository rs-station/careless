from careless.utils.distributions import Stacy
from careless.models.priors.base import Prior
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
import numpy as np


class Centric(tfd.HalfNormal, Prior):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(tf.math.sqrt(epsilon * self.sigma))


class Acentric(tfd.Weibull, Prior):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(
            2., 
            tf.math.sqrt(self.epsilon * self.sigma),
        )

class WilsonPrior(Prior):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon, sigma=1.):
        """
        Parameters
        ----------
        centric : array
            Floating point or boolean array with value 1/True for centric reflections and 0/False. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. 
        """
        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=np.bool)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.p_centric = Centric(self.epsilon, self.sigma)
        self.p_acentric = Acentric(self.epsilon, self.sigma)

    def log_prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.log_prob(x), self.p_acentric.log_prob(x))

    def prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.prob(x), self.p_acentric.prob(x))

    def mean(self):
        return tf.where(self.centric, self.p_centric.mean(), self.p_acentric.mean())

    def stddev(self):
        return tf.where(self.centric, self.p_centric.stddev(), self.p_acentric.stddev())

    def sample(self, *args, **kwargs):
        #### BLEERRRGG #####
        return tf.where(
            self.centric, 
            self.p_centric.sample(*args, **kwargs),
            self.p_acentric.sample(*args, **kwargs),
        )

