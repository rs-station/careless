from careless.utils.distributions import Stacy
from careless.models.priors.base import Prior
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
import numpy as np


class Centric(tfd.HalfNormal, Prior):
    def __init__(self, epsilon):
        self.epsilon = tf.convert_to_tensor(epsilon)
        super().__init__(tf.math.sqrt(epsilon))


class Acentric(tfd.Weibull, Prior):
    def __init__(self, epsilon):
        self.epsilon = tf.convert_to_tensor(epsilon)
        super().__init__(
            2., 
            tf.math.sqrt(self.epsilon),
        )

class WilsonPrior(Prior):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon):
        """
        Parameters
        ----------
        centric : array
            Floating point or boolean array with value 1/True for centric reflections and 0/False. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        """
        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=np.bool)
        self.p_centric = Centric(self.epsilon)
        self.p_acentric = Acentric(self.epsilon)

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

