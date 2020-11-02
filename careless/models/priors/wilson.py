from careless.utils.distributions import Stacy
from careless.models.priors.base import Prior
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
import numpy as np


#def Centric(**kw):
#    """
#    According to Herr Rupp, the pdf for centric normalized structure factor 
#    amplitudes, E, is:
#    P(E) = (2/pi)**0.5 * exp(-0.5*E**2)
#
#    In common parlance, this is known as halfnormal distribution with scale=1.0
#
#    RETURNS
#    -------
#    dist : tfp.distributions.HalfNormal
#        Centric normalized structure factor distribution
#    """
#    dist = tfd.HalfNormal(scale=1., **kw)
#    return dist

def Acentric(**kw):
    """
    According to Herr Rupp, the pdf for acentric normalized structure factor 
    amplitudes, E, is:
    P(E) = 2*E*exp(-E**2)

    This is exactly a Raleigh distribution with sigma**2 = 1. This is also
    the same as a Chi distribution with k=2 which has been transformed by 
    rescaling the argument. 

    RETURNS
    -------
    dist : tfp.distributions.TransformedDistribution
        Centric normalized structure factor distribution
    """
    dist = tfd.Weibull(2., 1., **kw)
    return dist


class StacyWilsonPrior(Stacy, Prior):
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
        theta_centric,alpha_centric,beta_centric = Stacy._stacy_params(tfd.HalfNormal(np.sqrt(self.epsilon)))
        theta_acentric,alpha_acentric,beta_acentric = Stacy._stacy_params(tfd.Weibull(2., np.sqrt(self.epsilon)))

        theta = self.centric*theta_centric + (1. - self.centric)*theta_acentric
        alpha = self.centric*alpha_centric + (1. - self.centric)*alpha_acentric
        beta  = self.centric*beta_centric  + (1. - self.centric)*beta_acentric

        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        beta  = tf.convert_to_tensor(beta , dtype=tf.float32)

        super().__init__(theta, alpha, beta)

class Centric(tfd.TransformedDistribution, Prior):
    def __init__(self, epsilon):
        self.epsilon = tf.convert_to_tensor(epsilon)
        super().__init__(
            tfd.Normal(0., tf.math.sqrt(self.epsilon)),
            tfb.AbsoluteValue(),
        )

    def log_prob(self, X):
        return tf.where(X >= 0., super().log_prob(X), -np.inf)

    def prob(self, X):
        return tf.where(X >= 0., super().log_prob(X), 0.)

    def mean(self):
        return super().distribution.scale * np.sqrt(2. / np.pi)

    def stddev(self):
        return super().distribution.scale * np.sqrt(1. - 2 / np.pi)

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

