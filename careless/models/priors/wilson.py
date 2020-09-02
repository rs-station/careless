from careless.models.priors.base import Prior
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import numpy as np


def Centric(**kw):
    """
    According to Herr Rupp, the pdf for centric normalized structure factor 
    amplitudes, E, is:
    P(E) = (2/pi)**0.5 * exp(-0.5*E**2)

    In common parlance, this is known as halfnormal distribution with scale=1.0

    RETURNS
    -------
    dist : tfp.distributions.HalfNormal
        Centric normalized structure factor distribution
    """
    dist = tfd.HalfNormal(scale=1., **kw)
    return dist

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

class WilsonPrior(Prior):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon):
        """
        Parameters
        ----------
        centric : array
            Floating point array with value 1. for centric reflections and 0. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        """
        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=np.float32)
        self.p_centric = Centric()
        self.p_acentric = Acentric()

    def log_prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        x = x / np.sqrt(self.epsilon)
        return self.centric*self.p_centric.log_prob(x) + (1. - self.centric)*self.p_acentric.log_prob(x)

    def prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        x = x / np.sqrt(self.epsilon)
        return self.centric*self.p_centric.prob(x) + (1. - self.centric)*self.p_acentric.prob(x)

    def mean(self):
        return np.sqrt(self.epsilon)*(self.centric*self.p_centric.mean() + (1. - self.centric)*self.p_acentric.mean())

    def stddev(self):
        return np.sqrt(self.epsilon)*(self.centric*self.p_centric.stddev() + (1. - self.centric)*self.p_acentric.stddev())

