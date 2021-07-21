from careless.models.likelihoods.base import Likelihood
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


class MonoBase(Likelihood):
    def __init__(self, distribution, weights=None):
        """
        Parameters
        ----------
        distribution : tensorflow_probability.distributions.Distribution
        weights : tensorflow.Tensor

        Attributes
        ----------
        likelihood : tensorflow_probability.distributions.Distribution
        weights : tensorflow.Tensor
        """
        self.likelihood = distribution
        if weights is not None:
            weights = np.array(weights, dtype=np.float32)
        self.weights = weights 

    def sample(self, *args, **kwargs):
        return self.likelihood.sample(*args, **kwargs)

    def log_prob(self, X):
        log_prob = self.likelihood.log_prob(X)
        if self.weights is None:
            return log_prob
        else:
            return self.weights * log_prob

    def prob(self, X):
        prob = self.likelihood.prob(X)
        if self.weights is None:
            return prob
        else:
            return self.weights * prob

class NormalLikelihood(MonoBase):
    def __init__(self, iobs, sigiobs, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        sigiobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)
        likelihood = tfd.Normal(loc, scale)
        super().__init__(likelihood, weights)

class LaplaceLikelihood(MonoBase):
    def __init__(self, iobs, sigiobs, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        sigiobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)/np.sqrt(2.)
        likelihood = tfd.Laplace(loc, scale)
        super().__init__(likelihood, weights)

class StudentTLikelihood(MonoBase):
    def __init__(self, iobs, sigiobs, dof, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        sigiobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        dof : float
            Degrees of freedom of the student t likelihood.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)
        likelihood = tfd.StudentT(dof, loc, scale)
        super().__init__(likelihood, weights)

