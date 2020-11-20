from careless.models.likelihoods.base import Likelihood
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


class MonoBase():
    def log_prob(self, X):
        log_prob = super().log_prob(X)
        if self.weights is None:
            return log_prob
        else:
            return self.weights * log_prob

class NormalLikelihood(tfd.Normal, Likelihood, MonoBase):
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
        super().__init__(loc, scale)

        if weights is None:
            self.weights = None
        else:
            self.weights = np.array(weights, dtype=np.float32)

class LaplaceLikelihood(tfd.Laplace, Likelihood, MonoBase):
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
        super().__init__(loc, scale)

        if weights is None:
            self.weights = None
        else:
            self.weights = np.array(weights, dtype=np.float32)

class StudentTLikelihood(tfd.StudentT, Likelihood, MonoBase):
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
        super().__init__(dof, loc, scale)

        if weights is None:
            self.weights = None
        else:
            self.weights = np.array(weights, dtype=np.float32)
