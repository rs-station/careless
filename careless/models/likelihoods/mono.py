from careless.models.likelihoods.base import Likelihood
from tensorflow_probability import distributions as tfd
import numpy as np


class NormalLikelihood(tfd.Normal, Likelihood):
    def __init__(self, iobs, sigiobs):
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

class LaplaceLikelihood(tfd.Laplace, Likelihood):
    def __init__(self, iobs, sigiobs):
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

class StudentTLikelihood(tfd.StudentT, Likelihood):
    def __init__(self, iobs, sigiobs, dof):
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

