from careless.models.likelihoods.base import Likelihood
from careless.models.likelihoods.mono import MonoBase
from careless.models.base import PerGroupModel
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class LaueBase(MonoBase):
    def __init__(self, distribution, harmonic_convolution_tensor, weights=None):
        """
        Parameters
        ----------
        distribution : tensorflow_probability.distributions.Distribution
            The likelihood distribution. There should be one entry for each reflection observation
        harmonic_convolution_tensor : tensorflow.SparseTensor
            A sparse binary tensor defining which observations are harmonics. 
        weights : tensorflow.Tensor or None
            Optional weight vector applied to the log probabilities. 

        Attributes
        ----------
        likelihood : tensorflow_probability.distributions.Distribution
        weights : tensorflow.Tensor
        """
        self.likelihood = distribution
        if weights is not None:
            weights = np.array(weights, dtype=np.float32)
        self.weights = weights
        self.harmonic_convolution_tensor = harmonic_convolution_tensor

    def log_prob(self, value, name='log_prob', **kwargs):
        log_probs = self.likelihood.log_prob(self.convolve(value), name, **kwargs)
        if self.weights is None:
            return log_probs
        else:
            return self.weights * log_probs

    def prob(self, value, name='prob', **kwargs):
        probs = self.likelihood.prob(self.convolve(value), name, **kwargs)
        if self.weights is None:
            return probs
        else:
            return self.weights * probs

    def convolve(self, tensor):
        """
        Parameters
        ---------
        tensor : tf.Tensor
            array of predicted reflection intensities with length self.harmonic_convolution_tensor.shape[1]
        
        Returns
        -------
        convolved : tf.Tensor
            array of predicted reflection intensities which have been convolved by a sparse matmul
        """
        if len(tensor.shape) == 1:
            convolved = tf.squeeze(tf.sparse.sparse_dense_matmul(
                self.harmonic_convolution_tensor, 
                tf.expand_dims(tensor, -1), 
                adjoint_a=True
            ))
        else:
            convolved = tf.transpose(tf.sparse.sparse_dense_matmul(
                self.harmonic_convolution_tensor, 
                tensor,
                adjoint_a=True,
                adjoint_b=True,
            ))
        return convolved

class NormalLikelihood(LaueBase):
    def __init__(self, iobs, sigiobs, harmonic_id, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        iobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        harmonic_index : array(int)
            Integer ids dictating which predictions will be convolved.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)

        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        likelihood = tfd.Normal(loc, scale)
        super().__init__(likelihood, harmonic_convolution_tensor, weights)

class LaplaceLikelihood(LaueBase):
    def __init__(self, iobs, sigiobs, harmonic_id, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        iobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        harmonic_index : array(int)
            Integer ids dictating which predictions will be convolved.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)/np.sqrt(2.)

        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        likelihood = tfd.Laplace(loc, scale)
        super().__init__(likelihood, harmonic_convolution_tensor, weights)

class StudentTLikelihood(LaueBase):
    def __init__(self, iobs, sigiobs, harmonic_id, dof, weights=None):
        """
        Parameters
        ----------
        iobs : array or tensor
            Numpy array or tf.Tensor of observed reflection intensities.
        iobs : array or tensor
            Numpy array or tf.Tensor of reflection intensity error estimates.
        harmonic_index : array(int)
            Integer ids dictating which predictions will be convolved.
        dof : float
            Degrees of freedom.
        """
        loc = np.array(iobs, dtype=np.float32)
        scale = np.array(sigiobs, dtype=np.float32)

        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        likelihood = tfd.StudentT(dof, loc, scale)
        super().__init__(likelihood, harmonic_convolution_tensor, weights)

