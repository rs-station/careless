from careless.models.likelihoods.base import Likelihood
from careless.models.base import PerGroupModel
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

class ConvolvedDist():
    """This mixin can be used to extend tensorflow_probability distributions to apply a convolution before computing probabilities."""
    def log_prob(self, value, name='log_prob', **kwargs):
        return super().log_prob(self.convolve(value), name, **kwargs)

    def prob(self, value, name='prob', **kwargs):
        return super().prob(self.convolve(value), name, **kwargs)

    def convolve(self, tensor):
        """
        Paramters
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

class NormalLikelihood(ConvolvedDist, tfd.Normal, Likelihood):
    def __init__(self, iobs, sigiobs, harmonic_id):
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
        super().__init__(loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

class LaplaceLikelihood(ConvolvedDist, tfd.Laplace, Likelihood):
    def __init__(self, iobs, sigiobs, harmonic_id):
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
        super().__init__(loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

class StudentTLikelihood(ConvolvedDist, tfd.StudentT, Likelihood):
    def __init__(self, iobs, sigiobs, harmonic_id, dof):
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
        super().__init__(dof, loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

