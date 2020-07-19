from careless.models.likelihoods.base import Likelihood
from careless.models.base import PerGroupModel
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import numpy as np

class HarmonicLikelihood():
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
        convolved = tf.squeeze(tf.sparse.sparse_dense_matmul(
            self.harmonic_convolution_tensor, 
            tf.expand_dims(tensor, -1), 
            adjoint_a=True
        ))
        return convolved

class NormalLikelihood(HarmonicLikelihood, tfd.Normal):
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

class LaplaceLikelihood(HarmonicLikelihood, tfd.Laplace):
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

class StudentTLikelihood(HarmonicLikelihood, tfd.StudentT):
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
        scale = np.array(sigiobs, dtype=np.float32)/np.sqrt(2.)
        super().__init__(dof, loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

