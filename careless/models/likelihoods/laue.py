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

class LaueBase(ConvolvedDist):
    weights = None

    def log_prob(self, value, name='log_prob', **kwargs):
        log_probs = super().log_prob(value, name, **kwargs)
        if self.weights is None:
            return log_probs
        else:
            return self.weights * log_probs

class NormalLikelihood(LaueBase, tfd.Normal, Likelihood):
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
        super().__init__(loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)

class LaplaceLikelihood(LaueBase, tfd.Laplace, Likelihood):
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
        super().__init__(loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)

class StudentTLikelihood(LaueBase, tfd.StudentT, Likelihood):
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
        super().__init__(dof, loc, scale)
        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)

class SdfacLikelihood(ConvolvedDist, tfd.Normal, Likelihood):
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
        self.iobs = tf.convert_to_tensor(np.array(iobs, dtype=np.float32))
        self.sigiobs = tf.convert_to_tensor(np.array(sigiobs, dtype=np.float32))

        smol = 1e-5
        self.Sdfac = tfp.util.TransformedVariable(1., tfb.Softplus())
        self.SdB   = tfp.util.TransformedVariable(smol, tfb.Softplus())
        self.SdAdd = tfp.util.TransformedVariable(smol, tfb.Softplus())
        sigprime = self.Sdfac * tf.sqrt(self.sigiobs**2. + self.SdB*self.iobs + (self.SdAdd * self.iobs)**2.)
        super().__init__(self.iobs, self.sigiobs)

        self.harmonic_index = np.array(harmonic_id, dtype=np.int32)
        self.harmonic_convolution_tensor = PerGroupModel(self.harmonic_index).expansion_tensor

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)


    @property
    def loc(self):
        return self.iobs

    @property
    def scale(self):
        return self.Sdfac * tf.sqrt(self.sigiobs**2. + self.SdB*self.iobs + (self.SdAdd * self.iobs)**2.)
