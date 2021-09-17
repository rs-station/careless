import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from careless.models.priors.base import Prior
from careless.models.merging.surrogate_posteriors import RiceWoolfson


class ReferencePrior():
    """
    A Prior class with a `log_prob` implementation that returns zeros for unobserved miller indices.
     - This class is not meant to be used directly. 
     - Extensions of this class must set the `base_dist` attribute with a tfd.Distribution or similar object with
       a `.log_prob(values)` method.
    """
    base_dist = None
    def __init__(self, observed=None):
        if observed is None:
            self.idx = None
        else:
            idx = tf.where(observed)
            self.idx = tf.reshape(idx, (-1,))

    def mean(self):
        """ This just passes through to self.base_dist. """
        return self.base_dist.mean()

    def stddev(self):
        """ This just passes through to self.base_dist. """
        return self.base_dist.stddev()

    def log_prob(self, values):
        if self.idx is None:
            return self.base_dist.log_prob(values)
        obs = tf.gather(values, self.idx, axis=-1)
        log_prob = self.base_dist.log_prob(obs)

        return tf.transpose(tf.scatter_nd(
            self.idx[...,None], 
            tf.transpose(log_prob), 
            values.shape[::-1]
        ))

class LaplaceReferencePrior(ReferencePrior):
    """
    A Laplacian prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, observed=None):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)/np.sqrt(2.)
        self.base_dist = tfd.Laplace(loc, scale)

class NormalReferencePrior(ReferencePrior):
    """
    A Normal prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, observed=None):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = tfd.Normal(loc, scale)

class StudentTReferencePrior(ReferencePrior):
    """
    A Student's T prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, dof, observed=None):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        dof : float
            degrees of freedom for the student's t distribution.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = tfd.StudentT(dof, loc, scale)

class RiceWoolfsonReferencePrior(ReferencePrior):
    """
    A Rice Woolfson prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, centric, observed=None):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        centric : array
            boolean numpy array or tf.Tensor which has True for all centric reflections.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = RiceWoolfson(loc, scale, centric)


