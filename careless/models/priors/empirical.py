from tensorflow_probability import distributions as tfd
import numpy as np
from careless.models.priors.base import Prior
from careless.models.merging.surrogate_posteriors import RiceWoolfson


class LaplaceReferencePrior(tfd.Laplace, Prior):
    """
    A Laplacian prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        """
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)/np.sqrt(2.)
        super().__init__(loc, scale)


class NormalReferencePrior(tfd.Normal, Prior):
    """
    A Normal prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        """
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        super().__init__(loc, scale)


class StudentTReferencePrior(tfd.StudentT, Prior):
    """
    A Student's T prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, dof):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        dof : float
            Degrees of freedom for the Student T distribution.
        """
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        super().__init__(dof, loc, scale)

class RiceWoolfsonReferencePrior(RiceWoolfson, Prior):
    """
    Just an alias to the Rice Woolfson surrogate posterior.
    """
