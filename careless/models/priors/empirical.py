from tensorflow_probability import distributions as tfd
from careless.models.distributions.base import Prior


class SparseReferencePrior(tfd.Laplace, Prior):
    """
    A Laplacian prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, epsilons):
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        epsilons : array
            numpy array or tf.Tensor containing 
        """
        loc = np.float(Fobs/epsilons, dtype=np.float32)
        scale = np.float(SigFobs/epsilons, dtype=np.float32)/np.sqrt(2.)
        super().__init__(loc, scale)

