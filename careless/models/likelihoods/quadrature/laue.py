from careless.models.likelihoods import laue
from careless.models.likelihoods.quadrature.base import QuadratureBase
from careless.utils.shame import sanitize_tensor
import tensorflow as tf
import numpy as np

class QuadratureMixin(QuadratureBase):
    def expected_log_likelihood(self, loc, scale, deg=10):
        loc = self.convolve(loc)
        scale = tf.math.sqrt(self.convolve(scale**2.))
        grid, weights = np.polynomial.hermite.hermgauss(deg)
        grid, weights = grid.astype(np.float32), weights.astype(np.float32)
        grid = np.sqrt(2.)*grid[:,None]*scale + loc
        ll = weights[None,:]@sanitize_tensor(self._log_prob(grid))/np.sqrt(np.pi)
        return ll

class NormalLikelihood(laue.NormalLikelihood, QuadratureMixin):
    """"""

class LaplaceLikelihood(laue.LaplaceLikelihood, QuadratureMixin):
    """"""

class StudentTLikelihood(laue.StudentTLikelihood, QuadratureMixin):
    """"""

