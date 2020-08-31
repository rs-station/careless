from careless.models.likelihoods import mono
from careless.models.likelihoods.quadrature.base import QuadratureBase
from careless.utils.shame import sanitize_tensor
import numpy as np

class QuadratureMixin(QuadratureBase):
    def expected_log_likelihood(self, loc, scale, deg=10):
        grid, weights = np.polynomial.hermite.hermgauss(deg)
        grid, weights = grid.astype(np.float32), weights.astype(np.float32)
        grid = np.sqrt(2.)*grid[:,None]*scale + loc
        ll = weights[None,:]@sanitize_tensor(self.log_prob(grid))/np.sqrt(np.pi)
        return ll

class NormalLikelihood(mono.NormalLikelihood, QuadratureMixin):
    """"""

class LaplaceLikelihood(mono.LaplaceLikelihood, QuadratureMixin):
    """"""

class StudentTLikelihood(mono.StudentTLikelihood, QuadratureMixin):
    """"""
