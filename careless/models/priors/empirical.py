import torch
import numpy as np
import math
from torch.distributions import Laplace, Normal, StudentT
from careless.models.priors.base import Prior
from rs_distributions.distributions import FoldedNormal, Rice


class ReferencePrior(Prior):
    """
    Prior with a log_prob implementation that returns zeros for unobserved Miller indices.
    Subclasses must set self.base_dist with a torch.distributions.Distribution or similar.
    """
    base_dist = None

    def __init__(self, observed=None):
        super().__init__()
        if observed is None:
            self.register_buffer('idx', None)
        else:
            idx = torch.where(torch.as_tensor(observed))[0]
            self.register_buffer('idx', idx)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev

    def log_prob(self, values):
        if self.idx is None:
            return self.base_dist.log_prob(values)
        # values shape: (..., n_refls)
        obs = values[..., self.idx]
        log_p_obs = self.base_dist.log_prob(obs)
        # Scatter back to full size, filling unobserved with 0
        n_refls = values.shape[-1]
        shape = values.shape[:-1] + (n_refls,)
        result = torch.zeros(shape, dtype=values.dtype, device=values.device)
        result[..., self.idx] = log_p_obs
        return result


class LaplaceReferencePrior(ReferencePrior):
    """Laplace prior centered at reference structure factor amplitudes."""

    def __init__(self, Fobs, SigFobs, observed=None):
        super().__init__(observed)
        loc = torch.as_tensor(np.array(Fobs, dtype=np.float32))
        scale = torch.as_tensor(np.array(SigFobs, dtype=np.float32)) / math.sqrt(2.0)
        self.base_dist = Laplace(loc, scale)


class NormalReferencePrior(ReferencePrior):
    """Normal prior centered at reference structure factor amplitudes."""

    def __init__(self, Fobs, SigFobs, observed=None):
        super().__init__(observed)
        loc = torch.as_tensor(np.array(Fobs, dtype=np.float32))
        scale = torch.as_tensor(np.array(SigFobs, dtype=np.float32))
        self.base_dist = Normal(loc, scale)


class StudentTReferencePrior(ReferencePrior):
    """Student-T prior centered at reference structure factor amplitudes."""

    def __init__(self, Fobs, SigFobs, dof, observed=None):
        super().__init__(observed)
        loc = torch.as_tensor(np.array(Fobs, dtype=np.float32))
        scale = torch.as_tensor(np.array(SigFobs, dtype=np.float32))
        self.base_dist = StudentT(float(dof), loc, scale)


class RiceWoolfsonReferencePrior(ReferencePrior):
    """Rice/Woolfson prior centered at reference structure factor amplitudes."""

    def __init__(self, Fobs, SigFobs, centric, observed=None):
        super().__init__(observed)
        loc = torch.as_tensor(np.array(Fobs, dtype=np.float32))
        scale = torch.as_tensor(np.array(SigFobs, dtype=np.float32))
        centric_t = torch.as_tensor(np.array(centric, dtype=bool))

        # Store as a hybrid distribution wrapper
        self._loc = loc
        self._scale = scale
        self.register_buffer('centric', centric_t)

        self._rice = Rice(loc.abs(), scale)
        self._folded = FoldedNormal(loc, scale)

    @property
    def mean(self):
        return torch.where(self.centric, self._folded.mean, self._rice.mean)

    @property
    def stddev(self):
        return torch.where(self.centric, self._folded.variance.sqrt(), self._rice.variance.sqrt())

    def log_prob(self, values):
        if self.idx is None:
            log_p_c = self._folded.log_prob(values)
            log_p_a = self._rice.log_prob(values)
            return torch.where(self.centric, log_p_c, log_p_a)
        obs = values[..., self.idx]
        log_p_c = FoldedNormal(self._loc[self.idx], self._scale[self.idx]).log_prob(obs)
        log_p_a = Rice(self._loc[self.idx].abs(), self._scale[self.idx]).log_prob(obs)
        log_p_obs = torch.where(self.centric[self.idx], log_p_c, log_p_a)
        n_refls = values.shape[-1]
        shape = values.shape[:-1] + (n_refls,)
        result = torch.zeros(shape, dtype=values.dtype, device=values.device)
        result[..., self.idx] = log_p_obs
        return result
