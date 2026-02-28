import torch
import math
from torch.distributions import Normal, Laplace, StudentT
from careless.models.likelihoods.base import Likelihood
from careless.models.likelihoods.mono import (
    NormalEv11Likelihood as MonoNormalEv11Likelihood,
    StudentTEv11Likelihood as MonoStudentTEv11Likelihood,
)


class ConvolvedLikelihood:
    """
    Wraps a base distribution with Laue harmonic convolution.
    Intensity predictions for each harmonic are summed before evaluating log_prob.
    """

    def __init__(self, distribution, harmonic_id):
        """
        Parameters
        ----------
        distribution : torch.distributions.Distribution
            Base likelihood distribution over individual harmonic intensities.
        harmonic_id : Tensor (int64)
            Shape (n_obs,) mapping each observation to its harmonic group index.
        """
        self.distribution = distribution
        self.harmonic_id = harmonic_id.squeeze(-1)

    def convolve(self, value):
        """
        Sum contributions from the same harmonic group.

        Parameters
        ----------
        value : Tensor
            Shape (..., n_obs) — predictions for each harmonic observation.

        Returns
        -------
        Tensor
            Same shape as value, with harmonic contributions accumulated.
        """
        # value: (..., n_obs)
        n_obs = value.shape[-1]
        n_harmonics = int(self.harmonic_id.max().item()) + 1
        shape = value.shape[:-1] + (n_harmonics,)
        out = torch.zeros(shape, dtype=value.dtype, device=value.device)
        # Accumulate: out[..., harmonic_id[i]] += value[..., i]
        idx = self.harmonic_id.expand_as(value) if value.dim() > 1 else self.harmonic_id
        out.scatter_add_(-1, idx, value)
        # Gather back to original indexing so output has same size as input
        return out[..., self.harmonic_id]

    def log_prob(self, value):
        return self.distribution.log_prob(self.convolve(value))

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def stddev(self):
        return self.distribution.stddev


class LaueBase(Likelihood):
    """Base class for Laue likelihoods that operate over harmonic observations."""

    def dist(self, inputs):
        raise NotImplementedError(
            "Subclasses must implement dist(inputs) returning a base distribution."
        )

    def forward(self, inputs):
        harmonic_id = self.get_harmonic_id(inputs)
        base = self.dist(inputs)
        return ConvolvedLikelihood(base, harmonic_id)


class NormalEv11Likelihood(LaueBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono = MonoNormalEv11Likelihood()

    def dist(self, inputs):
        return self.mono(inputs)


class StudentTEv11Likelihood(LaueBase):
    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono = MonoStudentTEv11Likelihood(dof)

    def dist(self, inputs):
        return self.mono(inputs)


class NormalLikelihood(LaueBase):
    def dist(self, inputs):
        loc = self.get_intensities(inputs).squeeze(-1).float()
        scale = self.get_uncertainties(inputs).squeeze(-1).float()
        return Normal(loc, scale)


class LaplaceLikelihood(LaueBase):
    def dist(self, inputs):
        loc = self.get_intensities(inputs).squeeze(-1).float()
        scale = self.get_uncertainties(inputs).squeeze(-1).float()
        return Laplace(loc, scale / math.sqrt(2.0))


class StudentTLikelihood(LaueBase):
    def __init__(self, dof):
        super().__init__()
        self.dof = float(dof)

    def dist(self, inputs):
        loc = self.get_intensities(inputs).squeeze(-1).float()
        scale = self.get_uncertainties(inputs).squeeze(-1).float()
        return StudentT(self.dof, loc, scale)
