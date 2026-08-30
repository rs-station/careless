"""
Degenerate (point mass) distribution for careless scaling models.

`torch.distributions` has no equivalent of `tfd.Deterministic`, which the
TensorFlow implementation of the tabulated spectral scaler returns. Scaling
models are expected to hand back a distribution, so a scaler with no stochastic
component needs a distribution concentrated on a single point.

`scale` is exposed and fixed at zero. That is not decoration: it lets a
deterministic scaler work with `train_model(deterministic_scale_noise=True)`,
which forms samples as ``loc + noise * scale`` instead of calling `rsample`, and
so reduces to `loc` here without a special case.
"""
import torch
from torch.distributions import Distribution, constraints


class Deterministic(Distribution):
    """
    Point mass at `loc`.

    Parameters
    ----------
    loc : Tensor
        Location of the point mass.
    """

    arg_constraints = {'loc': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, validate_args=None):
        self.loc = torch.as_tensor(loc)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    @property
    def stddev(self):
        return torch.zeros_like(self.loc)

    @property
    def scale(self):
        """Always zero. See the module docstring for why this is exposed."""
        return torch.zeros_like(self.loc)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Deterministic, _instance)
        new.loc = self.loc.expand(torch.Size(batch_shape))
        super(Deterministic, new).__init__(torch.Size(batch_shape), validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # Reparameterized, trivially: the sample is `loc` itself, so gradients
        # flow to whatever produced it.
        shape = self._extended_shape(torch.Size(sample_shape))
        return self.loc.expand(shape)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        """
        0 where `value == loc` and -inf elsewhere.

        This is the log of an indicator, not of a density, so it cannot be
        combined with a continuous prior to give a finite KL divergence. A
        deterministic scaler therefore should not be given a scale prior; see
        `TabulatedSpectralScaler`.
        """
        if self._validate_args:
            self._validate_sample(value)
        return torch.where(
            value == self.loc,
            torch.zeros_like(self.loc),
            torch.full_like(self.loc, float('-inf')),
        )

    def entropy(self):
        return torch.zeros_like(self.loc)
