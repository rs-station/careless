"""
Learnable distribution modules for careless.
These are nn.Module subclasses whose parameters can be optimized by gradient descent.
"""

import torch
import torch.nn as nn
import numpy as np
from rs_distributions.modules import TransformedParameter
from torch.distributions.transforms import (
    ExpTransform,
    AffineTransform,
    ComposeTransform,
)
from torch.distributions import constraints, Normal


class TruncatedNormal(nn.Module):
    """
    Learnable TruncatedNormal surrogate posterior for structure factor amplitudes.

    The loc and scale parameters are optimized via gradient descent using bijectors:
      - loc  : raw → exp(raw)         (keeps amplitude positive)
      - scale: raw → exp(raw) + shift  (keeps scale positive and bounded away from zero)

    low and high are fixed bounds registered as buffers.
    """

    def __init__(self, loc, scale, low=0.0, high=1e10, scale_shift=1e-7):
        """
        Parameters
        ----------
        loc : array-like
            Initial location parameters (one per reflection). Should be positive.
        scale : array-like
            Initial scale parameters (one per reflection). Should be positive.
        low : float or array-like
            Lower truncation bound (default 0, i.e. non-negative support).
        high : float or array-like
            Upper truncation bound.
        scale_shift : float
            Minimum scale value added for numerical stability.
        """
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)

        # loc bijector: unconstrained → positive via exp
        self._loc = TransformedParameter(loc, ExpTransform())

        # scale bijector: unconstrained → exp(raw) + scale_shift
        scale_transform = ComposeTransform([
            ExpTransform(),
            AffineTransform(scale_shift, 1.0),
        ])
        self._scale = TransformedParameter(scale - scale_shift, scale_transform)

        low = torch.as_tensor(low, dtype=torch.float32) * torch.ones_like(loc)
        high = torch.as_tensor(high, dtype=torch.float32) * torch.ones_like(loc)
        self.register_buffer('low', low)
        self.register_buffer('high', high)
        self.scale_shift = scale_shift

    @property
    def loc(self):
        return self._loc()

    @property
    def scale(self):
        return self._scale()

    def _normalized_bounds(self):
        """Return (alpha, beta) = ((low - loc) / scale, (high - loc) / scale)."""
        alpha = (self.low - self.loc) / self.scale
        beta = (self.high - self.loc) / self.scale
        return alpha, beta

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sample using the inverse-CDF transform.
        Gradients flow through loc and scale.
        """
        loc, scale = self.loc, self.scale
        alpha, beta = self._normalized_bounds()

        std_normal = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        Phi_alpha = std_normal.cdf(alpha)
        Phi_beta = std_normal.cdf(beta)

        shape = sample_shape + loc.shape
        u = torch.zeros(shape, dtype=loc.dtype, device=loc.device).uniform_()
        # Inverse CDF: Phi_inv(Phi_alpha + u * (Phi_beta - Phi_alpha))
        p = Phi_alpha + u * (Phi_beta - Phi_alpha)
        p = p.clamp(1e-6, 1.0 - 1e-6)
        # Phi_inv(p) = sqrt(2) * erfinv(2p - 1)
        z = torch.erfinv(2.0 * p - 1.0) * (2.0 ** 0.5)
        return loc + scale * z

    def log_prob(self, x):
        """Log probability of x under the truncated normal."""
        loc, scale = self.loc, self.scale
        alpha, beta = self._normalized_bounds()
        std_normal = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        log_normalizer = torch.log(
            (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-10)
        )
        std_normal_x = Normal(loc, scale)
        log_p = std_normal_x.log_prob(x) - log_normalizer
        # Zero probability outside support
        in_support = (x >= self.low) & (x <= self.high)
        return torch.where(in_support, log_p, torch.full_like(log_p, -1e38))

    @property
    def mean(self):
        loc, scale = self.loc, self.scale
        alpha, beta = self._normalized_bounds()
        std_normal = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        phi_alpha = torch.exp(std_normal.log_prob(alpha))
        phi_beta = torch.exp(std_normal.log_prob(beta))
        Z = (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-10)
        return loc + scale * (phi_alpha - phi_beta) / Z

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        loc, scale = self.loc, self.scale
        alpha, beta = self._normalized_bounds()
        std_normal = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        phi_alpha = torch.exp(std_normal.log_prob(alpha))
        phi_beta = torch.exp(std_normal.log_prob(beta))
        Z = (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-10)
        mean_correction = (phi_alpha - phi_beta) / Z
        var = scale ** 2 * (
            1.0
            + (alpha * phi_alpha - beta * phi_beta) / Z
            - mean_correction ** 2
        )
        return var

    def moment_4(self, method='scipy'):
        """
        Fourth moment of the distribution, used to compute SigI from SigF.

        Parameters
        ----------
        method : str
            Only 'scipy' is supported; uses scipy.stats.truncnorm.
        """
        if method == 'scipy':
            from scipy.stats import truncnorm
            loc = self.loc.detach().cpu().numpy()
            scale = self.scale.detach().cpu().numpy()
            low = self.low.cpu().numpy()
            high = self.high.cpu().numpy()
            a = (low - loc) / scale
            b = (high - loc) / scale
            return truncnorm.moment(4, a, b, loc, scale)
        raise ValueError(f"Unknown method '{method}' for computing moment_4")

    def parameter_properties(self):
        """Return the names of learnable parameters."""
        return ['loc', 'scale']

    @property
    def parameters_dict(self):
        """Return learnable parameters as numpy arrays (for output writing)."""
        return {
            'loc': self.loc.detach().cpu().numpy(),
            'scale': self.scale.detach().cpu().numpy(),
        }

    @classmethod
    def from_loc_and_scale(cls, loc, scale, low=0.0, high=1e10, scale_shift=1e-7):
        """
        Construct a learnable TruncatedNormal.

        Parameters
        ----------
        loc : array-like
            Initial location (positive values expected).
        scale : array-like
            Initial scale.
        low : float or array-like
            Lower truncation bound.
        high : float or array-like
            Upper truncation bound.
        scale_shift : float
            Minimum additive shift for scale stability.
        """
        return cls(loc, scale, low, high, scale_shift)
