"""
Learnable distribution modules for careless.
These are nn.Module subclasses whose parameters can be optimized by gradient descent.
"""

import math
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


def _accept_reject_truncnorm(loc, scale, low, high, shape, max_iter=100):
    """Vectorized accept-reject from Normal(loc, scale) truncated to [low, high].
    For careless (low=0, loc>0), acceptance rate is near 100% so max_iter=10 suffices."""
    x    = torch.empty(shape, dtype=loc.dtype, device=loc.device)
    done = torch.zeros(shape, dtype=torch.bool, device=loc.device)
    for _ in range(max_iter):
        if done.all():
            break
        cand   = loc + scale * torch.randn(shape, dtype=loc.dtype, device=loc.device)
        accept = (cand >= low) & (cand <= high) & ~done
        x      = torch.where(accept, cand, x)
        done   = done | accept
    # Fallback: clamp any unconverged entries (negligible bias, astronomically rare)
    if not done.all():
        x = torch.where(~done, loc.expand(shape).clamp(min=low, max=high), x)
    return x


class _TruncNormIRG(torch.autograd.Function):
    """
    Reparameterized sample via Implicit Reparameterization Gradient (Figurnov et al. 2018).

    Forward : accept-reject sampling — always in [low, high], no gradient tape.
    Backward: IRG formula
        dx/dmu  = 1 + [Phi(t)-Phi(a)] * [phi(a)-phi(b)]       / [Z * phi(t)]
        dx/dsig = t + [Phi(t)-Phi(a)] * [a*phi(a) - b*phi(b)] / [Z * phi(t)]
    where t=(x-mu)/sig, a=(low-mu)/sig, b=(high-mu)/sig, Z=Phi(b)-Phi(a).
    """

    @staticmethod
    def forward(ctx, loc, scale, low, high, shape_tuple):
        shape = torch.Size(shape_tuple)
        x = _accept_reject_truncnorm(loc, scale, low, high, shape)
        ctx.save_for_backward(x, loc, scale, low, high)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, loc, scale, low, high = ctx.saved_tensors
        t = (x    - loc) / scale
        a = (low  - loc) / scale
        b = (high - loc) / scale

        std   = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        phi_t = std.log_prob(t).exp()
        phi_a = std.log_prob(a).exp()
        phi_b = std.log_prob(b).exp()
        Phi_t = std.cdf(t)
        Phi_a = std.cdf(a)
        Phi_b = std.cdf(b)
        Z      = (Phi_b - Phi_a).clamp(min=1e-38)
        phi_t  = phi_t.clamp(min=1e-38)

        c          = (Phi_t - Phi_a) / (Z * phi_t)    # broadcasts over sample dim
        dx_dloc    = 1.0 + c * (phi_a - phi_b)
        dx_dscale  = t   + c * (a * phi_a - b * phi_b)

        # Sum over all sample dimensions (grad_x may be (n_samples, n_reflections))
        n_sample_dims = grad_x.ndim - loc.ndim
        if n_sample_dims > 0:
            sum_dims = list(range(n_sample_dims))
            g_loc   = (grad_x * dx_dloc  ).sum(sum_dims)
            g_scale = (grad_x * dx_dscale).sum(sum_dims)
        else:
            g_loc   = grad_x * dx_dloc
            g_scale = grad_x * dx_dscale
        return g_loc, g_scale, None, None, None


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
        Reparameterized sample via Implicit Reparameterization Gradient (IRG).
        Forward uses accept-reject sampling (always in support).
        Backward uses the IRG formula (Figurnov et al. 2018).
        """
        shape = sample_shape + self.loc.shape
        return _TruncNormIRG.apply(
            self.loc, self.scale, self.low, self.high, tuple(shape)
        )

    def log_prob(self, x):
        """Log probability of x under the truncated normal."""
        loc, scale = self.loc, self.scale
        alpha, beta = self._normalized_bounds()
        std_normal = Normal(torch.zeros_like(loc), torch.ones_like(loc))
        log_normalizer = torch.log(
            (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-38)
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
        Z = (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-38)
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
        Z = (std_normal.cdf(beta) - std_normal.cdf(alpha)).clamp(min=1e-38)
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


from torch.distributions import StudentT as _StudentTBase


class StudentT(_StudentTBase):
    """
    StudentT with TFP-style numerically stable log_prob.

    Uses the two-branch log1psquare formula:
      |y| <= 1 : log1p(y^2)
      |y| >  1 : 2*log|y| + log1p(1/y^2)
    where y = (value - loc) / (scale * sqrt(df)).

    This avoids squaring large residuals in float32, matching TFP's
    numeric.log1psquare approach and giving more accurate gradients for
    outliers beyond ~sqrt(df) sigma (~4 sigma for df=16).
    """

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # Normalize by scale*sqrt(df) (TFP convention avoids large y^2)
        y = (value - self.loc) / (self.scale * self.df.sqrt())
        abs_y = y.abs()
        # Guard against 1/0 in the |y|>1 branch (torch.where evaluates both)
        safe_abs_y = abs_y.clamp(min=1e-30)
        log1p_y2 = torch.where(
            abs_y <= 1.0,
            torch.log1p(y ** 2),
            2.0 * safe_abs_y.log() + torch.log1p(safe_abs_y.pow(-2)),
        )
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * log1p_y2 - Z
