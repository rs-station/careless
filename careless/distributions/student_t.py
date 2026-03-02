"""
Numerically stable Student-T distribution for careless likelihoods.
"""
import math
import torch
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
