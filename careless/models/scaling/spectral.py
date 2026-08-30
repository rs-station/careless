import torch
import torch.nn as nn
import numpy as np
from careless.models.scaling.base import Scaler
from careless.distributions.deterministic import Deterministic

class TabulatedSpectralScaler(Scaler):
    """
    A scaler that uses a pre-calculated regular grid lookup table for fast spectral scaling.
    """
    def __init__(self, x_grid, y_grid, trainable_scale=False, initial_value=1.0, num_grid_points=10000,
                 lorentz_correction=False):
        """
        Parameters
        ----------
        x_grid : array-like
            Input wavelengths (irregular).
        y_grid : array-like
            Input scale factors.
        trainable_scale : bool
            Enable global learnable multiplier.
        initial_value : float
            Initial value for global multiplier.
        num_grid_points : int
            Size of the regular lookup grid.
        """
        super().__init__()

        x_grid = np.asarray(x_grid)
        y_grid = np.asarray(y_grid)

        # 1. Resample onto Regular Grid (NumPy)
        self.x_min = float(np.min(x_grid))
        self.x_max = float(np.max(x_grid))

        # Create regular grid coordinates
        self.step = (self.x_max - self.x_min) / (num_grid_points - 1)
        regular_x = np.linspace(self.x_min, self.x_max, num_grid_points)

        # Interpolate y values onto this regular grid
        # Sort input to ensure np.interp works correctly
        sort_idx = np.argsort(x_grid)
        x_in = x_grid[sort_idx]
        y_in = y_grid[sort_idx]

        regular_y = np.interp(regular_x, x_in, y_in)

        # Store Lookup Table as buffers so it moves with .to(device) and is saved
        # in the state dict, without becoming a trainable parameter.
        self.register_buffer('y_grid', torch.as_tensor(regular_y, dtype=torch.float32))
        self.register_buffer('x_start', torch.tensor(self.x_min, dtype=torch.float32))
        self.register_buffer('dx', torch.tensor(self.step, dtype=torch.float32))
        self.max_idx = float(num_grid_points - 1)
        self.max_idx_int = num_grid_points - 1

        self.lorentz_correction = lorentz_correction

        self.trainable_scale = trainable_scale
        if self.trainable_scale:
            # tfp.util.TransformedVariable(initial_value, bijector=Exp) is a
            # variable stored in log space and exponentiated on read, which keeps
            # the multiplier positive. nn.Parameter holding the log does the same.
            self.global_w_unconstrained = nn.Parameter(
                torch.tensor(float(np.log(initial_value)), dtype=torch.float32)
            )

    @property
    def global_w(self):
        return torch.exp(self.global_w_unconstrained)

    def forward(self, inputs):
        wavelengths = self.get_wavelength(inputs)

        float_idx = (wavelengths - self.x_start) / self.dx
        float_idx = torch.clamp(float_idx, 0.0, self.max_idx)

        idx_lo = torch.floor(float_idx)
        idx_hi = idx_lo + 1.0

        # interpolation weight
        weight = float_idx - idx_lo

        idx_lo_int = idx_lo.to(torch.int64)
        idx_hi_int = idx_hi.to(torch.int64)
        idx_hi_int = torch.minimum(idx_hi_int, torch.tensor(self.max_idx_int, device=idx_hi_int.device))
        y_lo = self.y_grid[idx_lo_int]
        y_hi = self.y_grid[idx_hi_int]
        scale = y_lo + weight * (y_hi - y_lo)

        if self.lorentz_correction:
            dinvsq = self.get_dHKL(inputs)
            # L = 4 * lambda^2 * d^2
            lorentz = 4.0 * torch.square(wavelengths) / (dinvsq + 1e-12)
            scale = scale * lorentz

        if self.trainable_scale:
            scale = scale * self.global_w

        # Force the output to be 1D (BatchSize,) instead of matching wavelengths (BatchSize, 1)
        scale = torch.reshape(scale, [-1])

        return Deterministic(loc=scale)
