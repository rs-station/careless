import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import numpy as np
from careless.models.scaling.base import Scaler

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

        # Store Lookup Table as TF Constants
        self.y_grid = tf.constant(regular_y, dtype=tf.float32)
        self.x_start = tf.constant(self.x_min, dtype=tf.float32)
        self.dx = tf.constant(self.step, dtype=tf.float32)
        self.max_idx = tf.constant(num_grid_points - 1, dtype=tf.float32)
        self.max_idx_int = tf.constant(num_grid_points - 1, dtype=tf.int32)

        self.lorentz_correction = lorentz_correction

        self.trainable_scale = trainable_scale
        if self.trainable_scale:
            self.global_w = tfp.util.TransformedVariable(
                initial_value=initial_value,
                bijector=tfb.Exp(),
                dtype=tf.float32,
                name='spectral_global_scale'
            )

    def call(self, inputs):
        wavelengths = self.get_wavelength(inputs)

        float_idx = (wavelengths - self.x_start) / self.dx
        float_idx = tf.clip_by_value(float_idx, 0.0, self.max_idx)

        idx_lo = tf.floor(float_idx)
        idx_hi = idx_lo + 1.0

        # interpolation weight
        weight = float_idx - idx_lo

        idx_lo_int = tf.cast(idx_lo, tf.int32)
        idx_hi_int = tf.cast(idx_hi, tf.int32)
        idx_hi_int = tf.minimum(idx_hi_int, self.max_idx_int)
        y_lo = tf.gather(self.y_grid, idx_lo_int)
        y_hi = tf.gather(self.y_grid, idx_hi_int)
        scale = y_lo + weight * (y_hi - y_lo)

        if self.lorentz_correction:
            dinvsq = self.get_dHKL(inputs)
            # L = 4 * lambda^2 * d^2
            lorentz = 4.0 * tf.square(wavelengths) / (dinvsq + 1e-12)
            scale = scale * lorentz

        if self.trainable_scale:
            scale = scale * self.global_w

        # Force the output to be 1D (BatchSize,) instead of matching wavelengths (BatchSize, 1)
        scale = tf.reshape(scale, [-1])

        return tfd.Deterministic(loc=scale)
