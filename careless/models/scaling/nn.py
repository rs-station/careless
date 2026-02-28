import torch
import torch.nn as nn
from torch.distributions import Normal
from careless.models.scaling.base import Scaler


class NormalOutputLayer(nn.Module):
    """
    Converts a 2-channel linear output into a Normal distribution.
    The second channel (scale) is passed through softplus + epsilon shift.
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        loc, raw_scale = x.unbind(dim=-1)
        scale = torch.nn.functional.softplus(raw_scale) + self.epsilon
        return Normal(loc, scale)


class MetadataScaler(Scaler):
    """
    Neural-network scaler that maps reflection metadata → Normal scale distribution.
    """

    def __init__(self, n_layers, width, leakiness=0.01, epsilon=1e-7,
                 scale_bijector=None, scale_multiplier=None):
        """
        Parameters
        ----------
        n_layers : int
            Number of hidden MLP layers.
        width : int
            Width of each hidden layer.
        leakiness : float or None
            LeakyReLU negative slope; if None, use ReLU.
        epsilon : float
            Minimum scale value for numerical stability.
        scale_bijector : callable, optional
            Alternative activation for scale output. Ignored if None (uses softplus).
        scale_multiplier : float, optional
            Constant added to output location and scale for stability.
        """
        super().__init__()
        self.scale_multiplier = scale_multiplier

        mlp_layers = []
        in_features = None  # determined at first forward call via lazy init
        for i in range(n_layers):
            act = nn.LeakyReLU(leakiness) if leakiness is not None else nn.ReLU()
            if i == 0:
                mlp_layers.append(nn.LazyLinear(width))
            else:
                mlp_layers.append(nn.Linear(width, width))
            mlp_layers.append(act)

        self.network = nn.Sequential(*mlp_layers)

        # Output: 2 channels → (loc, scale)
        self.output_linear = nn.LazyLinear(2)
        self.epsilon = epsilon
        self._scale_bijector = scale_bijector  # e.g. 'exp' or 'softplus' string

    def _to_distribution(self, x):
        loc, raw_scale = x.unbind(dim=-1)
        if self._scale_bijector == 'exp':
            scale = torch.exp(raw_scale) + self.epsilon
        else:  # default: softplus
            scale = torch.nn.functional.softplus(raw_scale) + self.epsilon
        if self.scale_multiplier is not None:
            loc = loc + self.scale_multiplier
            scale = scale + self.scale_multiplier
        return Normal(loc, scale)

    def call(self, metadata):
        """
        Parameters
        ----------
        metadata : Tensor (float32), shape (n_obs, n_features)

        Returns
        -------
        dist : torch.distributions.Normal
        """
        h = self.network(metadata.float())
        out = self.output_linear(h)
        return self._to_distribution(out)

    def forward(self, metadata):
        return self.call(metadata)


class MLPScaler(MetadataScaler):
    """MLPScaler that extracts metadata from careless input tuples."""

    def forward(self, inputs):
        metadata = self.get_metadata(inputs)
        return self.call(metadata)
