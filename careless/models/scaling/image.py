import torch
import torch.nn as nn
from torch.distributions import Normal
from careless.models.scaling.base import Scaler


class ImageScaler(Scaler):
    """
    Simple per-image scale factors. The first image is pegged to 1 (reference);
    all others are freely learned.
    """

    def __init__(self, max_images):
        """
        Parameters
        ----------
        max_images : int
            Number of images.
        """
        super().__init__()
        # max_images - 1 free parameters; image 0 is always 1
        self._scales = nn.Parameter(torch.ones(max_images - 1))

    @property
    def scales(self):
        return torch.cat([torch.ones(1, device=self._scales.device), self._scales], dim=0)

    def forward(self, inputs):
        image_ids = self.get_image_id(inputs).squeeze(-1).long()
        return self.scales[image_ids]


class HybridImageScaler(Scaler):
    """
    Combines an MLPScaler (returns Normal distribution) with per-image scalar scales.
    The image scale multiplies both loc and scale of the MLP-predicted Normal.
    """

    def __init__(self, mlp_scaler, image_scaler):
        super().__init__()
        self.mlp_scaler = mlp_scaler
        self.image_scaler = image_scaler

    def forward(self, inputs):
        q = self.mlp_scaler(inputs)          # Normal(loc, scale)
        a = self.image_scaler(inputs)        # scalar per observation
        # Scale the distribution: Normal(a*loc, a*scale)
        return Normal(a * q.loc, a * q.scale)


class ImageLayer(nn.Module):
    """
    A linear layer whose weight matrix is indexed by image ID.
    Each image has its own weight matrix and bias.
    """

    def __init__(self, units, max_images, activation=None):
        super().__init__()
        self.units = units
        self.max_images = max_images
        self.activation = activation

        # weights: (max_images, units, in_features) — lazy initialisation
        # We use LazyLinear-style approach: create weights at first call
        self._initialized = False
        self._units = units
        self._max_images = max_images

    def _init_weights(self, in_features):
        self.w = nn.Parameter(
            torch.eye(self._units, in_features).unsqueeze(0)
            .expand(self._max_images, -1, -1).clone()
        )
        self.b = nn.Parameter(torch.zeros(self._max_images, self._units))
        self._initialized = True

    def forward(self, metadata_and_image_id):
        data, image_id = metadata_and_image_id
        image_id = image_id.squeeze(-1).long()

        if not self._initialized:
            self._init_weights(data.shape[-1])
            self.w = self.w.to(data.device)
            self.b = self.b.to(data.device)

        # Gather per-image weights: (batch, units, in_features)
        w = self.w[image_id]   # (batch, units, in_features)
        b = self.b[image_id]   # (batch, units)
        # Matrix multiply: (batch, units, in_features) @ (batch, in_features, 1) → (batch, units)
        result = torch.bmm(w, data.unsqueeze(-1)).squeeze(-1) + b
        if self.activation is not None:
            result = self.activation(result)
        return result


class NeuralImageScaler(Scaler):
    """
    Per-image neural network scaler: a stack of per-image linear layers followed by
    an MLP distribution head.
    """

    def __init__(self, image_layers, max_images, mlp_layers, mlp_width,
                 leakiness=0.01, epsilon=1e-7, scale_bijector=None, scale_multiplier=None):
        super().__init__()
        activation = nn.LeakyReLU(leakiness) if leakiness is not None else nn.ReLU()

        self.image_layer_list = nn.ModuleList([
            ImageLayer(mlp_width, max_images, activation)
            for _ in range(image_layers)
        ])

        from careless.models.scaling.nn import MetadataScaler
        self.metadata_scaler = MetadataScaler(
            mlp_layers, mlp_width, leakiness, epsilon=epsilon,
            scale_bijector=scale_bijector, scale_multiplier=scale_multiplier
        )

    def forward(self, inputs):
        result = self.get_metadata(inputs).float()
        image_id = self.get_image_id(inputs)

        # Pass through the MLP network layers first
        result = self.metadata_scaler.network(result)

        # Then through per-image layers
        for layer in self.image_layer_list:
            result = layer((result, image_id))

        # Finally through the distribution output layer
        out = self.metadata_scaler.output_linear(result)
        return self.metadata_scaler._to_distribution(out)
