import math
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.parameter import UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
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


class ImageLayer(LazyModuleMixin, nn.Module):
    """
    A linear layer whose weight matrix is indexed by image ID.
    Each image has its own weight matrix and bias.

    Uses LazyModuleMixin so that weights are allocated on the correct device
    on the first forward pass (after model.to(device) has been called),
    avoiding the fragile .to() reassignment pattern.
    """

    w: UninitializedParameter
    b: UninitializedParameter

    #: OPT-IN, and off by default on purpose. The fused Triton path in
    #: image_kernels is 8.7x faster on this layer at width 32, but end to end it
    #: is a 48 % *slowdown* at the production configuration (width 8,
    #: --num-batches=8) because it costs inductor a fusion it was making for
    #: free. See doc/performance/fused_image_layer.md for the full measurements
    #: and for what would have to change before this is worth enabling.
    #:
    #: Enable with CARELESS_FUSED_IMAGE_LAYER=1, or by setting this attribute.
    #: Tests set it directly to compare the two implementations.
    use_fused_kernel = os.environ.get("CARELESS_FUSED_IMAGE_LAYER", "0") == "1"

    def __init__(self, units, max_images, activation=None):
        super().__init__()
        self.units = units
        self.max_images = max_images
        self.activation = activation
        self.w = UninitializedParameter()
        self.b = UninitializedParameter()

    def initialize_parameters(self, inputs):
        if self.has_uninitialized_params():
            data, image_id = inputs
            in_features = data.shape[-1]
            with torch.no_grad():
                self.w.materialize((self.max_images, self.units, in_features))
                self.b.materialize((self.max_images, self.units))
                nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
                fan_in = in_features
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(self.b, -bound, bound)

    def forward(self, inputs):
        data, image_id = inputs
        image_id = image_id.squeeze(-1).long()

        # The Triton path computes the same affine map, but its backward is a
        # segmented reduction rather than a per-observation scatter-add. That is
        # worth ~8.7x on this layer at width 32; see image_kernels for why, and
        # for the profile that motivated it. Everything else -- CPU, non-float32,
        # uninitialized lazy parameters, no triton -- takes the reference path.
        if self.use_fused_kernel and not self.has_uninitialized_params():
            from careless.models.scaling import image_kernels

            if image_kernels.fast_path_available(data, self.w, self.b, image_id):
                result = torch.ops.careless.image_linear(data, self.w, self.b, image_id)
                if self.activation is not None:
                    result = self.activation(result)
                return result

        w = self.w[image_id]   # (batch, units, in_features)
        b = self.b[image_id]   # (batch, units)
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
                 leakiness=0.01, epsilon=1e-7, scale_bijector='exp', scale_multiplier=None):
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
