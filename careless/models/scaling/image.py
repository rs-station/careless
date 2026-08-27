import math
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


class ImageMajorPlan:
    """
    A fixed map between row-major reflection order and an image-major padded layout.

    The padded layout reserves `slots` rows for every image in every chunk. Chunk k
    holds reflections [k*slots, (k+1)*slots) of each image that has that many; images
    are ordered by descending reflection count, so the images still contributing in
    chunk k are always a prefix and the weight tensor can be sliced rather than
    gathered. Slots past the end of an image are padding: they are pointed at a real
    row of the same image so the arithmetic stays finite, and nothing ever reads them
    back, so they receive exactly zero gradient.

    Attributes
    ----------
    slots : int
        Reflections reserved per image per chunk (R).
    alive : list of int
        Number of images contributing to each chunk.
    offsets : list of int
        Start of each chunk in the flattened padded array.
    pad_index : Tensor (int64), shape (n_padded,)
        Padded slot -> row index.
    inv : Tensor (int64), shape (n_rows,)
        Row index -> padded slot. Only ever points at non-padding slots.
    img_order : Tensor (int64), shape (max_images,)
        Images sorted by descending reflection count.
    """

    __slots__ = ("slots", "alive", "offsets", "pad_index", "inv", "img_order",
                 "n_rows", "n_padded", "_keep")

    def __init__(self, slots, alive, offsets, pad_index, inv, img_order, keep):
        self.slots = slots
        self.alive = alive
        self.offsets = offsets
        self.pad_index = pad_index
        self.inv = inv
        self.img_order = img_order
        self.n_rows = int(inv.numel())
        self.n_padded = int(pad_index.numel())
        # Hold a reference to the image_id tensor this plan was built from. The plan
        # cache is keyed by that tensor's address, and keeping it alive stops the
        # allocator handing the same address to a different tensor.
        self._keep = keep

    @property
    def padding_fraction(self):
        return self.n_padded / max(self.n_rows, 1) - 1.0


def choose_slots(counts, candidates=(32, 64, 128, 256, 512), max_padding=0.25):
    """
    Largest candidate slot count whose padding overhead stays within `max_padding`.

    Bigger slot counts mean fewer, fatter batched matmuls but more wasted slots on
    small images; the cap keeps the waste bounded while preferring fewer launches.
    """
    n_rows = int(counts.sum())
    if n_rows == 0:
        return candidates[0]
    best = candidates[0]
    for slots in candidates:
        padded = int((((counts + slots - 1) // slots) * slots).sum())
        if padded / n_rows - 1.0 <= max_padding:
            best = slots
    return best


def build_image_major_plan(image_id, max_images, slots=None, max_padding=0.25):
    """
    Build an ImageMajorPlan for one set of reflections.

    Parameters
    ----------
    image_id : Tensor
        Image index per reflection, any shape broadcastable to (n_rows,).
    max_images : int
        Number of images the weight tensors are sized for.
    slots : int or None
        Reflections per image per chunk. Chosen from the count distribution if None.
    max_padding : float
        Padding budget used when choosing `slots` automatically.
    """
    gid = image_id.reshape(-1).long()
    device = gid.device
    n_rows = gid.numel()
    counts = torch.bincount(gid, minlength=max_images)
    if slots is None:
        slots = choose_slots(counts, max_padding=max_padding)

    img_order = torch.argsort(counts, descending=True, stable=True)
    sorted_counts = counts[img_order]
    largest = int(sorted_counts[0]) if n_rows else 0
    n_chunks = (largest + slots - 1) // slots

    # Reflections grouped by image, images in descending-count order.
    rank_of_image = torch.empty_like(img_order)
    rank_of_image[img_order] = torch.arange(max_images, device=device)
    gid_ranked = rank_of_image[gid]
    row_order = torch.argsort(gid_ranked, stable=True)
    rank = torch.empty(n_rows, dtype=torch.long, device=device)
    rank[row_order] = torch.arange(n_rows, device=device)

    zero = torch.zeros(1, dtype=torch.long, device=device)
    image_offset = torch.cat([zero, sorted_counts.cumsum(0)[:-1]])
    local = rank - image_offset[gid_ranked]          # position within its image

    alive, offsets, pads, running = [], [], [], 0
    within = torch.arange(slots, device=device)
    for k in range(n_chunks):
        a = int((sorted_counts > k * slots).sum())
        base = image_offset[:a, None] + k * slots + within[None, :]
        cap = (image_offset[:a] + sorted_counts[:a] - 1)[:, None]
        pads.append(row_order[torch.minimum(base, cap)].reshape(-1))
        alive.append(a)
        offsets.append(running)
        running += a * slots

    pad_index = torch.cat(pads) if pads else torch.empty(0, dtype=torch.long, device=device)
    offset_t = torch.as_tensor(offsets, dtype=torch.long, device=device)
    chunk_of_row = local // slots
    inv = offset_t[chunk_of_row] + gid_ranked * slots + (local % slots)

    return ImageMajorPlan(slots, alive, offsets, pad_index, inv, img_order, image_id)


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
        """
        Row-major path: one weight matrix gathered per reflection.

        Kept for callers that have no plan. This materializes an
        (n_obs, units, in_features) tensor and is quadratic in width; prefer
        `apply_chunks` with an ImageMajorPlan.
        """
        data, image_id = inputs
        image_id = image_id.squeeze(-1).long()
        w = self.w[image_id]   # (batch, units, in_features)
        b = self.b[image_id]   # (batch, units)
        result = torch.bmm(w, data.unsqueeze(-1)).squeeze(-1) + b
        if self.activation is not None:
            result = self.activation(result)
        return result

    def apply_chunks(self, chunks, plan):
        """
        Image-major path: no per-reflection weight gather at all.

        Each chunk is (n_live_images, slots, in_features) and its batch dimension is
        the image itself, so the layer is a batched matmul against a *slice* of the
        weight tensor. Because an image appears at most once per chunk the slice has
        no repeated indices, and the weight gradient comes back as an ordinary batched
        GEMM instead of a sorted scatter-add.

        Parameters
        ----------
        chunks : list of Tensor
            One (n_live, slots, in_features) tensor per chunk.
        plan : ImageMajorPlan

        Returns
        -------
        list of Tensor
            One (n_live, slots, units) tensor per chunk.
        """
        w = self.w[plan.img_order]
        b = self.b[plan.img_order]
        out = []
        for chunk, live in zip(chunks, plan.alive):
            result = torch.baddbmm(
                b[:live].unsqueeze(1), chunk, w[:live].transpose(1, 2)
            )
            if self.activation is not None:
                result = self.activation(result)
            out.append(result)
        return out


class NeuralImageScaler(Scaler):
    """
    Per-image neural network scaler: a stack of per-image linear layers followed by
    an MLP distribution head.
    """

    def __init__(self, image_layers, max_images, mlp_layers, mlp_width,
                 leakiness=0.01, epsilon=1e-7, scale_bijector='exp', scale_multiplier=None,
                 image_major=True, slots=None, max_padding=0.25):
        """
        Parameters
        ----------
        image_major : bool
            Run the per-image layers in the padded image-major layout, which removes
            the per-reflection weight gather. Set False for the original row-major
            path.
        slots : int or None
            Reflections reserved per image per chunk. Chosen from the reflection count
            distribution when None.
        max_padding : float
            Padding budget used when choosing `slots` automatically.
        """
        super().__init__()
        activation = nn.LeakyReLU(leakiness) if leakiness is not None else nn.ReLU()
        self.max_images = max_images
        self.image_major = image_major
        self.slots = slots
        self.max_padding = max_padding
        # One plan per distinct set of reflections. Gradient accumulation presents one
        # per batch, plus validation and prediction sets, so keep room for plenty --
        # a miss costs an argsort over the batch, on every step.
        self._plan_cache = {}
        self._plan_cache_size = 256
        self.plan_misses = 0


        self.image_layer_list = nn.ModuleList([
            ImageLayer(mlp_width, max_images, activation)
            for _ in range(image_layers)
        ])

        from careless.models.scaling.nn import MetadataScaler
        self.metadata_scaler = MetadataScaler(
            mlp_layers, mlp_width, leakiness, epsilon=epsilon,
            scale_bijector=scale_bijector, scale_multiplier=scale_multiplier
        )

    def has_uninitialized_image_layers(self):
        return any(l.has_uninitialized_params() for l in self.image_layer_list)

    def get_plan(self, image_id):
        """
        Fetch or build the image-major plan for this set of reflections.

        Building a plan costs an argsort over the reflections, so it is cached. The
        cache is keyed by the image_id tensor's address and length, and each entry
        holds a reference to that tensor, which prevents the allocator from reusing
        the address for a different tensor while the key is live.
        """
        key = (image_id.data_ptr(), image_id.numel(), image_id.device)
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = build_image_major_plan(
                image_id, self.max_images, slots=self.slots,
                max_padding=self.max_padding,
            )
            if len(self._plan_cache) >= self._plan_cache_size:
                self._plan_cache.pop(next(iter(self._plan_cache)))
            self.plan_misses += 1
            self._plan_cache[key] = plan
        return plan

    def forward(self, inputs):
        result = self.get_metadata(inputs).float()
        image_id = self.get_image_id(inputs)

        # Pass through the MLP network layers first, in reflection order.
        result = self.metadata_scaler.network(result)

        if self.image_layer_list:
            if self.image_major and self.has_uninitialized_image_layers():
                # apply_chunks bypasses __call__, so the LazyModuleMixin hook that
                # materializes w and b would never fire. Push a single row through the
                # row-major path once to trigger it.
                with torch.no_grad():
                    probe = result[:1]
                    probe_id = image_id.reshape(-1, 1)[:1]
                    for layer in self.image_layer_list:
                        probe = layer((probe, probe_id))

            if self.image_major and image_id.numel() > 0:
                # Enter the padded image-major layout, run the per-image layers there,
                # and leave it again. The scaling model is a pure row-wise map, so
                # permuting rows in and out is transparent to everything downstream:
                # the returned distribution is in the original reflection order and
                # padded slots are never read back. Nothing outside this block --
                # refl_id, harmonic_id, the Laue convolution -- sees the padded layout.
                plan = self.get_plan(image_id)
                padded = result[plan.pad_index]
                chunks = [
                    padded[off:off + live * plan.slots].view(live, plan.slots, -1)
                    for off, live in zip(plan.offsets, plan.alive)
                ]
                for layer in self.image_layer_list:
                    chunks = layer.apply_chunks(chunks, plan)
                # Apply the output head while still padded: it is row-wise too, and
                # narrowing to 2 channels first makes the return trip cheaper.
                width = chunks[0].shape[-1] if chunks else 0
                flat = torch.cat([c.reshape(-1, width) for c in chunks], dim=0)
                out = self.metadata_scaler.output_linear(flat)[plan.inv]
                return self.metadata_scaler._to_distribution(out)

            for layer in self.image_layer_list:
                result = layer((result, image_id))

        # Finally through the distribution output layer
        out = self.metadata_scaler.output_linear(result)
        return self.metadata_scaler._to_distribution(out)
