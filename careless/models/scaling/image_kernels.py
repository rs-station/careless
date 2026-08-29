"""
Triton kernels for :class:`careless.models.scaling.image.ImageLayer`.

WHY THIS EXISTS
---------------
``ImageLayer`` is a linear layer whose weight matrix is selected per image::

    out[n, i] = sum_j w[image_id[n], i, j] * data[n, j] + b[image_id[n], i]

Written that way in PyTorch the forward is cheap -- inductor fuses ``w[image_id]``
away and never materialises the per-observation copies. The *backward* is not. The
gradient of a gather is a scatter-add, so ``grad_w`` is accumulated with one atomic
add per (observation, i, j) triple. An Nsight Compute profile of a production shape
(4.1 M reflections, 13,482 images, width 32, 32 gradient-accumulation batches)
found that this one kernel family was **60 % of all GPU time**, that
``RED.E.ADD.F32`` was 84 % of its global memory traffic with **74 % of those
sectors wasted**, and that it ran at 3 % of DRAM and 7 % of SM throughput with
16.7 % occupancy -- 45x off the memory roofline. It was not bandwidth- or
compute-bound; it was standing in a queue.

The structural fact the scatter throws away is that **observations are sorted by
image**: the dataset forms exactly one contiguous run per image, and careless cuts
gradient-accumulation batches as contiguous slices, so the runs survive batching.
That makes ``grad_w`` a *segmented reduction*, not a scatter -- 132 M atomic adds
doing the work of 432 K coalesced stores.

WHAT THIS DOES INSTEAD
----------------------
Tile the observation axis into fixed ``BLOCK_N``-row blocks. Because rows are
sorted, a tile spans only a handful of images (median run length is 265
observations). Each program loops over the images present in its tile, accumulates
that image's whole ``(units, in_features)`` block in registers with one ``tl.dot``,
and issues a single coalesced ``atomic_add`` of the finished block. Atomic traffic
falls by about two orders of magnitude and every add is contiguous in ``j``.

SORTING IS A PERFORMANCE ASSUMPTION, NOT A CORRECTNESS ONE
----------------------------------------------------------
The per-image loop masks rows by image id, so **any ordering gives the right
answer**; sorting is only what keeps the loop short. Unsorted input would make a
tile span many images and run slowly, but never wrongly. There is deliberately no
data-dependent guard here: checking sortedness would cost a host synchronisation on
every call, and careless' formatters already emit image-sorted data. See
``test_image_kernels.py``, which checks correctness on shuffled input.

The layer's activation stays outside these ops, so ``grad_output`` arriving here is
already multiplied by the activation derivative. That keeps the kernels independent
of which activation the layer was built with.

INDEXING WIDTH
--------------
All offsets are computed in int64. careless has already been bitten once by an
int32 gather overflow; at 463 M reflections ``row * in_features`` exceeds int32,
and at 94.5 k images ``image * units * in_features`` does too at width 128.
"""

import torch

try:  # triton ships with CUDA torch builds, but do not hard-depend on it
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - only on triton-less installs
    HAVE_TRITON = False


__all__ = ["image_linear", "fast_path_available", "BLOCK_N"]


# Rows per program. 128 keeps the (BLOCK_N, units) tiles in registers at the widths
# careless uses while still covering a median image run in one or two tiles.
BLOCK_N = 128


if HAVE_TRITON:

    @triton.jit
    def _image_linear_fwd_kernel(
        data_ptr, w_ptr, b_ptr, image_id_ptr, out_ptr,
        n_obs,
        BLOCK_N: tl.constexpr, BLOCK_U: tl.constexpr, BLOCK_I: tl.constexpr,
        UNITS: tl.constexpr, IN_FEATURES: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        row_mask = rows < n_obs

        u = tl.arange(0, BLOCK_U)
        i = tl.arange(0, BLOCK_I)
        u_mask = u < UNITS
        i_mask = i < IN_FEATURES

        img = tl.load(image_id_ptr + rows, mask=row_mask, other=0).to(tl.int64)

        d = tl.load(
            data_ptr + rows[:, None] * IN_FEATURES + i[None, :],
            mask=row_mask[:, None] & i_mask[None, :],
            other=0.0,
        )

        # Images present in this tile. Padding lanes are pushed out of the range
        # so they cannot widen the loop.
        g_lo = tl.min(tl.where(row_mask, img, 0x7FFFFFFF))
        g_hi = tl.max(tl.where(row_mask, img, 0))

        acc = tl.zeros([BLOCK_N, BLOCK_U], dtype=tl.float32)

        g = g_lo
        while g <= g_hi:
            member = row_mask & (img == g)
            w_base = g * (UNITS * IN_FEATURES)
            wg = tl.load(
                w_ptr + w_base + u[:, None] * IN_FEATURES + i[None, :],
                mask=u_mask[:, None] & i_mask[None, :],
                other=0.0,
            )
            bg = tl.load(b_ptr + g * UNITS + u, mask=u_mask, other=0.0)
            part = tl.dot(d, tl.trans(wg), input_precision=PRECISION)
            acc += tl.where(member[:, None], part + bg[None, :], 0.0)
            g += 1

        tl.store(
            out_ptr + rows[:, None] * UNITS + u[None, :],
            acc,
            mask=row_mask[:, None] & u_mask[None, :],
        )

    @triton.jit
    def _image_linear_bwd_kernel(
        grad_out_ptr, data_ptr, w_ptr, image_id_ptr,
        grad_data_ptr, grad_w_ptr, grad_b_ptr,
        n_obs,
        BLOCK_N: tl.constexpr, BLOCK_U: tl.constexpr, BLOCK_I: tl.constexpr,
        UNITS: tl.constexpr, IN_FEATURES: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        row_mask = rows < n_obs

        u = tl.arange(0, BLOCK_U)
        i = tl.arange(0, BLOCK_I)
        u_mask = u < UNITS
        i_mask = i < IN_FEATURES

        img = tl.load(image_id_ptr + rows, mask=row_mask, other=0).to(tl.int64)

        go = tl.load(
            grad_out_ptr + rows[:, None] * UNITS + u[None, :],
            mask=row_mask[:, None] & u_mask[None, :],
            other=0.0,
        )
        d = tl.load(
            data_ptr + rows[:, None] * IN_FEATURES + i[None, :],
            mask=row_mask[:, None] & i_mask[None, :],
            other=0.0,
        )

        g_lo = tl.min(tl.where(row_mask, img, 0x7FFFFFFF))
        g_hi = tl.max(tl.where(row_mask, img, 0))

        grad_d = tl.zeros([BLOCK_N, BLOCK_I], dtype=tl.float32)

        g = g_lo
        while g <= g_hi:
            member = row_mask & (img == g)
            go_m = tl.where(member[:, None], go, 0.0)
            w_base = g * (UNITS * IN_FEATURES)

            # grad_w[g] += go_m.T @ d -- the whole point: one coalesced atomic per
            # (i, j) for the entire tile, instead of one per (row, i, j).
            gw = tl.dot(tl.trans(go_m), d, input_precision=PRECISION)
            tl.atomic_add(
                grad_w_ptr + w_base + u[:, None] * IN_FEATURES + i[None, :],
                gw,
                mask=u_mask[:, None] & i_mask[None, :],
                sem="relaxed",
            )

            gb = tl.sum(go_m, axis=0)
            tl.atomic_add(grad_b_ptr + g * UNITS + u, gb, mask=u_mask, sem="relaxed")

            wg = tl.load(
                w_ptr + w_base + u[:, None] * IN_FEATURES + i[None, :],
                mask=u_mask[:, None] & i_mask[None, :],
                other=0.0,
            )
            grad_d += tl.dot(go_m, wg, input_precision=PRECISION)
            g += 1

        tl.store(
            grad_data_ptr + rows[:, None] * IN_FEATURES + i[None, :],
            grad_d,
            mask=row_mask[:, None] & i_mask[None, :],
        )


def _dot_dim(x):
    """tl.dot needs every dimension to be a power of two and at least 16."""
    n = 16
    while n < x:
        n *= 2
    return n


def _precision():
    """Match whatever torch is doing, so the kernel and the reference agree."""
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


#: Number of times the fused forward has run. Tests use this to prove the fast
#: path actually executed -- an equivalence test whose fast path can silently fall
#: back to the reference is not a test, it is two runs of the reference.
fast_path_calls = 0


def fast_path_available(data, w, b, image_id):
    """
    True when the Triton path can run for these tensors.

    Looks only at devices, dtypes and shapes -- never at values -- so it costs no
    host synchronisation. Non-contiguous inputs are not rejected; the op makes
    them contiguous, which is far cheaper than the scatter it avoids.
    """
    if not HAVE_TRITON:
        return False
    if not (data.is_cuda and w.is_cuda and b.is_cuda and image_id.is_cuda):
        return False
    if data.dtype != torch.float32 or w.dtype != torch.float32 or b.dtype != torch.float32:
        return False
    if data.dim() != 2 or w.dim() != 3 or b.dim() != 2:
        return False
    return True


# --------------------------------------------------------------------- ops ---
# Registered as custom operators rather than a torch.autograd.Function so that
# torch.compile keeps them in the graph: a Function containing a Triton launch
# forces a graph break, a custom op with a fake (meta) implementation does not.

@torch.library.custom_op("careless::image_linear", mutates_args=())
def image_linear(
    data: torch.Tensor, w: torch.Tensor, b: torch.Tensor, image_id: torch.Tensor
) -> torch.Tensor:
    """
    out[n, i] = sum_j w[image_id[n], i, j] * data[n, j] + b[image_id[n], i]

    The layer's activation is applied by the caller, not here.
    """
    global fast_path_calls
    fast_path_calls += 1

    n_obs, in_features = data.shape
    _, units, w_in = w.shape
    assert w_in == in_features, (w.shape, data.shape)

    data = data.contiguous()
    out = torch.empty((n_obs, units), dtype=data.dtype, device=data.device)
    if n_obs == 0:
        return out

    _image_linear_fwd_kernel[(triton.cdiv(n_obs, BLOCK_N),)](
        data, w, b, image_id.contiguous(), out,
        n_obs,
        BLOCK_N=BLOCK_N,
        BLOCK_U=_dot_dim(units), BLOCK_I=_dot_dim(in_features),
        UNITS=units, IN_FEATURES=in_features,
        PRECISION=_precision(),
    )
    return out


@image_linear.register_fake
def _image_linear_fake(data, w, b, image_id):
    return data.new_empty((data.shape[0], w.shape[1]))


@torch.library.custom_op("careless::image_linear_backward", mutates_args=())
def image_linear_backward(
    grad_out: torch.Tensor, data: torch.Tensor, w: torch.Tensor, image_id: torch.Tensor
) -> list[torch.Tensor]:
    n_obs, in_features = data.shape
    n_images, units, _ = w.shape

    data = data.contiguous()
    grad_data = torch.empty_like(data)
    # grad_w and grad_b are accumulated into, so they must start zeroed.
    grad_w = torch.zeros_like(w)
    grad_b = torch.zeros((n_images, units), dtype=w.dtype, device=w.device)
    if n_obs == 0:
        return [grad_data, grad_w, grad_b]

    _image_linear_bwd_kernel[(triton.cdiv(n_obs, BLOCK_N),)](
        grad_out.contiguous(), data, w, image_id.contiguous(),
        grad_data, grad_w, grad_b,
        n_obs,
        BLOCK_N=BLOCK_N,
        BLOCK_U=_dot_dim(units), BLOCK_I=_dot_dim(in_features),
        UNITS=units, IN_FEATURES=in_features,
        PRECISION=_precision(),
    )
    return [grad_data, grad_w, grad_b]


@image_linear_backward.register_fake
def _image_linear_backward_fake(grad_out, data, w, image_id):
    return [
        torch.empty_like(data),
        torch.empty_like(w),
        w.new_empty((w.shape[0], w.shape[1])),
    ]


def _setup_context(ctx, inputs, output):
    data, w, b, image_id = inputs
    ctx.save_for_backward(data, w, image_id)


def _backward(ctx, grad_out):
    data, w, image_id = ctx.saved_tensors
    grad_data, grad_w, grad_b = torch.ops.careless.image_linear_backward(
        grad_out, data, w, image_id
    )
    return grad_data, grad_w, grad_b, None


torch.library.register_autograd(
    "careless::image_linear", _backward, setup_context=_setup_context
)
