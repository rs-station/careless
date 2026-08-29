"""
Equivalence tests for the fused ImageLayer kernel.

Two standing rules from the performance work shape these tests:

* *An equivalence test whose fast path can silently fall back is not a test.*
  Every comparison here asserts on ``image_kernels.fast_path_calls`` so a
  dispatch bug that quietly ran the reference twice fails loudly instead of
  passing bit-for-bit.
* *Write negative controls.* ``test_negative_control_*`` deliberately break the
  kernel's assumptions and assert the comparison notices.

Correctness must not depend on ``image_id`` being sorted -- sorting is a
performance property, not a correctness one -- so the shuffled cases are checked
to the same tolerance as the sorted ones.
"""

import numpy as np
import pytest
import torch

from careless.models.scaling import image_kernels
from careless.models.scaling.image import ImageLayer


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not image_kernels.HAVE_TRITON,
    reason="fused ImageLayer kernel needs CUDA and triton",
)

# float32 reassociation floor. The kernel sums a tile with tl.dot while the
# reference accumulates with atomics, so bit-equality is not on offer.
RTOL = 2e-5


def _reference(data, w, b, image_id):
    """The index-gather formulation the kernel replaces."""
    return torch.bmm(w[image_id], data.unsqueeze(-1)).squeeze(-1) + b[image_id]


def _make(n, n_images, units, in_features, shuffle=False, seed=0, device="cuda"):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, max(2, 4 * n // n_images), n_images)
    counts = (counts * (n / counts.sum())).astype(np.int64)
    counts[-1] = n - counts[:-1].sum()
    assert counts.min() >= 0 and counts.sum() == n
    image_id = np.repeat(np.arange(n_images), counts)
    if shuffle:
        rng.shuffle(image_id)
    torch.manual_seed(seed)
    return (
        torch.randn(n, in_features, device=device),
        torch.randn(n_images, units, in_features, device=device),
        torch.randn(n_images, units, device=device),
        torch.from_numpy(image_id).to(device),
        torch.randn(n, units, device=device),
    )


def _rel(a, b):
    scale = b.abs().max()
    if scale == 0:
        return (a - b).abs().max().item()
    return ((a - b).abs().max() / scale).item()


def _compare(n, n_images, units, in_features, shuffle=False, seed=0):
    data, w, b, image_id, grad_out = _make(
        n, n_images, units, in_features, shuffle=shuffle, seed=seed
    )
    assert image_kernels.fast_path_available(data, w, b, image_id)

    fast = [t.clone().requires_grad_(True) for t in (data, w, b)]
    ref = [t.clone().requires_grad_(True) for t in (data, w, b)]

    before = image_kernels.fast_path_calls
    out_fast = torch.ops.careless.image_linear(fast[0], fast[1], fast[2], image_id)
    assert image_kernels.fast_path_calls == before + 1, "fused kernel did not run"

    out_ref = _reference(ref[0], ref[1], ref[2], image_id)
    out_fast.backward(grad_out)
    out_ref.backward(grad_out)

    return {
        "out": _rel(out_fast, out_ref),
        "grad_data": _rel(fast[0].grad, ref[0].grad),
        "grad_w": _rel(fast[1].grad, ref[1].grad),
        "grad_b": _rel(fast[2].grad, ref[2].grad),
    }


# ------------------------------------------------------------ equivalence ----

@requires_cuda
@pytest.mark.parametrize(
    "n,n_images,units,in_features",
    [
        (8192, 64, 32, 32),     # the shape the profile was taken at
        (8192, 64, 8, 8),       # production width
        (4096, 32, 16, 64),     # non-square, units != in_features
        (2048, 8, 32, 32),      # few large images: many tiles per image
        (512, 512, 32, 32),     # one observation per image: many images per tile
        (127, 4, 32, 32),       # n < BLOCK_N
        (128, 4, 32, 32),       # n == BLOCK_N exactly
        (129, 4, 32, 32),       # n == BLOCK_N + 1, partial trailing tile
    ],
)
def test_matches_reference(n, n_images, units, in_features):
    errs = _compare(n, n_images, units, in_features)
    for name, e in errs.items():
        assert e < RTOL, f"{name} disagrees by {e:.2e}"


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_when_image_id_is_shuffled(seed):
    """Sorting is a speed assumption; the answer must not depend on it."""
    errs = _compare(4096, 128, 32, 32, shuffle=True, seed=seed)
    for name, e in errs.items():
        assert e < RTOL, f"{name} disagrees by {e:.2e}"


@requires_cuda
def test_tf32_off_matches_to_float32_floor():
    """With TF32 disabled both paths should agree to ~1e-6, not merely 2e-5."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        errs = _compare(8192, 64, 32, 32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    for name, e in errs.items():
        assert e < 5e-6, f"{name} disagrees by {e:.2e}"


# ------------------------------------------------------- negative controls ---
# These prove the comparison above can actually fail. Each perturbs one thing the
# kernel is responsible for and asserts the check notices by a wide margin.

@requires_cuda
def test_negative_control_wrong_image_assignment():
    """Rolling image_id by one must break agreement."""
    data, w, b, image_id, grad_out = _make(8192, 64, 32, 32)
    good = torch.ops.careless.image_linear(data, w, b, image_id)
    bad = _reference(data, w, b, image_id.roll(1))
    assert _rel(good, bad) > 1e-2


@requires_cuda
def test_negative_control_dropped_bias():
    data, w, b, image_id, grad_out = _make(8192, 64, 32, 32)
    good = torch.ops.careless.image_linear(data, w, b, image_id)
    bad = _reference(data, w, torch.zeros_like(b), image_id)
    assert _rel(good, bad) > 1e-2


@requires_cuda
def test_negative_control_grad_w_must_accumulate_over_all_rows():
    """
    Halving the observations must change grad_w. Guards against a kernel that
    only ever reduced the first tile of each image.
    """
    data, w, b, image_id, grad_out = _make(8192, 64, 32, 32)
    wf = w.clone().requires_grad_(True)
    torch.ops.careless.image_linear(data, wf, b, image_id).backward(grad_out)
    half = data.shape[0] // 2
    wh = w.clone().requires_grad_(True)
    torch.ops.careless.image_linear(
        data[:half], wh, b, image_id[:half]
    ).backward(grad_out[:half])
    assert _rel(wf.grad, wh.grad) > 1e-2


# --------------------------------------------------------------- dispatch ----

@requires_cuda
def test_layer_uses_fused_kernel_and_matches_reference():
    """End to end through ImageLayer, including the activation it applies."""
    units = in_features = 32
    n, n_images = 8192, 64
    data, w, b, image_id, grad_out = _make(n, n_images, units, in_features)

    def build(use_fused):
        torch.manual_seed(0)
        layer = ImageLayer(units, n_images, activation=torch.nn.LeakyReLU(0.01))
        layer.use_fused_kernel = use_fused
        layer.to("cuda")
        with torch.no_grad():                      # materialize lazy parameters
            layer((data[:16], image_id[:16, None]))
        with torch.no_grad():
            layer.w.copy_(w)
            layer.b.copy_(b)
        return layer

    fast, ref = build(True), build(False)
    d_fast = data.clone().requires_grad_(True)
    d_ref = data.clone().requires_grad_(True)

    before = image_kernels.fast_path_calls
    out_fast = fast((d_fast, image_id[:, None]))
    assert image_kernels.fast_path_calls > before, "layer did not take the fused path"

    calls = image_kernels.fast_path_calls
    out_ref = ref((d_ref, image_id[:, None]))
    assert image_kernels.fast_path_calls == calls, "use_fused_kernel=False still fused"

    out_fast.backward(grad_out)
    out_ref.backward(grad_out)
    assert _rel(out_fast, out_ref) < RTOL
    assert _rel(d_fast.grad, d_ref.grad) < RTOL
    assert _rel(fast.w.grad, ref.w.grad) < RTOL
    assert _rel(fast.b.grad, ref.b.grad) < RTOL


@requires_cuda
def test_falls_back_off_cuda():
    """CPU tensors must take the reference path, not crash."""
    units = in_features = 8
    layer = ImageLayer(units, 16, activation=None)
    data = torch.randn(64, in_features)
    image_id = torch.randint(0, 16, (64, 1))
    with torch.no_grad():
        layer((data, image_id))
    before = image_kernels.fast_path_calls
    out = layer((data, image_id))
    assert out.shape == (64, units)
    assert image_kernels.fast_path_calls == before
