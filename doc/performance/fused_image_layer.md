# A fused ImageLayer kernel: what it does, and why it was not merged

**Status: parked, do not merge.** The kernel is correct, tested, and 8.7x faster
at the *layer* level at width 32. But on the production dataset at the production
configuration it makes the step **48 % slower** and costs 800 MiB more memory
(see 5). It is kept on this branch for the profile, the measurements and the dead
ends, so nobody repeats the investigation from scratch.

Branch: `perf/fused-image-layer-kernel`, off `perf/jit-gradient-accumulation`.

**The kernel is off by default.** `ImageLayer.use_fused_kernel` is False unless
`CARELESS_FUSED_IMAGE_LAYER=1` is set, so this branch is inert as merged and
cannot regress anything by accident. Every A/B below was taken by flipping that
flag; the code path is identical either way.

---

## 1. The problem it attacks

`ImageLayer` is a linear layer whose weights are selected per image:

```python
w = self.w[image_id]                                   # (batch, units, in_features)
result = torch.bmm(w, data.unsqueeze(-1)).squeeze(-1) + self.b[image_id]
```

The forward is cheap -- inductor fuses the gather away and never materialises the
per-observation copies. The backward is not: the gradient of a gather is a
scatter-add, so `grad_w` is accumulated with **one atomic add per (observation, i,
j) triple**.

An Nsight Compute profile at width 32, `--num-batches=32`, 4.1 M reflections /
13,482 images (`examples/window_merge_no_preprocess/bench/nsys_w32nb32/`) found:

| | |
|---|---|
| share of all GPU time | **60 %**, in 64 launches/step |
| `RED.E.ADD.F32` share of global sectors | 84 %, of which **74 % wasted** |
| DRAM throughput | 3.3 % |
| Compute (SM) throughput | 7.6 % |
| achieved occupancy | 16.4 %, register-capped at 255 regs/thread |
| distance from memory roofline | **45x** |

Not bandwidth-bound, not compute-bound, not contended on atomics -- **latency
bound on the number and shape of the atomic messages.** Each thread owned four
consecutive `j`, so the lane stride was 16 B where 4 B would coalesce.

## 2. The structural fact

Observations are **sorted by image**: the dataset forms exactly one contiguous run
per image (13,482 runs for 13,482 images, zero descents), and `_batch_boundaries`
cuts contiguous slices, so runs survive gradient-accumulation batching. Mean run
length is ~305 observations.

That makes `grad_w` a **segmented reduction**, not a scatter: 131,893,248 atomic
adds doing the work of 432,416 coalesced stores -- **305x more write traffic than
the operation needs**.

## 3. What the kernel does

Tile the observation axis into fixed 128-row blocks. Because rows are sorted, a
tile spans only a few images. Each program loops over the images present in its
tile, accumulates that image's whole `(units, in_features)` block in registers
with one `tl.dot`, and issues a single coalesced `atomic_add`.

Registered with `torch.library.custom_op` plus a fake (meta) implementation rather
than a `torch.autograd.Function`, so `torch.compile` keeps it in the graph instead
of breaking on it.

Correctness does not depend on the sorting -- the loop masks by image id, so any
ordering is right; sorting only keeps the loop short.

## 4. Correctness

`tests/models/scaling/test_image_kernels.py`, 17 tests, all passing; full suite
1966 passed.

* Matches the index-gather reference to 1e-7..1e-6 relative across width 8/16/32/64,
  non-square `units != in_features`, one-observation-per-image, few-huge-images,
  and `n` at 127/128/129 (the `BLOCK_N` boundary).
* Matches on **shuffled** `image_id`, to the same tolerance.
* With TF32 disabled, agrees to <5e-6.
* Three negative controls (rolled `image_id`, dropped bias, halved observation
  set) confirm the comparison can fail.
* Every equivalence assertion checks `image_kernels.fast_path_calls` first, so a
  dispatch bug that quietly ran the reference twice fails instead of passing
  bit-for-bit.
* End to end the final loss is **bit-identical** with and without the kernel
  (17.736940383911133 at width 32; 17.744365692138672 at width 8).

## 5. Why it was parked

**The layer gets much faster. The step barely does.**

Isolated, two stacked image layers, forward + backward, compiled, real batch
(n=128,802, 427 images):

| width | reference | fused | |
|---:|---:|---:|---:|
| 32 | 6.83 ms | 0.78 ms | **8.73x** |
| 8 | 0.40 ms | 0.42 ms | 0.96x |

End to end, small dataset, `--num-batches=32`, 40 iterations, steady-state median:

| width | reference | fused | speedup | peak MiB (ref -> fused) |
|---:|---:|---:|---:|---|
| 8 | 55.5 ms | 80.5 ms | **0.69x** | 417 -> 441 |
| 16 | 101.8 ms | 122.4 ms | **0.83x** | 582 -> 627 |
| 32 | 318.5 ms | 261.2 ms | **1.22x** | 1099 -> 1187 |

**The gap between 8.73x on the layer and 1.22x on the step is lost fusion.** The
kernel being replaced is named
`triton_per_fused_bmm_index_index_put_leaky_relu_leaky_relu_backward_new_zeros_...`
-- inductor was welding `ImageLayer`'s own activation into the same kernel. This
implementation deliberately leaves the activation outside the custom op, to stay
independent of which activation the layer was built with, and an opaque custom op
cannot be fused into. At width 32 the kernel win covers that cost; at widths 8 and
16 there is no win left to cover it, so the layer optimisation shows up as a net
**slowdown**.

### The ceiling, by Amdahl

At width 32 the image layers are 60 % (backward) + 3.2 % (forward) of GPU time and
the step is ~100 % GPU-bound:

| | step speedup |
|---|---:|
| layers become free (unreachable wall) | 2.72x |
| layers 8.73x faster, fusion preserved | 2.27x |
| **measured** | **1.22x** |

So the whole optimisation is worth *at most* ~2x, and roughly 1.9x of the
available 2.27x is currently being handed back to the lost fusion.

### The decisive measurement: production config, production dataset

`examples/data/big_friedel_{plus,minus}.mtz` (~28.9 M reflections, ~94.5 k images),
`--mlp-width=8 --num-batches=8` -- exactly what `examples/merge_big.sh` runs. Two
replicates per arm, interleaved:

| run | ms/step | peak MiB | final loss |
|---|---:|---:|---:|
| reference | 280.0 | 3842 | 19.379580 |
| fused | 413.0 | 4544 | 19.379580 |
| reference (replicate) | 276.7 | 3843 | 19.379580 |
| fused (replicate) | 410.4 | 4654 | 19.379580 |

**278.4 -> 411.7 ms/step: 0.68x, a 48 % slowdown, at +800 MiB.** Replicate spread
on the reference arm is 3.3 ms, so this is not noise. The loss is bit-identical
across all four runs, so it is purely a performance verdict.

**A hypothesis that turned out to be wrong, recorded so it is not re-tried.** The
expectation going into this run was that scale would rescue the kernel: at 13,482
images the live `grad_w` region is 1.65 MiB and sits in L2 (97.4 % hit rate), which
flatters the scatter, whereas at 94.5 k images it should not fit and the scatter
should degrade. It does not play out, for a simple reason -- **at width 8 the kernel
is not faster than the scatter in isolation either** (0.96x). There was never a
layer-level win at that width for scale to amplify; all scale did was enlarge the
fusion loss and the memory cost. The L2 argument only bites at widths where the
kernel already wins, and those are the widths production does not use.

## 6. If anyone picks this up again

In rough order of value:

1. **Pull the activation into the kernel.** Apply LeakyReLU before the store in
   the forward and multiply by its derivative in the backward (using the saved
   output's sign -- LeakyReLU is monotonic). This is the single biggest lever: it
   recovers the fusion that costs ~1.9x today, and additionally saves a full
   read+write of `(n, units)` per layer per direction. Contained entirely in
   `image_kernels.py`.
2. **Gate on width.** Even fixed, the kernel is unlikely to beat the fused
   reference below width ~32. A width threshold would be the shipping form.
3. **Measure the image layers' share at width 8** before doing either. It was
   never measured -- the width-8 numbers above report fusion loss, not layer
   share, because the kernel is not faster there in isolation either. Comparing
   `--image-layers=0` against `--image-layers=2` at width 8 gives it directly and
   takes minutes. If the share is small, this whole line of work has a low ceiling
   for production and should stay parked.
4. **Re-check at production scale.** Every number here is on the *small* dataset
   (4.1 M reflections, 13,482 images), where at `nb=32` the live `grad_w` region is
   1.65 MiB and fits entirely in the A6000's 6 MB L2 -- a 97.4 % hit rate. That is
   the regime most favourable to the scatter this kernel replaces. At 94.5 k images
   the region does not fit, and the scatter should degrade while the kernel does
   not.

## 7. Dead ends, already paid for

* **`torch._grouped_mm`.** Exists in torch 2.13, runs on sm86, takes exactly the
  offsets this problem produces, accurate to 6.9e-7 relative. At the real shape
  (422 groups, 128,800 rows, 32x32 out) it is **3.87 ms against the scatter's
  1.94 ms** -- 2x slower. The GEMMs are far too small for a grouped-GEMM path tuned
  for sm90. Also requires the row count to be a multiple of 8.
* **Padding into a dense `(images, slots, features)` layout** (the `expand` trick,
  whose backward autograd turns into a batched GEMM). Works, and is 6-7x on
  synthetic data -- but real image sizes are heavily skewed (p50 265, p100 2073),
  so padding to the per-batch max costs **4.46x waste**. That pulls it back to
  1.87x on the layer at **2.85x activation memory** (145.7 -> 414.8 MiB), and
  `(G, S)` differs for all 32 batches, which would blow `cache_size_limit` and drop
  torch.compile silently back to eager. Strictly worse than the kernel here.
* **`repeat_interleave`** instead of the gather: 3.85 ms vs 1.33 ms. Worse. It
  materialises, and its backward is still `index_add`.
* **Inductor autotune flags.** `max-autotune-no-cudagraphs` already sets
  `coordinate_descent_tuning=True`, so the autotuner had already searched
  `XBLOCK`/`num_warps` and *chose* the layout that causes the problem.
  `coordinate_descent_check_all_directions=1` is worth ~5 % on its own and is
  unrelated to this branch.
