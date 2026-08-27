# Scaling-model width, compiled, with gradient accumulation

A scan of `--mlp-width` at `--num-batches=32` on the compiled and eager paths.
Companion to [README.md](README.md), same machine and dataset: RTX A6000 (48 GB,
sm86), PyTorch 2.13.0+cu130, Triton 3.7.1, `small_friedel_{plus,minus}.mtz`
(4,121,693 reflections), `--mlp-layers=10 --image-layers=2 --dmin=1.8
--studentt-likelihood-dof=8 --kl-weight=0.005`, double-Wilson prior. Compiled
runs use the default `--jit-compile-mode=max-autotune-no-cudagraphs`.

30 steps per cell, median of steps 11-30, one run each. Cold compile is the first
step with a fresh `TORCHINDUCTOR_CACHE_DIR`. Timing is of the training loop only
(`BENCH_STOP_AFTER_TRAIN=1`).

| width | eager ms | eager MiB | compiled ms | compiled MiB | speedup | memory | cold compile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 231.6 | 527 | 57.9 | 411 | 4.00x | 1.28x | 35 s |
| 9 | 261.1 | 566 | 66.5 | 426 | 3.93x | 1.33x | 43 s |
| **10** (default) | 274.8 | 616 | **71.4** | 448 | 3.85x | 1.38x | 39 s |
| 16 | 320.2 | 980 | 101.0 | 576 | 3.17x | 1.70x | 39 s |
| 17 | 368.3 | 1054 | 158.3 | 602 | 2.33x | 1.75x | 58 s |
| 24 | 457.1 | 1686 | 191.3 | 814 | 2.39x | 2.07x | 43 s |
| 32 | 625.3 | 2628 | 327.3 | 1094 | 1.91x | 2.40x | 48 s |
| 33 | 716.2 | 2761 | 434.3 | 1154 | 1.65x | 2.39x | 47 s |
| 48 | 988.1 | 5231 | 583.4 | 1880 | 1.69x | 2.78x | 53 s |
| 64 | 1568.6 | 8805 | 749.8 | 2933 | 2.09x | 3.00x | 64 s |
| 96 | 3030.3 | 18849 | 1264.7 | 5825 | 2.40x | 3.24x | 144 s |
| 128 | 5043.6 | 32757 | 1693.0 | 9774 | 2.98x | 3.35x | 177 s |

**Run-to-run spread is not the same on the two paths.** Repeating widths 16 and 17
gave 100.8 and 157.4 ms compiled against 101.0 and 158.3 first time round -- 0.5%
-- but 305.9 and 361.8 ms eager against 320.2 and 368.3 -- up to 4.7%. Read the
eager column, and therefore the speedup column, as +/-5%; the compiled column is
tight.

## Compiled time is a staircase in width, with steps at multiples of 16

This is the most actionable result, and it does not show up in eager.

| crossing | eager | compiled |
|---|---:|---:|
| 16 -> 17 | +15% | **+56%** |
| 32 -> 33 | +15% | **+33%** |

One extra channel past 16 costs 56% of the step. And it is not a spike at that one
width: widths 18 and 20 sit on the same raised plateau (155.7 and 165.3 ms against
100.8 at width 16), so the compiled path is quantized -- crossing a multiple of 16
buys a whole new tile and the cost is paid at once, then grows slowly until the
next one. Against a straight line drawn between the neighbouring 16-multiples,
widths 17, 18 and 20 come in +40%, +26% and +13% compiled, against +11%, +8% and
+7% eager.

`HANDOFF.md` §2 describes "alignment potholes" at widths congruent to 1 mod 8,
measured on an isolated GEMM stack at constant bytes moved. This scan measures the
whole step at a fixed dataset size, and what it sees is a staircase rather than
isolated potholes -- widths 18 and 20 are penalised nearly as much as 17. Both can
be true of different experiments; for choosing a width on the compiled path, the
staircase is the model to use.

**Pick a multiple of 16.** Widths 8, 16, 32, 48, 64, 96, 128 are all on the
efficient side of a step. Width 10, careless' default, costs 23% more per step than
width 8 and 3.5% more per unit of width; that is a small price and 10 is fine, but
there is nothing to be gained between 17 and 24, or between 33 and 48.

## The compiler's advantage is U-shaped in width

4.00x at width 8, down to 1.65x at 33, back up to 2.98x at 128.

A plausible reading, **not measured**: at small widths the step is dominated by
launch and per-kernel overhead -- at `--num-batches=32` there are 32 batches of
many small kernels -- which is exactly what fusion removes. In the middle the
GEMMs dominate and cuBLAS is already good at them, so there is less to win. At
large widths memory traffic dominates again and inductor's fusion wins ground
back; the memory column supports this reading, since the compiled path's peak
falls from 1.28x to 3.35x below eager over the same range, meaning it is
materializing steadily less.

## Compile time

Flat at 35-65 s from width 8 through 64, then 144 s at 96 and 177 s at 128. Width
17 is an outlier at 58 s against 39 s for both neighbours, which is consistent
with the staircase -- more tile configurations to autotune. All of this is once per
machine per shape; the inductor cache makes later runs far cheaper.

## Inference used to be the ceiling; it is batched now

`--num-batches` originally applied only to the training loop. The prediction and
mtz-writing stage that follows ran the scaling model over the **whole** dataset in
one call, so `ImageLayer`'s per-observation weight gather materialized
`n_obs x width^2` floats at once -- 35.4 GiB at width 48, 62.9 at 64, 251.6 at 128.
Measured on this 48 GB card, width 48 completed at 39,960 MiB peak and **width 64
died with "tried to allocate 62.89 GiB" after training had already finished
successfully**. Accumulation had moved the bottleneck out of training and into the
output stage.

`scale_moments` now evaluates the scaling model in the same contiguous chunks
training uses, and `--num-batches` reaches the prediction pass. Only the scaling
model is chunked; the full-length arrays are assembled before anything downstream
runs, so the Laue convolution -- which ranges over harmonic groups and indexes by a
global `harmonic_id` -- still sees the whole array exactly as before.

| width | peak, whole-dataset inference | peak, `--num-batches=32` |
|---:|---:|---:|
| 32 | 18,565 MiB | 1,077 MiB |
| 48 | 39,960 MiB | -- |
| 64 | **OOM at 62.9 GiB** | 3,258 MiB |
| 128 | **OOM at 251.6 GiB** | 11,965 MiB |

Width 128 now completes the full pipeline, writes its mtz files, and peaks at
11,965 MiB -- below what width 32 used to need just to write output.

The batched path is bit-identical on the `Scale` and `SigScale` columns across
2.15 M predicted reflections. `Ipred` and `SigIpred` differ by 1-5e-7 of column
scale, but so do two runs of the *same* build: the residual is run-to-run training
nondeterminism, not the batching. `tests/models/merging/test_batched_inference.py`
pins the equivalence in-process, where there is no such noise, on mono and Laue
inputs over cpu and cuda, and counts the scaling-model calls so a silent fallback
to the whole-dataset path cannot pass.

## Equivalence

Compiled against eager at the same width, worst relative deviation over 30 steps:

| | Loss | NLL | F KLDiv | Grad Norm |
|---|---|---|---|---|
| every width 8-128 | <= 2.2e-7 | <= 2.2e-7 | <= 1.4e-7 | <= 2.3e-7 |
| except width 64 | 2.1e-7 | 2.1e-7 | **6.9e-6** | **6.0e-7** |

No width shows the divergence that would signal a precision change. That was worth
checking specifically, because `train_model` sets
`torch.set_float32_matmul_precision('high')` unconditionally, and `HANDOFF.md` §2
reports TF32 starting to bite above width 40 -- so widths 48 and up could have
diverged between a TF32 eager GEMM and an inductor kernel that chose
`ALLOW_TF32=False`. They do not, at least over 30 steps. Width 64's `F KLDiv` at
6.9e-6 is in line with the 4-8e-6 seen for that metric elsewhere in
[README.md](README.md#equivalence).

## What to actually run

On this card, for this dataset:

* **Width 8-16 with `--jit-compile`**: 58-101 ms/step, under 600 MiB, 3.2-4.0x
  faster than eager. If the science tolerates it, this is where the compiler pays
  most.
* **Width 32-48**: 327-583 ms/step, 1.1-1.9 GiB training, still writes output.
  Width 48 is the largest that completes the pipeline.
* **Above 48**: now runs end to end. Width 128 trains in 9,774 MiB and completes
  the whole pipeline at 11,965 MiB.

## Reproducing

The whole width scan is one command; see [PORTING.md](PORTING.md), which also
covers what to expect on other hardware -- in particular that the width-16 quantum
is a property of the kernels Triton selected here and must be re-derived, not
assumed. `analyze.py` prints the excess for Q = 8, 16 and 32 side by side for
exactly that reason.

### By hand

`doc/performance/bench_compile_mode.py` with `--mlp-width` varied and
`BENCH_STOP_AFTER_TRAIN=1`; see [README.md](README.md#reproducing) for the
invocation. Drop `BENCH_STOP_AFTER_TRAIN` to exercise the prediction pass as well.
