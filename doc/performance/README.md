# torch.compile modes: measurements and how to reproduce them

`--jit-compile` wraps the training step in `torch.compile`. `--jit-compile-mode`
chooses the compiler mode. This note records what each mode was worth, why the
default is what it is, and how to re-measure it.

**Summary: `--jit-compile --jit-compile-mode=max-autotune-no-cudagraphs` (the
default mode) ran 2.9x faster and used 2.5x less peak memory than eager, with
merged structure factors agreeing to 1.3e-7 relative.**

## The measurement

RTX A6000 (48 GB, sm86), PyTorch 2.13.0+cu130, Triton 3.7.1, careless 0.5.5,
measured 2026-08-27 on `perf/nsys-training-fixes`.

Dataset: the window-merge example from
[careless-examples](https://github.com/rs-station/careless-examples),
`small_friedel_{plus,minus}.mtz` -- 4,121,693 reflections over 13,482 images --
with production parameters: `--mlp-layers=10 --mlp-width=8 --image-layers=2
--dmin=1.8 --studentt-likelihood-dof=8 --kl-weight=0.005`, a double-Wilson prior
with `--optimize-double-wilson-r`.

Timings are the mean of steps 6-29 over two repeats, so they exclude compilation
(step 0) and CUDA-graph warm-up (steps 1-2). Compile time is with a cold
`TORCHINDUCTOR_CACHE_DIR`.

| `--jit-compile-mode` | ms/step | speedup | peak MiB | memory saving | cold compile |
|---|---:|---:|---:|---:|---:|
| *(no `--jit-compile`)* | 119.3 | 1.00x | 6521 | 1.00x | -- |
| `default` | 72.4 | 1.65x | 4681 | 1.39x | ~9 s |
| `reduce-overhead` | 72.4 | 1.65x | 4788 | 1.36x | ~9 s |
| `max-autotune` | 41.3 | 2.89x | 2734 | 2.39x | ~42 s |
| **`max-autotune-no-cudagraphs`** (default) | **40.6** | **2.94x** | **2655** | **2.46x** | ~43 s |

With the inductor cache warm -- that is, on every run after the first with a
given shape -- `max-autotune` compilation drops from ~42 s to ~12 s.

## Why the default is `max-autotune-no-cudagraphs`

It was both the fastest and the least memory hungry, and it has the steadiest
step time (sd 0.3 ms against 13.7 ms for `max-autotune` over the same window).

**CUDA graphs buy nothing here, and cost a little.** `reduce-overhead` is
indistinguishable from `default` (0.1 ms apart), and `max-autotune` is 0.7 ms per
step *slower* than the same autotuning without CUDA graphs. The reason is that
the model does not compile into one graph. `_accept_reject_truncnorm` in
`careless/distributions/truncated_normal.py` branches on `if done.all():` inside
a loop, which is data-dependent control flow; dynamo cannot trace it and skips
the frame entirely. The model ends up as roughly six dynamo frames, of which only
two forward/backward partitions get recorded as graphs. Both CUDA-graphs modes
also spend an extra ~100 ms step on warm-up and recording. Making the sampler
capturable -- a fixed two-iteration version with a clamp fallback, given that the
acceptance rate is ~100% for careless' bounds -- is the prerequisite for CUDA
graphs ever paying off, and it needs its own equivalence check.

**`--reduce-retracing` with a CUDA-graphs mode segfaults.** SIGSEGV immediately
after compilation, no traceback; reproduced on both `reduce-overhead` and
`max-autotune`, and *not* on the two non-CUDA-graphs modes, which is what
identifies CUDA graphs rather than autotuning as the culprit.
`VariationalMergingModel._torch_compile_kwargs` raises a `ValueError` on the
combination rather than letting the process die. `--reduce-retracing` is
pointless here anyway: careless' shapes are fixed for a whole run, and
`dynamic=True` measured within noise of `dynamic=False`.

## Equivalence

Over 30 steps, every compiled mode tracks the eager run to:

| metric | max relative deviation |
|---|---|
| `Loss`, `NLL` | 2.1e-7 |
| `F KLDiv` | 4.1e-6 (7.7e-6 for `max-autotune-no-cudagraphs`) |
| `Grad Norm` | 8e-7 |
| merged `F`, `SigF` | 1.3e-7, median deviation exactly 0, CC = 1.0000000000 |

That is float32 reassociation, and it is the same size as the deviation between
the modes themselves. The eager run is bit-reproducible from run to run at a
fixed seed, apart from `Grad Norm` at 1.4e-7 (the sum is taken in
`named_parameters()` order).

**This was checked over 30 steps, not the 10,000-30,000 of a production run.
Nobody has verified that the trajectories stay locked that far.** One 10,000-step
eager run was made and shows no step-time drift (per-decile median 121.1 ->
123.5 ms), but there is no compiled run of that length to compare it against.

## Host synchronizations

Counted with `torch.cuda.set_sync_debug_mode("warn")` and a traceback-collecting
warning hook, per training step:

| site | eager | compiled |
|---|---:|---:|
| `truncated_normal.py:22` `if done.all():` | 1 | 1 |
| `truncated_normal.py:29` `if not done.all():` | 1 | 1 |
| `variational.py` `if not torch.isfinite(loss):` | 1 | 1 |
| `variational.py` batched `.tolist()` | 1 | 1 |
| `scaling/nn.py` `Normal(loc, scale)` constraint check | 1 | 0 |
| `likelihoods/mono.py` `StudentT(...)` constraint check | 1 | 0 |
| `distributions/student_t.py` `_validate_sample(value)` | 1 | 0 |
| **total** | **7** | **4** |

### Are they a bottleneck? No -- about 5% of the compiled step, nothing in eager

Measured by ablation: rebuild `train_model` from `inspect.getsource` with
`if not torch.isfinite(loss):` and the batched `.tolist()` edited out (everything
else byte-identical), and/or swap `_accept_reject_truncnorm` for a single draw
plus a clamp fallback, which has no `done.all()`. 60 steps after a 15-step
warm-up, with `torch.cuda.synchronize()` at both ends of the timed region:

| syncs removed | max-autotune-no-cudagraphs | eager |
|---|---:|---:|
| none | 41.19 ms | 122.37 ms |
| the two in `train_model` | 40.25 (-2.3%) | 122.16 (-0.2%) |
| the two in the sampler | 40.67 (-1.3%) | 122.56 (+0.2%) |
| all four | 38.94 (**-5.5%**) | 121.93 (-0.4%) |

**In eager the syncs are free.** Removing four of the seven changes nothing
measurable, because the GPU is saturated: the host blocks on a sync only after
the GPU has already been given more work than the host can produce, so the stall
costs wall clock it was going to spend waiting anyway.

Compiling shrinks the GPU work per step by 3x without shrinking the host work by
as much, so host overhead becomes a larger share and the syncs start to show --
but still only 5.5%. Treat that as an upper bound: the sampler ablation also
removes a dynamo graph break and does slightly less arithmetic than the real
accept-reject, so not all of its 1.3% is the sync.

Two practical notes. The effect is close to all-or-nothing and the pieces are in
different files, so half the work gets much less than half the payoff. And
measuring this needs the final `torch.cuda.synchronize()`: with every sync
removed the host runs far ahead of the GPU, and per-iteration wall times then
measure launch speed rather than throughput -- an earlier version of this
measurement without the drain reported 23 ms/step for a step that really takes 39.

Two things follow from the table above. The metric-sync commit's "single batched
host sync per step"
is really four: the `torch.isfinite(loss)` divergence check two lines above the
batched read is a second sync, and the sampler adds two more. And three of
eager's seven are `torch.distributions` argument validation, which
`torch.compile` removes for free -- part of why compiling wins. Disabling that
validation directly is worth 4.3% eager, 1.3% `default`, 0.4%
`max-autotune-no-cudagraphs`, with a bit-identical loss:

```python
torch.distributions.Distribution.set_default_validate_args(False)
```

It is not wired to a flag; `BENCH_NOVALIDATE=1` turns it on in the harness below.

## Reproducing

`sync_ablation.py` reproduces the sync table above; `ABLATE=loop,sampler` selects
which syncs to remove. `bench_compile_mode.py` forwards everything after `--` to
the ordinary CLI. It
replaces tqdm's progress bar with one that timestamps the top of each step, so
the loop being measured is careless' own `train_model`, unmodified, driven by the
real `--jit-compile-mode` flag.

```bash
export ROOT=/path/to/careless-examples          # for small_friedel_*.mtz
export METADATA=dHKL,cartesian_fixed_x,cartesian_fixed_y,cartesian_fixed_z,ewald_offset
COMMON="--disable-progress-bar --separate-files --mlp-layers=10 --mlp-width=8
        --image-layers=2 --dmin=1.8 --iterations=30 --studentt-likelihood-dof=8
        --double-wilson-r 0.0,0.98 --double-wilson-parents None,0
        --optimize-double-wilson-r --kl-weight=0.005"

mkdir -p runs && cd runs

# eager baseline
python ../doc/performance/bench_compile_mode.py --bench-out eager.json -- \
    mono $COMMON $METADATA \
    $ROOT/data/small_friedel_plus.mtz $ROOT/data/small_friedel_minus.mtz eager

# one run per mode
for mode in default reduce-overhead max-autotune max-autotune-no-cudagraphs; do
    python ../doc/performance/bench_compile_mode.py --bench-out $mode.json -- \
        mono $COMMON --jit-compile --jit-compile-mode=$mode $METADATA \
        $ROOT/data/small_friedel_plus.mtz $ROOT/data/small_friedel_minus.mtz $mode
done

python ../doc/performance/summarize.py *.json
```

30 steps is enough: the step time is flat after warm-up and the whole sweep takes
about ten minutes. Run each mode twice if you want an error bar -- `summarize.py`
pools repeats that share a mode.

Two things worth setting when you care about the numbers:

* `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a fresh directory per run,
  if you want an honest *cold* compile time. Leave them alone to measure what a
  user sees on their second run.
* Nothing else on the GPU. `nvidia-smi` should show it idle before you start.

To compare merged output rather than the loss trajectory, read the `_0.mtz` files
each run writes and compare `F`/`SigF` after an inner join on the Miller indices.

## Gotchas

* Set `MPLBACKEND=Agg` when running the test suite. The default backend here is
  interactive, and the stats tests draw figures; with an unhappy X session they
  come back zero-sized and dozens of tests fail with `ValueError: height and
  width must be > 0`, which looks exactly like a real regression.
* careless installs editable, and `site-packages/__editable___*_finder.py` holds
  an absolute path to one checkout. Running `pip install -e .` from a second
  checkout -- a `git worktree`, say -- repoints it and breaks imports elsewhere.
  To benchmark a branch in a worktree, put it on `PYTHONPATH` instead of
  installing it; the editable finder is appended to `sys.meta_path`, so `sys.path`
  wins.
* `torch.compile` fails silently in the direction that matters least: if dynamo
  bails out and falls back to eager, the numbers still agree perfectly. Check
  that the fast path ran -- the step time is the tell here, but a test comparing
  compiled against eager output needs to prove compilation happened.
