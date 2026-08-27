# Repeating these measurements on another GPU

Everything in [README.md](README.md) and [width.md](width.md) was measured on one
RTX A6000. This note is for an agent re-running it elsewhere -- the immediate case
is NERSC Perlmutter with A100s. It covers what to run, what should hold, what
should change, and what went wrong the first time.

## Run it

```bash
export CARELESS_DATA=/path/to/small_friedel_mtz_directory
./doc/performance/sweeps.sh $SCRATCH/careless-perf-a100
python doc/performance/analyze.py $SCRATCH/careless-perf-a100
```

That is the whole thing. `sweeps.sh` runs three sweeps and writes one JSON per
cell; `analyze.py` prints every table. Budget roughly **80 minutes of GPU time**
on hardware like an A6000 (modes ~10 min, batches ~25, width ~45).

Useful knobs, all environment variables (see the header of `sweeps.sh`):

| variable | why you would set it |
|---|---|
| `CACHE_ROOT` | put Triton/inductor caches on fast local storage -- see below |
| `WIDTHS`, `BATCHES` | trim the grid for a first pass, e.g. `WIDTHS="8 16 17 32 33"` |
| `ITERS` | steps per cell, default 30. Timing is a median after warm-up, so 30 is enough |
| `COLD=0` | share one compiler cache and measure *warm* compile instead of cold |

Runs are independent and each is skipped if its JSON already exists, so an
interrupted sweep resumes by re-running the same command, and a single failed
cell can be re-run alone. Each cell reports `[ok]`, `[ERROR]` with the recorded
status, or `[CRASH]` if the process died hard enough to take the harness with it
(a segfault or an OOM kill); its `.log` sits next to the JSON either way.
`analyze.py` lists any cell that produced no usable timing, so a partly-failed
sweep cannot be mistaken for a complete one.

Do the whole thing on **one** GPU. The harness reads
`torch.cuda.max_memory_allocated()` on device 0 and nothing here is
multi-GPU-aware; on a 4-GPU node set `CUDA_VISIBLE_DEVICES=0`.

## Perlmutter specifics

**Unverified from here.** The CFS paths below are the ones already baked into
`examples/window_merge_no_preprocess/merge.sh`, so they are real; the module and
Slurm details are the standard shape of a Perlmutter GPU job and should be
checked on the machine (`module avail pytorch`, `sacctmgr show assoc user=$USER`)
rather than trusted.

The data and an existing careless environment are on CFS:

```bash
export CARELESS_DATA=/global/cfs/cdirs/ntrain6/LCLS/examples/data
source /global/cfs/cdirs/ntrain6/LCLS/software/setup.sh
```

An interactive node, which is the right way to do the first pass:

```bash
salloc --nodes 1 --constraint gpu --gpus 4 --qos interactive \
       --time 02:00:00 --account <your-account>
export CUDA_VISIBLE_DEVICES=0
```

Three things that bite on a cluster and not on a workstation:

* **Put the compiler caches somewhere that likes small files.** Triton writes many
  small files and takes file locks. On Lustre (`$SCRATCH`, `$PSCRATCH`, CFS) that
  is slow and can hang. Prefer node-local storage:
  `export CACHE_ROOT=/tmp/$USER/careless-cache`. If `/tmp` is a small RAM disk,
  use `$PSCRATCH` and expect the cold-compile column to be inflated -- say so when
  reporting it.
* **Compute nodes have no outbound network.** Install or update anything from a
  login node first. `sweeps.sh` itself needs no network.
* **Batch the long run.** The width sweep is the expensive one. A job script:

```bash
#!/bin/bash
#SBATCH --constraint gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 02:30:00
#SBATCH --qos regular
#SBATCH --account <your-account>
#SBATCH --job-name careless-perf
#SBATCH --output %x-%j.out

source /global/cfs/cdirs/ntrain6/LCLS/software/setup.sh
export CARELESS_DATA=/global/cfs/cdirs/ntrain6/LCLS/examples/data
export CACHE_ROOT=/tmp/$USER/careless-cache
export CUDA_VISIBLE_DEVICES=0
srun ./doc/performance/sweeps.sh $SCRATCH/careless-perf-a100
```

**Check the card's memory before starting.** Perlmutter has both 40 GB and 80 GB
A100s. The A6000 used here has 48 GB, and the eager width-128 cell peaked at
**32,757 MiB** -- that fits on a 40 GB card but not by much, and eager width 96 and
128 are the first cells that will fall over if the node is smaller than expected.
`analyze.py` prints the card's total memory from `meta.json`, and a cell that OOMs
is recorded as a `[FAIL]` rather than stopping the sweep.

## What should hold, and what should not

Separating these matters: a number that changes is not a contradiction, and a
*structural* claim that changes is a real finding.

### Should hold -- these are properties of the code, not the card

* **CUDA graphs buy nothing.** `reduce-overhead` should match `default`, and
  `max-autotune` should not beat `max-autotune-no-cudagraphs`. The cause is
  `_accept_reject_truncnorm` in `careless/distributions/truncated_normal.py`
  branching on `if done.all():` inside a loop -- data-dependent control flow that
  dynamo cannot trace, so it skips the frame and the model never becomes one
  graph. Nothing about that depends on the GPU. If CUDA graphs suddenly help on
  A100, something else changed and it is worth chasing.
* **Two dynamo graphs at every `--num-batches`.** Automatic-dynamic promotion
  kicks in on the second distinct shape and stops. This is a torch-version
  property; if the Perlmutter torch is older, re-check for a `cache_size_limit`
  warning, which would mean silent fallback to eager and a quietly wrong-looking
  speedup.
* **Accumulation costs less compiled than eager.** The mechanism is that
  accumulation adds per-batch fixed overhead and compilation removes it.
* **Equivalence.** Compiled should track eager to ~1e-7 on Loss and NLL. See the
  TF32 caveat below, which is the one place this could legitimately break.

### Should change -- read these as "re-measure", not "verify"

* **Every absolute time.** An A100 has roughly twice the memory bandwidth of an
  A6000 (~1.5-2.0 TB/s against 768 GB/s), and careless at these widths is
  bandwidth- and launch-bound rather than FLOP-bound, so expect the whole table to
  shift down by something like 2x. The *ratios* are the portable part.
* **The tile quantum.** On the A6000 the compiled step time was a staircase with
  steps at multiples of **16** -- one channel past 16 cost 56%. An A100 is sm80
  with different tensor-core shapes and a 40 MB L2 against the A6000's 6 MB, and
  Triton may well pick different tiles. `analyze.py` prints the excess for
  Q = 8, 16 and 32 side by side precisely so the quantum is **re-derived rather
  than assumed**. Whichever Q shows a large *compiled* excess and a small *eager*
  one is the answer on that machine.
* **The shape of the speedup-vs-width curve.** It was U-shaped here (4.00x at
  width 8, 1.65x at 33, 2.98x at 128). The middle dip was attributed -- as a
  hypothesis, never measured -- to cuBLAS already being good at the GEMMs there.
  A100 tensor cores are far stronger, so the dip could deepen or move.
* **Peak memory.** Should be broadly similar since the allocations are the same,
  but inductor may make different fusion choices.
* **The `--reduce-retracing` segfault.** On torch 2.13 + Triton 3.7, a CUDA-graphs
  mode combined with `--reduce-retracing` segfaulted immediately after
  compilation -- SIGSEGV, no traceback -- reproducibly, on both `reduce-overhead`
  and `max-autotune` and on neither non-CUDA-graphs mode.
  `VariationalMergingModel._torch_compile_kwargs` now refuses the combination, so
  the CLI can no longer reach it and `sweeps.sh` cannot test it: the
  `max-autotune_dynamic_guard` cell exercises the *guard*, and its expected
  result is `[ERROR] ... ValueError`.

  To find out whether the underlying bug still exists on the Perlmutter stack:

  ```bash
  python doc/performance/check_cudagraph_dynamic_segfault.py --data $CARELESS_DATA
  ```

  It bypasses the guard on purpose and runs all eight mode x dynamic combinations
  in child processes, so a crash is observed rather than inherited. On the A6000
  it prints:

  ```
  max-autotune                         True  CRASHED with signal 11
  reduce-overhead                      True  CRASHED with signal 11
  max-autotune-no-cudagraphs           True  ok
  default                              True  ok            (+ all four dynamic=False: ok)
  ```

  If nothing crashes there, the guard may be relaxable -- but record the torch and
  Triton versions alongside the output before touching it.

### The thing most likely to genuinely break: TF32

`train_model` calls `torch.set_float32_matmul_precision('high')` unconditionally,
which lets eager matmuls use TF32. Inductor, in this build, chose kernels with
`ALLOW_TF32=False`. On an A6000 that mismatch is harmless because TF32 barely
helps below width ~40, so both paths were doing much the same arithmetic and
agreed to 1e-7 at every width.

**On an A100 TF32 is a far bigger lever.** The eager and compiled paths may then
be doing genuinely different arithmetic, and the equivalence check could exceed
1e-5. If it does:

* That is not automatically a bug in compilation. It is more likely eager using
  TF32 where the compiled path is not.
* Confirm by re-running the same width with
  `torch.set_float32_matmul_precision('highest')` forced -- if the two paths then
  agree, TF32 is the whole story.
* Then find out which side is *right*, against a float64 reference, before
  concluding anything. `HANDOFF.md` §9 has the standing rule and the fused-trunk
  case where eager turned out to be the less accurate side.

`analyze.py` flags any width whose deviation exceeds 1e-5 with `<-- check this`.

## Traps already paid for

Do not rediscover these.

* **`MPLBACKEND=Agg` when running the test suite.** The stats tests draw figures;
  with an unhappy display they come back zero-sized and dozens of tests fail with
  `ValueError: height and width must be > 0`, which looks exactly like a real
  regression.
* **Do not `pip install -e .` from a second checkout.** careless installs editable
  and `site-packages/__editable___*_finder.py` holds an absolute path to one
  checkout; installing from a worktree repoints it and breaks imports everywhere
  else. To benchmark a branch from a worktree, put it on `PYTHONPATH` -- the
  editable finder is appended to `sys.meta_path`, so `sys.path` wins.
* **Measure throughput with the queue drained.** Removing every host sync lets the
  CPU run far ahead of the GPU, and per-iteration wall times then measure launch
  speed rather than throughput. An earlier version of the sync ablation reported
  23.5 ms/step for a step that really takes 38.9. `sync_ablation.py` now
  synchronizes at both ends of the timed region.
* **Do not instrument inside a compiled region.** A `time.perf_counter()` call
  inside a function dynamo is tracing forces a graph break. Timing the sampler
  that way cost 3 ms/step and 30x the variance -- it changed the thing being
  measured.
* **An equivalence test whose fast path can silently fall back is not a test.**
  If compilation quietly bails to eager, every number still agrees. Check that the
  fast path ran; the step time is the tell here, and
  `tests/models/merging/test_batched_inference.py` does it properly by counting
  calls.
* **Relative tolerance on an array that passes near zero is the wrong metric.** A
  scale posterior predicting ~6e-3 inside an array spanning ~1 turns the float32
  floor into a 1e-5 "failure". Measure deviation against the scale of the whole
  array. This cost a real debugging detour.

## The files

| file | what it is |
|---|---|
| `sweeps.sh` | runs all three sweeps; one JSON per cell |
| `analyze.py` | turns the JSON tree into every table, including the quantization scan and the equivalence checks |
| `bench_compile_mode.py` | times one careless run; the sweeps call this |
| `check_cudagraph_dynamic_segfault.py` | re-tests the CUDA-graphs + dynamic-shapes crash on a new stack |
| `sync_ablation.py` | the per-step host-synchronization ablation (worth ~5% compiled, nothing eager) |
| `README.md`, `width.md` | the A6000 results these scripts produced |

## Reporting back

`meta.json` in the output directory records GPU, compute capability, total
memory, torch/CUDA/Triton versions and the careless commit, so a results tree is
self-describing. Keep it with the JSONs.

When writing the comparison up, keep the A6000 numbers as a column rather than
replacing them -- the interesting content is which conclusions travelled and which
did not.
