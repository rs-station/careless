#!/bin/bash
# Run the performance sweeps behind doc/performance/README.md and width.md.
#
#   ./doc/performance/sweeps.sh <outdir> [sweep ...]
#
# Sweeps: modes  - torch.compile mode comparison        (~10 min)
#         batches- --num-batches at fixed width         (~25 min)
#         width  - --mlp-width at fixed --num-batches   (~45 min)
#         all    - all three (default)
#
# Required:
#   CARELESS_DATA   directory holding small_friedel_plus.mtz and
#                   small_friedel_minus.mtz
# Optional:
#   ITERS           steps per run (default 30). Timing uses the median of the
#                   steps after warm-up, so 30 is plenty; raise it only to check
#                   for drift.
#   WIDTHS          widths to scan (default "8 9 10 16 17 24 32 33 48 64 96 128")
#   BATCHES         batch counts to scan (default "1 2 4 8 16 32")
#   WIDTH_NB        --num-batches used during the width sweep (default 32)
#   BATCH_W         --mlp-width used during the batches sweep (default 8)
#   COLD            1 (default) gives every run its own inductor/triton cache so
#                   the recorded first step is an honest cold compile. Set 0 to
#                   share one cache and measure warm compile instead.
#   CACHE_ROOT      where per-run compiler caches go (default <outdir>/cache).
#                   On a cluster, point this at node-local or scratch storage --
#                   Triton's cache does a lot of small-file I/O and file locking,
#                   which a shared parallel filesystem handles badly.
#
# Every run writes <outdir>/<sweep>/<tag>.json; analyze.py turns the tree into
# the tables. Runs are independent, so a failed cell does not stop the sweep and
# can be re-run on its own.
set -u

OUT=${1:?usage: sweeps.sh <outdir> [modes|batches|width|all ...]}
shift || true
SWEEPS=${*:-all}

: "${CARELESS_DATA:?set CARELESS_DATA to the directory holding small_friedel_*.mtz}"
PLUS=$CARELESS_DATA/small_friedel_plus.mtz
MINUS=$CARELESS_DATA/small_friedel_minus.mtz
for f in "$PLUS" "$MINUS"; do
  [ -r "$f" ] || { echo "cannot read $f" >&2; exit 2; }
done

ITERS=${ITERS:-30}
WIDTHS=${WIDTHS:-"8 9 10 16 17 24 32 33 48 64 96 128"}
BATCHES=${BATCHES:-"1 2 4 8 16 32"}
WIDTH_NB=${WIDTH_NB:-32}
BATCH_W=${BATCH_W:-8}
COLD=${COLD:-1}
CACHE_ROOT=${CACHE_ROOT:-$OUT/cache}

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
BENCH=$HERE/bench_compile_mode.py
METADATA=dHKL,cartesian_fixed_x,cartesian_fixed_y,cartesian_fixed_z,ewald_offset

mkdir -p "$OUT"

# Record what this was measured on, so the JSON tree is self-describing.
python - "$OUT/meta.json" <<'PY'
import json, platform, subprocess, sys
meta = {"python": platform.python_version(), "host": platform.node()}
try:
    import torch
    meta.update(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                total_mem_mib=(torch.cuda.get_device_properties(0).total_memory / 2**20
                               if torch.cuda.is_available() else None))
except Exception as exc:
    meta["torch_error"] = str(exc)
try:
    import triton; meta["triton"] = triton.__version__
except Exception: pass
try:
    import careless; meta["careless"] = careless.__version__
except Exception: pass
try:
    meta["commit"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception: pass
json.dump(meta, open(sys.argv[1], "w"), indent=2)
print("environment:", json.dumps(meta))
PY

# $1 tag  $2 subdir  $3.. careless args
run_case () {
  local tag=$1 sub=$2; shift 2
  local dir=$OUT/$sub
  mkdir -p "$dir"
  local json=$dir/$tag.json
  if [ -s "$json" ]; then
    echo "  [skip] $sub/$tag (already present)"
    return 0
  fi
  local cache=$CACHE_ROOT/$sub/$tag
  rm -rf "$cache"; mkdir -p "$cache"
  local work; work=$(mktemp -d "$dir/.work.$tag.XXXX")
  (
    cd "$work" || exit 1
    if [ "$COLD" = "1" ]; then
      export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
    else
      export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/shared/inductor TRITON_CACHE_DIR=$CACHE_ROOT/shared/triton
    fi
    export BENCH_STOP_AFTER_TRAIN=1
    python "$BENCH" --bench-out "$json" -- mono \
      --disable-progress-bar --separate-files --mlp-layers=10 --dmin=1.8 \
      --double-wilson-r 0.0,0.98 --double-wilson-parents None,0 \
      --optimize-double-wilson-r --studentt-likelihood-dof=8 --kl-weight=0.005 \
      --iterations="$ITERS" "$@" \
      "$METADATA" "$PLUS" "$MINUS" out
  ) > "$dir/$tag.log" 2>&1
  local rc=$?
  rm -rf "$work"
  if [ ! -s "$json" ]; then
    # No JSON at all means the process died hard -- a segfault or an OOM-kill
    # takes the harness with it before it can record anything.
    echo "  [CRASH] $sub/$tag (rc=$rc, see $dir/$tag.log)"
    return 0
  fi
  # The harness records a JSON even when careless raises, so existence is not
  # success: read the status it wrote.
  local status
  status=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$json" 2>/dev/null)
  case "$status" in
    ok*) echo "  [ok]    $sub/$tag" ;;
    *)   echo "  [ERROR] $sub/$tag: ${status:0:100}" ;;
  esac
  return 0
}

want () { case " $SWEEPS " in *" all "*|*" $1 "*) return 0;; *) return 1;; esac; }

if want modes; then
  echo "== compile modes (width $BATCH_W, --num-batches=1) =="
  run_case eager modes --mlp-width=$BATCH_W --image-layers=2 --num-batches=1
  for m in default reduce-overhead max-autotune max-autotune-no-cudagraphs; do
    run_case "$m" modes --mlp-width=$BATCH_W --image-layers=2 --num-batches=1 \
      --jit-compile --jit-compile-mode="$m"
  done
  # --reduce-retracing with a non-CUDA-graphs mode is legitimate and worth timing.
  run_case "max-autotune-no-cudagraphs_dynamic" modes \
    --mlp-width=$BATCH_W --image-layers=2 --num-batches=1 \
    --jit-compile --jit-compile-mode=max-autotune-no-cudagraphs --reduce-retracing
  # With a CUDA-graphs mode it segfaulted on torch 2.13, so _torch_compile_kwargs
  # now refuses the combination. This cell therefore exercises the *guard*: the
  # expected result is [ERROR] ... ValueError. To re-test whether the underlying
  # segfault still exists on a newer stack, run
  # check_cudagraph_dynamic_segfault.py, which bypasses the guard deliberately.
  run_case "max-autotune_dynamic_guard" modes \
    --mlp-width=$BATCH_W --image-layers=2 --num-batches=1 \
    --jit-compile --jit-compile-mode=max-autotune --reduce-retracing
fi

if want batches; then
  echo "== --num-batches (width $BATCH_W) =="
  for nb in $BATCHES; do
    run_case "eager_nb$nb" batches --mlp-width=$BATCH_W --image-layers=2 --num-batches="$nb"
    run_case "jit_nb$nb"   batches --mlp-width=$BATCH_W --image-layers=2 --num-batches="$nb" \
      --jit-compile
  done
fi

if want width; then
  echo "== --mlp-width (--num-batches=$WIDTH_NB) =="
  for w in $WIDTHS; do
    run_case "eager_w$w" width --mlp-width="$w" --image-layers=2 --num-batches=$WIDTH_NB
    run_case "jit_w$w"   width --mlp-width="$w" --image-layers=2 --num-batches=$WIDTH_NB \
      --jit-compile
  done
fi

echo
echo "done. analyze with:"
echo "  python $HERE/analyze.py $OUT"
