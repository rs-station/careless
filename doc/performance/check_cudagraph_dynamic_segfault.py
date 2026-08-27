#!/usr/bin/env python
"""
Re-test the CUDA-graphs + dynamic-shapes segfault on a new torch/Triton stack.

On torch 2.13.0+cu130 / Triton 3.7.1, compiling careless with a CUDA-graphs mode
(`reduce-overhead` or `max-autotune`) *and* dynamic shapes segfaulted the process
immediately after compilation -- SIGSEGV, no traceback -- reproducibly, and on
neither non-CUDA-graphs mode. `VariationalMergingModel._torch_compile_kwargs` now
refuses that combination, so the CLI can no longer reach it.

This script bypasses the guard on purpose and runs a handful of steps in a child
process, so a crash is observed rather than inherited. Use it when the torch
version changes and you want to know whether the guard is still earning its keep.

    python doc/performance/check_cudagraph_dynamic_segfault.py \
        --data /path/to/small_friedel_mtz_directory

Exit code 0 means every combination was checked and the results are printed; it
does not mean they all passed. Read the table.
"""
import argparse
import os
import subprocess
import sys
import tempfile

CHILD = r'''
import sys, torch
from careless.models.merging.variational import VariationalMergingModel as M

mode, dynamic = sys.argv[1], sys.argv[2] == "1"
# Bypass the guard deliberately -- reproducing the bug is the whole point.
M._torch_compile_kwargs = staticmethod(
    lambda m, d: {"dynamic": d} if m == "default" else {"mode": m, "dynamic": d}
)

plus, minus, out = sys.argv[3], sys.argv[4], sys.argv[5]
argv = ["careless", "mono", "--disable-progress-bar", "--separate-files",
        "--mlp-layers=10", "--mlp-width=8", "--image-layers=2", "--dmin=1.8",
        "--iterations=5", "--double-wilson-r", "0.0,0.98",
        "--double-wilson-parents", "None,0", "--optimize-double-wilson-r",
        "--studentt-likelihood-dof=8", "--kl-weight=0.005", "--num-batches=1",
        "--jit-compile", "--jit-compile-mode=" + mode]
if dynamic:
    argv.append("--reduce-retracing")
argv += ["dHKL,cartesian_fixed_x,cartesian_fixed_y,cartesian_fixed_z,ewald_offset",
         plus, minus, out]
sys.argv = argv

import careless.models.merging.variational as v
_orig = v.VariationalMergingModel.train_model
def stop_after_training(self, *a, **kw):
    h = _orig(self, *a, **kw)
    print("TRAINED_OK", flush=True)
    raise SystemExit(0)
v.VariationalMergingModel.train_model = stop_after_training

from careless.careless import main
main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory holding small_friedel_{plus,minus}.mtz")
    args = ap.parse_args()

    plus = os.path.join(args.data, "small_friedel_plus.mtz")
    minus = os.path.join(args.data, "small_friedel_minus.mtz")
    for f in (plus, minus):
        if not os.path.isfile(f):
            sys.exit(f"cannot read {f}")

    import torch
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except ImportError:
        pass
    print(f"gpu {torch.cuda.get_device_name(0)}\n")

    cases = [(m, d) for m in ("max-autotune", "reduce-overhead",
                              "max-autotune-no-cudagraphs", "default")
             for d in (True, False)]

    print(f"{'mode':32s} {'dynamic':>8s}  result")
    crashed = []
    with tempfile.TemporaryDirectory() as td:
        for mode, dynamic in cases:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, mode, "1" if dynamic else "0",
                 plus, minus, os.path.join(td, "out")],
                capture_output=True, text=True, cwd=td,
            )
            if proc.returncode < 0:
                verdict = f"CRASHED with signal {-proc.returncode}"
                crashed.append((mode, dynamic))
            elif "TRAINED_OK" in proc.stdout:
                verdict = "ok"
            else:
                tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
                verdict = f"failed (rc={proc.returncode}): {tail[:70]}"
            print(f"{mode:32s} {str(dynamic):>8s}  {verdict}")

    print()
    if crashed:
        print("The segfault is still present for:",
              ", ".join(f"{m}+dynamic" if d else m for m, d in crashed))
        print("Keep the guard in _torch_compile_kwargs.")
    else:
        print("Nothing crashed. If the CUDA-graphs + dynamic cases all pass, the")
        print("guard in _torch_compile_kwargs may be relaxable -- record the torch")
        print("and Triton versions alongside this output before doing so.")


if __name__ == "__main__":
    main()
