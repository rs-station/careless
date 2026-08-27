#!/usr/bin/env python
"""
Time one careless run, step by step, so torch.compile modes can be compared.

This is a thin wrapper around the ordinary CLI: it forwards every argument to
`careless`, swaps tqdm's progress bar for one that timestamps the top of each
iteration, and writes a JSON record of the per-step times, the peak memory and
the full metric history.

    python doc/performance/bench_compile_mode.py \
        --bench-out run.json -- mono --jit-compile \
        --jit-compile-mode=max-autotune-no-cudagraphs ... <files> <out>

The training loop being measured is careless' own `train_model`, unmodified --
only the progress bar is replaced. The compile mode comes from the real
`--jit-compile-mode` flag, so this measures the shipped code path.

Optional environment variables:
    BENCH_NOVALIDATE=1   disable torch.distributions argument validation
"""
import argparse
import json
import os
import sys
import time

import torch


def parse_bench_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--bench-out", default="bench.json")
    known, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return known, rest


BENCH, CARELESS_ARGV = parse_bench_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + CARELESS_ARGV

if os.environ.get("BENCH_NOVALIDATE") == "1":
    # Distribution.__init__ and log_prob validate their arguments with a full-size
    # elementwise kernel followed by a blocking read -- three host syncs per step.
    torch.distributions.Distribution.set_default_validate_args(False)
    print("[bench] distribution validate_args disabled", flush=True)

STATE = {"times": []}


class TimingBar:
    """Stands in for tqdm.trange; records a timestamp at the top of every step."""

    def __init__(self, n, **kwargs):
        self.n = n

    def __iter__(self):
        times = STATE["times"]
        for i in range(self.n):
            times.append(time.perf_counter())
            yield i
        times.append(time.perf_counter())

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass


import tqdm  # noqa: E402  (patched after argv handling, before careless imports it)

tqdm.trange = lambda n, **kwargs: TimingBar(n, **kwargs)

from careless.models.merging.variational import VariationalMergingModel  # noqa: E402

_train_model = VariationalMergingModel.train_model


def _timed_train_model(self, data, steps, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = _train_model(self, data, steps, **kwargs)
    STATE["wall_s"] = time.perf_counter() - started
    if torch.cuda.is_available():
        STATE["peak_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    STATE["history"] = dict(history)
    STATE["jit_compile"] = bool(kwargs.get("jit_compile"))
    STATE["jit_compile_mode"] = kwargs.get("jit_compile_mode")
    STATE["reduce_retracing"] = kwargs.get("reduce_retracing")
    return history


VariationalMergingModel.train_model = _timed_train_model

from careless.careless import main  # noqa: E402

status = "ok"
try:
    main()
except SystemExit as exc:
    if exc.code not in (None, 0):
        status = f"exit {exc.code}"
except BaseException as exc:  # noqa: BLE001 -- record the failure, then re-report it
    import traceback

    traceback.print_exc()
    status = f"{type(exc).__name__}: {exc}"

times = STATE.pop("times")
record = dict(
    STATE,
    status=status,
    torch=torch.__version__,
    step_times_s=[times[i + 1] - times[i] for i in range(len(times) - 1)],
)
with open(BENCH.bench_out, "w") as handle:
    json.dump(record, handle)
print(f"[bench] wrote {BENCH.bench_out} status={status}", flush=True)
