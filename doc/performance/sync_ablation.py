#!/usr/bin/env python
"""
Ablate the per-step host synchronizations and see whether removing them is worth
anything.  ABLATE is a comma-separated subset of {loop, sampler}.

  loop     - remove `if not torch.isfinite(loss)` and the batched metric .tolist()
             from train_model.  The loop is rebuilt from inspect.getsource with two
             textual edits, so everything else stays byte-identical to the shipped
             version.  History values become garbage; this is a timing probe only.
  sampler  - replace _accept_reject_truncnorm with a single draw plus a clamp
             fallback: no done.all(), so no sync and no dynamo graph break.

Also reports how long the host actually sits blocked in each sync, which is an
upper bound on what removing it could ever save.
"""
import inspect
import json
import os
import re
import sys
import textwrap
import time

import torch

ABLATE = set(filter(None, os.environ.get("ABLATE", "").split(",")))
OUT = os.environ["ABL_OUT"]
JIT = os.environ.get("ABL_JIT", "1") == "1"
WARMUP_STEPS = int(os.environ.get("ABL_WARMUP", "15"))

# NOTE: no timing instrumentation anywhere inside the model. A time.perf_counter()
# call inside a function dynamo is tracing forces a graph break, which perturbs
# exactly what is being measured -- the first version of this script did that and
# cost 3 ms/step and 30x the variance.
import careless.distributions.truncated_normal as tn


def _nosync_sampler(loc, scale, low, high, shape, max_iter=100):
    """One draw plus a clamp fallback: no done.all(), so no sync, no graph break."""
    cand = loc + scale * torch.randn(shape, dtype=loc.dtype, device=loc.device)
    ok = (cand >= low) & (cand <= high)
    return torch.where(ok, cand, loc.expand(shape).clamp(min=low, max=high))


if "sampler" in ABLATE:
    tn._accept_reject_truncnorm = _nosync_sampler

# ------------------------------------------------------------------ loop ablation
from careless.models.merging.variational import VariationalMergingModel

if "loop" in ABLATE:
    src = textwrap.dedent(inspect.getsource(VariationalMergingModel.train_model))
    before = src
    src = src.replace("if not torch.isfinite(loss):", "if False:")
    src = re.sub(
        r"synced = torch\.stack\(\[metrics\[k\] for k in tensor_keys\]\)\.tolist\(\).*",
        "synced = [0.0] * len(tensor_keys)",
        src,
    )
    assert src != before and "if False:" in src and "[0.0] * len" in src
    ns = dict(sys.modules[VariationalMergingModel.__module__].__dict__)
    exec(compile(src, "<train_model ablated>", "exec"), ns)
    VariationalMergingModel.train_model = ns["train_model"]
    print("[abl] train_model rebuilt without the two loop syncs", flush=True)

# ------------------------------------------------------------------- timing bar
STATE = {"times": []}


class Bar:
    def __init__(self, n, **kw):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            STATE["times"].append(time.perf_counter())
            yield i
        STATE["times"].append(time.perf_counter())

    def set_postfix(self, *a, **k):
        pass


import tqdm

tqdm.trange = lambda n, **kw: Bar(n, **kw)

_train = VariationalMergingModel.train_model


def _wrapped(self, data, steps, **kw):
    kw["jit_compile"] = JIT
    torch.cuda.reset_peak_memory_stats()
    # Warm up first so compilation is outside the timed region, then time the whole
    # loop and DRAIN THE QUEUE before stopping the clock. Without the final
    # synchronize, an ablation that removes every sync lets the host run ahead and
    # the per-iteration wall times measure launch speed, not throughput.
    _train(self, data, WARMUP_STEPS, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = _train(self, data, steps, **kw)
    torch.cuda.synchronize()
    STATE["drained_total_s"] = time.perf_counter() - t0
    STATE["drained_steps"] = steps
    STATE["peak_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return h


VariationalMergingModel.train_model = _wrapped

from careless.careless import main

try:
    main()
except SystemExit:
    pass

t = STATE.pop("times")
steps = [t[i + 1] - t[i] for i in range(len(t) - 1)]
json.dump(
    {
        "ablate": sorted(ABLATE),
        "jit": JIT,
        "step_times_s": steps,
        "peak_mib": STATE.get("peak_mib"),
        "drained_total_s": STATE.get("drained_total_s"),
        "drained_steps": STATE.get("drained_steps"),
        "n_steps": len(steps),
    },
    open(OUT, "w"),
)
print(f"[abl] wrote {OUT}", flush=True)
