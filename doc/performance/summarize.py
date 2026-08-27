#!/usr/bin/env python
"""
Summarize the JSON records written by bench_compile_mode.py.

    python doc/performance/summarize.py runs/*.json

Reports the mean step time excluding warm-up, the speedup against the run whose
mode is "eager", peak memory, and the largest relative deviation of every logged
metric from the eager run.
"""
import json
import math
import statistics
import sys

#: Steps to drop before averaging. Step 0 contains compilation; the CUDA-graphs
#: modes spend another step or two on warm-up and graph recording.
WARMUP = 6


def label(record):
    if not record.get("jit_compile"):
        return "eager"
    mode = record.get("jit_compile_mode") or "default"
    return mode + ("+dynamic" if record.get("reduce_retracing") else "")


def main(paths):
    runs = {}
    for path in paths:
        record = json.load(open(path))
        runs.setdefault(label(record), []).append(record)

    order = [k for k in ("eager", "default", "reduce-overhead", "max-autotune",
                         "max-autotune-no-cudagraphs") if k in runs]
    order += sorted(k for k in runs if k not in order)

    def steady(name):
        return [t for r in runs[name] for t in r["step_times_s"][WARMUP:]]

    base = statistics.mean(steady("eager")) if "eager" in runs else None
    base_mem = (statistics.mean([r["peak_mib"] for r in runs["eager"]])
                if "eager" in runs else None)

    print(f"{'mode':30s} {'n':>3s} {'ms/step':>9s} {'sd':>6s} {'speedup':>8s} "
          f"{'peak MiB':>9s} {'mem':>7s} {'step0 s':>8s}")
    for name in order:
        times = steady(name)
        mean = statistics.mean(times) * 1e3
        sd = statistics.stdev(times) * 1e3 if len(times) > 1 else 0.0
        mem = statistics.mean([r["peak_mib"] for r in runs[name] if "peak_mib" in r] or [float("nan")])
        step0 = statistics.mean([r["step_times_s"][0] for r in runs[name]])
        speedup = f"{base * 1e3 / mean:7.2f}x" if base else "      --"
        memx = f"{base_mem / mem:6.2f}x" if base_mem else "     --"
        print(f"{name:30s} {len(runs[name]):3d} {mean:9.2f} {sd:6.2f} {speedup} "
              f"{mem:9.0f} {memx} {step0:8.1f}")

    if "eager" not in runs:
        return
    reference = runs["eager"][0]["history"]
    keys = [k for k in reference if k != "NLL_val"]
    print("\nmax relative deviation from eager")
    print(f"{'mode':30s} " + " ".join(f"{k:>12s}" for k in keys))
    for name in order:
        if name == "eager":
            continue
        history = runs[name][0]["history"]
        cells = []
        for key in keys:
            worst = 0.0
            for a, b in zip(reference[key], history.get(key, [])):
                if math.isnan(a) or math.isnan(b):
                    continue
                worst = max(worst, abs(a - b) / max(abs(a), 1e-30))
            cells.append(worst)
        print(f"{name:30s} " + " ".join(f"{c:12.2e}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1:])
