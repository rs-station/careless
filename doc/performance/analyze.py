#!/usr/bin/env python
"""
Turn a sweeps.sh output tree into the tables in README.md and width.md.

    python doc/performance/analyze.py <outdir>

Reads <outdir>/{modes,batches,width}/*.json and prints whichever sweeps are
present. Nothing here is specific to one GPU: every threshold is derived from the
data rather than hard-coded, so the same script produces the comparable tables on
different hardware.
"""
import glob
import json
import math
import os
import statistics as stats
import sys

#: Steps to drop before summarizing. Step 0 carries compilation; the CUDA-graphs
#: modes spend another step or two on warm-up and graph recording.
WARMUP = 6

#: Cells that produced a JSON but no usable timing, keyed by sweep. Reported at
#: the end so a partly-failed sweep cannot be mistaken for a complete one.
BAD = {}

MODE_ORDER = ["eager", "default", "reduce-overhead", "max-autotune",
              "max-autotune-no-cudagraphs", "max-autotune_dynamic",
              "max-autotune-no-cudagraphs_dynamic"]


def load(outdir, sub):
    out = {}
    for path in sorted(glob.glob(os.path.join(outdir, sub, "*.json"))):
        tag = os.path.basename(path)[:-5]
        try:
            rec = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        steps = rec.get("step_times_s") or []
        if len(steps) > WARMUP:
            out[tag] = rec
        else:
            BAD.setdefault(sub, []).append((tag, rec.get("status", "no steps recorded")))
    return out


def steady(rec):
    return rec["step_times_s"][WARMUP:]


def ms(rec):
    return stats.median(steady(rec)) * 1e3


def worst_dev(ref_hist, hist, keys=("Loss", "NLL", "F KLDiv", "Grad Norm")):
    worst, where = 0.0, None
    for key in keys:
        for a, b in zip(ref_hist.get(key, []), hist.get(key, [])):
            if math.isnan(a) or math.isnan(b):
                continue
            d = abs(a - b) / max(abs(a), 1e-30)
            if d > worst:
                worst, where = d, key
    return worst, where


def rule(title):
    print(f"\n{title}\n" + "-" * len(title))


def section_meta(outdir):
    path = os.path.join(outdir, "meta.json")
    if os.path.exists(path):
        m = json.load(open(path))
        rule("Environment")
        for k in ("gpu", "capability", "total_mem_mib", "torch", "cuda", "triton",
                  "careless", "commit", "host"):
            if m.get(k) is not None:
                print(f"  {k:14s} {m[k]}")


def section_modes(outdir):
    runs = load(outdir, "modes")
    if not runs:
        return
    rule("torch.compile modes  (median of steps %d+, one run each)" % WARMUP)
    base = runs.get("eager")
    order = [m for m in MODE_ORDER if m in runs] + [m for m in runs if m not in MODE_ORDER]
    print(f"{'mode':34s} {'ms/step':>9s} {'sd':>7s} {'speedup':>8s} "
          f"{'peak MiB':>9s} {'memory':>7s} {'compile s':>10s}")
    for m in order:
        r = runs[m]
        t = ms(r)
        sd = stats.stdev(steady(r)) * 1e3 if len(steady(r)) > 1 else 0.0
        sp = f"{ms(base)/t:7.2f}x" if base else "      --"
        mem = f"{base['peak_mib']/r['peak_mib']:6.2f}x" if base and r.get("peak_mib") else "     --"
        print(f"{m:34s} {t:9.2f} {sd:7.2f} {sp} {r.get('peak_mib', float('nan')):9.0f} "
              f"{mem} {r['step_times_s'][0]:10.1f}")
    if base:
        rule("Equivalence vs eager, worst relative deviation over the run")
        for m in order:
            if m == "eager":
                continue
            w, key = worst_dev(base["history"], runs[m]["history"])
            print(f"  {m:34s} {w:9.2e}   (worst metric: {key})")


def section_batches(outdir):
    runs = load(outdir, "batches")
    if not runs:
        return
    nbs = sorted({int(t.split("_nb")[1]) for t in runs if "_nb" in t})
    rule("--num-batches  (gradient accumulation)")
    print(f"{'nb':>4s} | {'eager ms':>9s} {'eager MiB':>10s} | {'jit ms':>9s} {'jit MiB':>9s} "
          f"| {'speedup':>8s} | {'eager cost':>11s} {'jit cost':>9s} {'mem saved':>10s}")
    e0 = j0 = em0 = jm0 = None
    for nb in nbs:
        e, j = runs.get(f"eager_nb{nb}"), runs.get(f"jit_nb{nb}")
        if not (e and j):
            continue
        et, jt = ms(e), ms(j)
        if e0 is None:
            e0, j0, em0, jm0 = et, jt, e["peak_mib"], j["peak_mib"]
        print(f"{nb:4d} | {et:9.1f} {e['peak_mib']:10.0f} | {jt:9.1f} {j['peak_mib']:9.0f} "
              f"| {et/jt:7.2f}x | {et/e0:10.2f}x {jt/j0:8.2f}x {jm0/j['peak_mib']:9.2f}x")
    print("\n  'cost' is step time relative to --num-batches=1 on the same path:")
    print("  accumulation is cheap exactly when the compiled column stays near 1.00x.")

    ref = runs.get("eager_nb1")
    if ref:
        rule("Equivalence vs eager --num-batches=1")
        for tag in sorted(runs, key=lambda t: (t.split("_nb")[0], int(t.split("_nb")[1]))):
            w, key = worst_dev(ref["history"], runs[tag]["history"])
            print(f"  {tag:22s} {w:9.2e}   (worst metric: {key})")


def section_width(outdir):
    runs = load(outdir, "width")
    if not runs:
        return
    ws = sorted({int(t.split("_w")[1]) for t in runs if "_w" in t})
    rule("--mlp-width")
    print(f"{'width':>5s} | {'eager ms':>9s} {'eager MiB':>10s} | {'jit ms':>9s} {'jit MiB':>9s} "
          f"| {'speedup':>8s} {'memory':>7s} | {'compile s':>10s}")
    table = {}
    for w in ws:
        e, j = runs.get(f"eager_w{w}"), runs.get(f"jit_w{w}")
        if not (e and j):
            continue
        et, jt = ms(e), ms(j)
        table[w] = (et, jt, e["peak_mib"], j["peak_mib"])
        print(f"{w:5d} | {et:9.1f} {e['peak_mib']:10.0f} | {jt:9.1f} {j['peak_mib']:9.0f} "
              f"| {et/jt:7.2f}x {e['peak_mib']/j['peak_mib']:6.2f}x | {j['step_times_s'][0]:10.1f}")

    # Quantization: how far above a straight line between the nearest "round"
    # widths does each in-between width sit? The quantum is derived from the data,
    # not assumed: try each candidate and report which one the timings support.
    rule("Tile quantization: excess over a line between neighbouring multiples of Q")
    print("  A large compiled excess at Q means the compiled path pays a whole extra")
    print("  tile for crossing a multiple of Q. On an RTX A6000 / torch 2.13 the")
    print("  answer was Q=16 (+41% at width 17). Re-derive it here; do not assume it.")
    for q in (8, 16, 32):
        anchors = [w for w in table if w % q == 0]
        if len(anchors) < 2:
            continue
        rows = []
        for w in sorted(table):
            lo = max((a for a in anchors if a < w), default=None)
            hi = min((a for a in anchors if a > w), default=None)
            if lo is None or hi is None or w % q == 0:
                continue
            frac = (w - lo) / (hi - lo)
            for idx, label in ((0, "eager"), (1, "compiled")):
                base = table[lo][idx] + (table[hi][idx] - table[lo][idx]) * frac
                rows.append((w, label, 100 * (table[w][idx] / base - 1)))
        if not rows:
            continue
        print(f"\n  Q = {q}")
        for w in sorted({r[0] for r in rows}):
            cells = {lab: v for ww, lab, v in rows if ww == w}
            print(f"    width {w:3d}   eager {cells.get('eager', float('nan')):+6.1f}%   "
                  f"compiled {cells.get('compiled', float('nan')):+6.1f}%")

    rule("Equivalence, compiled vs eager at the same width")
    for w in ws:
        e, j = runs.get(f"eager_w{w}"), runs.get(f"jit_w{w}")
        if not (e and j):
            continue
        dev, key = worst_dev(e["history"], j["history"])
        flag = "   <-- check this" if dev > 1e-5 else ""
        print(f"  width {w:4d}  {dev:9.2e}   (worst metric: {key}){flag}")
    print("\n  train_model sets float32_matmul_precision('high') unconditionally, so")
    print("  TF32 is live for eager matmuls while inductor may pick non-TF32 kernels.")
    print("  On an A100 TF32 is far more capable than on an RTX A6000, so this is the")
    print("  most likely place for the two paths to part company. Anything above ~1e-5")
    print("  deserves an explanation before the timing numbers are trusted.")


def section_failures():
    if not BAD:
        return
    rule("Cells with no usable timing")
    for sub, items in BAD.items():
        for tag, status in items:
            print(f"  {sub}/{tag:38s} {str(status)[:90]}")
    print("\n  A ValueError on max-autotune_dynamic_guard is the expected result:")
    print("  that cell exercises the guard against the CUDA-graphs + dynamic-shapes")
    print("  segfault. Anything else here means a cell is missing from the tables")
    print("  above -- check its .log before comparing against another machine.")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    outdir = sys.argv[1]
    section_meta(outdir)
    section_modes(outdir)
    section_batches(outdir)
    section_width(outdir)
    section_failures()
    print()


if __name__ == "__main__":
    main()
