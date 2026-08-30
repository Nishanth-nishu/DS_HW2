#!/usr/bin/env python3
"""
analyze.py -- turn benchmark/results.csv into the tables and plots the report
needs.

Produces on stdout, in the layout of report_format.md:
  * runtime table   (median t_algo, seconds)
  * speed-up table  S(P) = T1 / TP
  * efficiency table E(P) = S(P) / P
  * communication vs computation table (% of t_algo spent in MPI_Allreduce)
  * convergence round counts

And writes to benchmark/plots/:
  speedup.png, efficiency.png, comm_fraction.png

The reported time per configuration is the MEDIAN over trials: it is robust
against a single slow run caused by another job sharing the node.

T1 is the MPI binary at P=1, not the sequential program, so the speed-up
measures parallel scaling of one implementation rather than a comparison of
two different programs.

Usage: python3 benchmark/analyze.py [results.csv]
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.csv")
PLOTS = os.path.join(HERE, "plots")


def load(path):
    if not os.path.exists(path):
        sys.exit(f"no results file at {path}\nRun benchmark/run_bench.sh first.")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["P"] = int(r["P"])
                r["rounds"] = int(r["rounds"])
                for k in ("t_input", "t_compute", "t_comm", "t_total"):
                    r[k] = float(r[k])
                r["t_algo"] = r["t_compute"] + r["t_comm"]
            except (ValueError, KeyError):
                continue
            rows.append(r)
    if not rows:
        sys.exit("results file contains no usable rows")
    return rows


def aggregate(rows):
    """(dataset, P) -> median of each metric across trials."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["dataset"], r["P"])
        for m in ("t_input", "t_compute", "t_comm", "t_total", "t_algo"):
            buckets[key][m].append(r[m])
        buckets[key]["rounds"].append(r["rounds"])
    out = {}
    for key, met in buckets.items():
        out[key] = {m: statistics.median(v) for m, v in met.items()}
        out[key]["trials"] = len(met["t_algo"])
    return out


def table(title, header, body):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([":--"] + [":-:"] * (len(header) - 1)) + "|")
    for row in body:
        print("| " + " | ".join(row) + " |")


def main():
    rows = load(CSV)
    agg = aggregate(rows)

    datasets = []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    procs = sorted({r["P"] for r in rows})

    print("# Q6 Benchmark Analysis")
    print(f"\nSource: `{os.path.relpath(CSV)}`  |  process counts: "
          f"{', '.join(map(str, procs))}  |  statistic: median over trials")
    print("\n`t_algo` = `t_compute` + `t_comm` (excludes file reading and "
          "distribution).")

    # ---- runtime -----------------------------------------------------------
    body = []
    for d in datasets:
        cells = [d]
        for p in procs:
            a = agg.get((d, p))
            cells.append(f"{a['t_algo']:.4f}" if a else "--")
        body.append(cells)
    table("Runtime — median `t_algo` (seconds)",
          ["Input"] + [f"P={p}" for p in procs], body)

    # ---- speed-up ----------------------------------------------------------
    speed = {}
    body = []
    for d in datasets:
        base = agg.get((d, 1))
        cells = [d]
        for p in procs:
            a = agg.get((d, p))
            if a and base and a["t_algo"] > 0:
                s = base["t_algo"] / a["t_algo"]
                speed[(d, p)] = s
                cells.append(f"{s:.2f}")
            else:
                cells.append("--")
        body.append(cells)
    table("Speed-up  S(P) = T1 / TP",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- efficiency --------------------------------------------------------
    body = []
    for d in datasets:
        cells = [d]
        for p in procs:
            s = speed.get((d, p))
            cells.append(f"{100.0 * s / p:.1f}%" if s else "--")
        body.append(cells)
    table("Efficiency  E(P) = S(P) / P",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- communication share ----------------------------------------------
    body = []
    for d in datasets:
        cells = [d]
        for p in procs:
            a = agg.get((d, p))
            if a and a["t_algo"] > 0:
                cells.append(f"{100.0 * a['t_comm'] / a['t_algo']:.1f}%")
            else:
                cells.append("--")
        body.append(cells)
    table("Communication share — `t_comm` as % of `t_algo`",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- rounds ------------------------------------------------------------
    body = []
    for d in datasets:
        cells = [d]
        for p in procs:
            a = agg.get((d, p))
            cells.append(f"{int(a['rounds'])}" if a else "--")
        body.append(cells)
    table("Convergence rounds (deterministic, machine-independent)",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- plots -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available - plots skipped)")
        return

    os.makedirs(PLOTS, exist_ok=True)

    def lineplot(fname, ylabel, title, valfn, ideal=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for d in datasets:
            xs, ys = [], []
            for p in procs:
                v = valfn(d, p)
                if v is not None:
                    xs.append(p); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=d)
        if ideal:
            ax.plot(procs, procs, "k--", alpha=0.4, label="ideal (linear)")
        ax.set_xlabel("Number of MPI processes (P)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(procs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(PLOTS, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    p1 = lineplot("speedup.png", "Speed-up  S(P)",
                  "Q6 Connected Components — Speed-up",
                  lambda d, p: speed.get((d, p)), ideal=True)
    p2 = lineplot("efficiency.png", "Efficiency  E(P)",
                  "Q6 Connected Components — Efficiency",
                  lambda d, p: (speed[(d, p)] / p) if (d, p) in speed else None)
    p3 = lineplot("comm_fraction.png", "t_comm / t_algo",
                  "Q6 — Communication share of algorithm time",
                  lambda d, p: (agg[(d, p)]["t_comm"] / agg[(d, p)]["t_algo"])
                  if (d, p) in agg and agg[(d, p)]["t_algo"] > 0 else None)

    print("\n### Plots written\n")
    for p in (p1, p2, p3):
        print(f"- `{os.path.relpath(p)}`")


if __name__ == "__main__":
    main()
