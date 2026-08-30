#!/usr/bin/env python3
"""
analyze.py -- turn benchmark/results.csv into the tables and plots for the Q1
report, in the layout of report_format.md.

Produces on stdout:
  * runtime table       (median t_algo, seconds)
  * speed-up table      S(P) = T1 / TP
  * efficiency table    E(P) = S(P) / P
  * communication breakdown: t_dist / t_compute / t_gather, and comm as % of
    t_algo -- this is the "communication vs computation" analysis the report
    format asks for

And writes benchmark/plots/{time,speedup,efficiency,comm_fraction}.png

The reported value per configuration is the MEDIAN over trials, which is
robust against a single slow run caused by another job on the node.

T1 is the MPI binary at P=1, not the sequential program, so the speed-up
measures parallel scaling of one implementation rather than comparing two
different programs.

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

METRICS = ("t_dist", "t_compute", "t_gather", "t_total")


def load(path):
    if not os.path.exists(path):
        sys.exit(f"no results file at {path}\nRun benchmark/run_bench.sh first.")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["P"] = int(r["P"])
                for k in METRICS:
                    r[k] = float(r[k])
                r["t_comm"] = r["t_dist"] + r["t_gather"]
                r["t_algo"] = r["t_compute"] + r["t_comm"]
            except (ValueError, KeyError):
                continue
            rows.append(r)
    if not rows:
        sys.exit("results file contains no usable rows")
    return rows


def aggregate(rows):
    b = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["size"], r["P"])
        for m in METRICS + ("t_comm", "t_algo"):
            b[key][m].append(r[m])
    return {k: {m: statistics.median(v) for m, v in met.items()}
            for k, met in b.items()}


def table(title, header, body, note=None):
    print(f"\n### {title}\n")
    if note:
        print(f"{note}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([":--"] + [":-:"] * (len(header) - 1)) + "|")
    for row in body:
        print("| " + " | ".join(row) + " |")


def main():
    rows = load(CSV)
    agg = aggregate(rows)

    sizes, shape = [], {}
    for r in rows:
        if r["size"] not in sizes:
            sizes.append(r["size"])
            shape[r["size"]] = f"{r['m']}x{r['n']}x{r['p']}"
    procs = sorted({r["P"] for r in rows})

    print("# Q1 Benchmark Analysis — Row-Row Matrix Multiplication")
    print(f"\nSource: `{os.path.relpath(CSV)}`  |  process counts: "
          f"{', '.join(map(str, procs))}  |  statistic: median over trials")
    print("\n`t_algo` = `t_compute` + `t_comm`, where "
          "`t_comm` = `t_dist` (Bcast dims + Scatterv A + Bcast B) + "
          "`t_gather` (Gatherv C).")

    table("Matrix sizes", ["Label", "m x n x p"],
          [[s, shape[s]] for s in sizes])

    # ---- runtime -----------------------------------------------------------
    body = []
    for s in sizes:
        cells = [s]
        for p in procs:
            a = agg.get((s, p))
            cells.append(f"{a['t_algo']:.4f}" if a else "--")
        body.append(cells)
    table("Runtime — median `t_algo` (seconds)",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- speed-up ----------------------------------------------------------
    speed = {}
    body = []
    for s in sizes:
        base = agg.get((s, 1))
        cells = [s]
        for p in procs:
            a = agg.get((s, p))
            if a and base and a["t_algo"] > 0:
                v = base["t_algo"] / a["t_algo"]
                speed[(s, p)] = v
                cells.append(f"{v:.2f}")
            else:
                cells.append("--")
        body.append(cells)
    table("Speed-up  S(P) = T1 / TP",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- efficiency --------------------------------------------------------
    body = []
    for s in sizes:
        cells = [s]
        for p in procs:
            v = speed.get((s, p))
            cells.append(f"{100.0 * v / p:.1f}%" if v else "--")
        body.append(cells)
    table("Efficiency  E(P) = S(P) / P",
          ["Input size"] + [f"P={p}" for p in procs], body)

    # ---- communication share ----------------------------------------------
    body = []
    for s in sizes:
        cells = [s]
        for p in procs:
            a = agg.get((s, p))
            if a and a["t_algo"] > 0:
                cells.append(f"{100.0 * a['t_comm'] / a['t_algo']:.1f}%")
            else:
                cells.append("--")
        body.append(cells)
    table("Communication share — `t_comm` as % of `t_algo`",
          ["Input size"] + [f"P={p}" for p in procs], body,
          note="Rising with P is expected: the Bcast of B moves n*p elements "
               "to every process regardless of P, while the computation per "
               "process falls as 1/P.")

    # ---- phase breakdown ---------------------------------------------------
    body = []
    for s in sizes:
        for p in procs:
            a = agg.get((s, p))
            if not a:
                continue
            body.append([s, str(p), f"{a['t_dist']:.4f}",
                         f"{a['t_compute']:.4f}", f"{a['t_gather']:.4f}",
                         f"{a['t_algo']:.4f}"])
    table("Phase breakdown (seconds, median)",
          ["Size", "P", "t_dist", "t_compute", "t_gather", "t_algo"], body)

    # ---- plots -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available - plots skipped)")
        return

    os.makedirs(PLOTS, exist_ok=True)

    def lineplot(fname, ylabel, title, valfn, ideal=False, logy=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in sizes:
            xs, ys = [], []
            for p in procs:
                v = valfn(s, p)
                if v is not None:
                    xs.append(p); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{s} ({shape[s]})")
        if ideal:
            ax.plot(procs, procs, "k--", alpha=0.4, label="ideal (linear)")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Number of MPI processes (P)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(procs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(PLOTS, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    made = [
        lineplot("time.png", "t_algo (s, log scale)",
                 "Q1 — Execution time vs processes",
                 lambda s, p: agg[(s, p)]["t_algo"] if (s, p) in agg else None,
                 logy=True),
        lineplot("speedup.png", "Speed-up  S(P)",
                 "Q1 — Speed-up vs processes",
                 lambda s, p: speed.get((s, p)), ideal=True),
        lineplot("efficiency.png", "Efficiency  E(P)",
                 "Q1 — Efficiency vs processes",
                 lambda s, p: (speed[(s, p)] / p) if (s, p) in speed else None),
        lineplot("comm_fraction.png", "t_comm / t_algo",
                 "Q1 — Communication share of algorithm time",
                 lambda s, p: (agg[(s, p)]["t_comm"] / agg[(s, p)]["t_algo"])
                 if (s, p) in agg and agg[(s, p)]["t_algo"] > 0 else None),
    ]

    print("\n### Plots written\n")
    for path in made:
        print(f"- `{os.path.relpath(path)}`")


if __name__ == "__main__":
    main()
