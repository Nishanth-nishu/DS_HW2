#!/bin/bash
# ---------------------------------------------------------------------------
# run_bench.sh -- benchmark harness for Q6.
#
# Produces benchmark/results.csv with one row per (dataset, P, trial).
# Timings come from the program's own --stats instrumentation, which reports
# the MAX across ranks (a parallel phase ends when its slowest rank ends).
#
# Timing definitions
#   t_input    read on rank 0 + Scatterv distribution
#   t_compute  local union-find work, summed over all rounds
#   t_comm     time inside MPI_Allreduce, summed over all rounds
#   t_total    whole program, MPI_Init to just before MPI_Finalize
#   t_algo     t_compute + t_comm  <- the headline parallel figure
#
# Usage:  bash benchmark/run_bench.sh [--quick]
# Env:    MPIRUN_FLAGS  extra mpirun flags (e.g. "--oversubscribe")
#         PROCS         process counts, default "1 2 4 8"
#         TRIALS        repetitions per configuration, default 5
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MPI="$ROOT/src/q6_mpi"
GEN="$ROOT/tools/gen_graph.py"
OUT="$ROOT/benchmark/results.csv"
DATA="${DATA_DIR:-$ROOT/benchmark/data}"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"
TRIALS="${TRIALS:-5}"

mkdir -p "$DATA"

# --------------------------------------------------------------------------
# Datasets.  Columns: label  vertices  edges  topology  seed
#
# The first four rows are the Small/Medium/Large/Very large sizes required by
# report_format.md and stay inside the assignment constraints
# (V <= 1e5, E <= 1e6).
#
# The remaining rows are additional studies, clearly separated:
#   *_path / *_shuffled  topology effect on convergence round count
#   xl_*                 BEYOND the assignment constraints, included only to
#                        locate the size at which parallel speedup appears.
#                        Label these clearly as out-of-constraint in the report.
# --------------------------------------------------------------------------
DATASETS="
small       10000   100000    random        101
medium      50000   500000    random        102
large      100000  1000000    random        103
verylarge  100000  1000000    cluster       104
path       100000        0    path          105
shuffled   100000        0    shuffled      106
chain      100000        0    chain-blocks  107
xl_1       500000  5000000    random        108
xl_2      1000000 10000000    random        109
"

# Allow a caller to substitute the dataset list, e.g. for a fast smoke test:
#   BENCH_DATASETS="tiny 2000 10000 random 1" bash benchmark/run_bench.sh
DATASETS="${BENCH_DATASETS:-$DATASETS}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

echo "dataset,V,E,topology,P,trial,rounds,t_input,t_compute,t_comm,t_total" > "$OUT"

gen_shuffled() {   # a path graph whose vertex IDs are randomly permuted
    python3 - "$1" "$2" "$3" <<'PY'
import random, sys
out, V, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
random.seed(seed)
perm = list(range(V)); random.shuffle(perm)
adj = [[] for _ in range(V)]
for i in range(V-1):
    u, v = perm[i], perm[i+1]
    adj[u].append(v); adj[v].append(u)
with open(out, "w") as f:
    f.write(f"{V}\n")
    for a in adj:
        f.write(f"{len(a)}" + (" " + " ".join(map(str, a)) if a else "") + "\n")
PY
}

echo "$DATASETS" | while read -r label V E topo seed; do
    [ -z "${label:-}" ] && continue
    case "$label" in \#*) continue;; esac
    if [ "$QUICK" -eq 1 ]; then
        case "$label" in xl_*) continue;; esac
    fi

    f="$DATA/$label.in"
    mkdir -p "$DATA"
    if [ ! -f "$f" ]; then
        echo "generating $label (V=$V E=$E topology=$topo seed=$seed) ..." >&2
        if [ "$topo" = "shuffled" ]; then
            gen_shuffled "$f" "$V" "$seed"
        elif [ "$E" -gt 0 ]; then
            python3 "$GEN" -v "$V" -e "$E" -t "$topo" -s "$seed" -c 16 -o "$f"
        else
            python3 "$GEN" -v "$V" -t "$topo" -s "$seed" -c 16 -o "$f"
        fi
        if [ ! -s "$f" ]; then
            echo "  ERROR: could not generate $f -- skipping $label" >&2
            continue
        fi
    fi

    for p in $PROCS; do
        for trial in $(seq 1 "$TRIALS"); do
            line=$(mpirun $FLAGS -np "$p" "$MPI" --stats < "$f" 2>&1 >/dev/null \
                   | grep '^\[csv\]' | sed 's/^\[csv\] //')
            if [ -z "$line" ]; then
                echo "  WARNING: no timing from $label P=$p trial=$trial" >&2
                continue
            fi
            # [csv] P,V,rounds,t_input,t_compute,t_comm,t_total
            rounds=$(echo "$line" | cut -d, -f3)
            ti=$(echo "$line" | cut -d, -f4)
            tc=$(echo "$line" | cut -d, -f5)
            tm=$(echo "$line" | cut -d, -f6)
            tt=$(echo "$line" | cut -d, -f7)
            echo "$label,$V,$E,$topo,$p,$trial,$rounds,$ti,$tc,$tm,$tt" >> "$OUT"
        done
        echo "  $label P=$p done" >&2
    done
done

echo "" >&2
echo "results written to $OUT" >&2
echo "now run:  python3 benchmark/analyze.py" >&2
