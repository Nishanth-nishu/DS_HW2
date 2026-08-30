#!/bin/bash
# ---------------------------------------------------------------------------
# run_bench.sh -- benchmark harness for Q6 (connected components).
#
# Writes benchmark/results.csv, one row per (dataset, P, trial), parsed from
# the "[csv]" line q6_mpi emits on stderr under --stats.
#
# Datasets are generated once into benchmark/data/ by ./tools/gen_graph (C++)
# and reused across runs.
#
# Timing definitions (max across ranks):
#   t_input    read on rank 0 + Scatterv distribution
#   t_compute  local union-find work, summed over all rounds
#   t_comm     time inside MPI_Allreduce, summed over all rounds
#   t_algo     t_compute + t_comm    <- headline parallel figure
#
# Usage: bash benchmark/run_bench.sh [--quick]     (from the Q6 project root)
# Env:   MPIRUN_FLAGS, PROCS (default "1 2 4 8"), TRIALS (default 5),
#        BENCH_DATASETS (override the dataset list)
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MPI="$ROOT/src/q6_mpi"
GEN="$ROOT/tools/gen_graph"
OUT="$ROOT/benchmark/results.csv"
DATA="${DATA_DIR:-$ROOT/benchmark/data}"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"
TRIALS="${TRIALS:-5}"

if [ ! -x "$MPI" ] || [ ! -x "$GEN" ]; then
    echo "ERROR: binaries not built. Run 'make' first." >&2
    exit 2
fi

# Columns: label  vertices  edges  topology  seed
#
# The first four are the Small/Medium/Large/Very large sizes required by
# report_format.md, all inside the assignment constraints (V<=1e5, E<=1e6).
# The topology rows study convergence behaviour. The xl_* rows are BEYOND the
# constraints and exist only to locate the size at which parallel speed-up
# appears -- label them clearly as out-of-constraint in the report.
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
DATASETS="${BENCH_DATASETS:-$DATASETS}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

if [ -f "$OUT" ]; then
    ts=$(date +%Y%m%d_%H%M%S)
    mv "$OUT" "$ROOT/benchmark/results_$ts.csv"
    echo "existing results preserved as benchmark/results_$ts.csv" >&2
fi

mkdir -p "$DATA"
echo "dataset,V,E,topology,P,trial,rounds,t_input,t_compute,t_comm,t_total" > "$OUT"

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
        if [ "$E" -gt 0 ]; then
            "$GEN" -v "$V" -e "$E" -t "$topo" -s "$seed" -c 16 -o "$f"
        else
            "$GEN" -v "$V" -t "$topo" -s "$seed" -c 16 -o "$f"
        fi
        if [ ! -s "$f" ]; then
            echo "  ERROR: could not generate $f -- skipping $label" >&2
            continue
        fi
    fi

    for p in $PROCS; do
        for trial in $(seq 1 "$TRIALS"); do
            # stdin is redirected from "$f", so mpirun cannot consume the
            # dataset list this loop is reading from.
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
echo "now run:  ./tools/analyze" >&2
