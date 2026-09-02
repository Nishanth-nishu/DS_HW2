#!/bin/bash
# ---------------------------------------------------------------------------
# run_bench.sh -- benchmark harness for Q7 (server log analytics).
#
# Writes benchmark/results.csv, one row per (size, P, trial), parsed from the
# "[csv]" line q7_mpi emits on stderr under --stats.
#
# Logs are GENERATED in-process via --gen rather than read from disk. A
# 5,000,000-record log is roughly 250 MB of text, and parsing it would dwarf
# the aggregation we are trying to measure. --no-output suppresses the report.
#
# Timing definitions (max across ranks, since a parallel phase ends when its
# slowest rank ends):
#   t_dist     Bcast(header) + Scatterv(records)
#   t_compute  local aggregation
#   t_reduce   Reduce(scalars) + Gatherv(per-key maps)
#   t_comm     t_dist + t_reduce
#   t_algo     t_compute + t_comm    <- headline parallel figure
#
# Usage: bash benchmark/run_bench.sh [--quick]      (from the Q7 project root)
# Env:   MPIRUN_FLAGS, PROCS (default "1 2 4 8"), TRIALS (default 5),
#        SEED (default 2024), BENCH_SIZES (override the size list)
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MPI="$ROOT/src/q7_mpi"
OUT="$ROOT/benchmark/results.csv"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-2024}"

if [ ! -x "$MPI" ]; then
    echo "ERROR: $MPI not built. Run 'make' first." >&2
    exit 2
fi

# Columns: label  N  K  S
#
# The first four are the Small/Medium/Large/Very large rows required by
# report_format.md. The remaining rows vary the KEY CARDINALITY at fixed N:
# the per-key Gatherv cost grows with the number of distinct servers and
# endpoints, so these expose where the merge step starts to matter.
SIZES="
small        100000  10    32
medium      1000000  10    64
large       5000000  10   128
verylarge  10000000  10   128
keys_few    1000000  10     4
keys_many   1000000  10  5000
topk_large  1000000 100   128
"
SIZES="${BENCH_SIZES:-$SIZES}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# Never silently clobber previous results.
if [ -f "$OUT" ]; then
    ts=$(date +%Y%m%d_%H%M%S)
    mv "$OUT" "$ROOT/benchmark/results_$ts.csv"
    echo "existing results preserved as benchmark/results_$ts.csv" >&2
fi

echo "size,N,K,S,P,trial,t_dist,t_compute,t_reduce,t_total" > "$OUT"

echo "$SIZES" | while read -r label N K S; do
    [ -z "${label:-}" ] && continue
    case "$label" in \#*) continue;; esac
    if [ "$QUICK" -eq 1 ]; then
        case "$label" in verylarge|large) continue;; esac
    fi

    for np in $PROCS; do
        for trial in $(seq 1 "$TRIALS"); do
            # </dev/null is ESSENTIAL: mpirun reads stdin, and this loop is fed
            # by a pipe ("echo $SIZES | while read"). Without the redirect,
            # mpirun swallows the remaining size lines and every size after the
            # first is silently skipped.
            line=$(mpirun $FLAGS -np "$np" "$MPI" --gen "$N" "$K" "$S" "$SEED" \
                   --stats --no-output 2>&1 >/dev/null </dev/null \
                   | grep '^\[csv\]' | sed 's/^\[csv\] //')
            if [ -z "$line" ]; then
                echo "  WARNING: no timing for $label P=$np trial=$trial" >&2
                continue
            fi
            # [csv] P,N,K,t_dist,t_compute,t_reduce,t_total
            td=$(echo "$line" | cut -d, -f4)
            tc=$(echo "$line" | cut -d, -f5)
            tr=$(echo "$line" | cut -d, -f6)
            tt=$(echo "$line" | cut -d, -f7)
            echo "$label,$N,$K,$S,$np,$trial,$td,$tc,$tr,$tt" >> "$OUT"
        done
        echo "  $label N=$N K=$K S=$S  P=$np done" >&2
    done
done

echo "" >&2
echo "results written to $OUT" >&2
echo "now run:  ./tools/analyze" >&2
