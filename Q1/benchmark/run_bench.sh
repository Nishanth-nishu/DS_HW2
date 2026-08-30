#!/bin/bash
# ---------------------------------------------------------------------------
# run_bench.sh -- benchmark harness for Q1 (Row-Row matrix multiplication).
#
# Writes benchmark/results.csv, one row per (size, P, trial), parsed from the
# "[csv]" line that q1_mpi emits on stderr under --stats.
#
# Matrices are GENERATED in-process via --gen rather than read from disk. The
# PDF permits the master to read OR generate A and B, and generating avoids
# letting multi-megabyte text parsing dominate a measurement that is meant to
# be about multiplication. --no-output likewise suppresses printing C.
#
# Timing definitions (max across ranks, since a parallel phase ends when its
# slowest rank ends):
#   t_dist     Bcast(dims) + Scatterv(A) + Bcast(B)
#   t_compute  the local Row-Row multiply
#   t_gather   Gatherv(C)
#   t_comm     t_dist + t_gather
#   t_algo     t_compute + t_comm    <- headline parallel figure
#
# Usage: bash benchmark/run_bench.sh [--quick]      (from the Q1 project root)
# Env:   MPIRUN_FLAGS, PROCS (default "1 2 4 8"), TRIALS (default 5),
#        SEED (default 2024), BENCH_SIZES (override the size list)
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MPI="$ROOT/src/q1_mpi"
OUT="$ROOT/benchmark/results.csv"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-2024}"

if [ ! -x "$MPI" ]; then
    echo "ERROR: $MPI not built. Run 'make' first." >&2
    exit 2
fi

# Columns: label  m  n  p
#
# The first four are the Small/Medium/Large/Very large rows required by
# report_format.md. The skew_* entries vary the SHAPE at roughly constant
# work, to expose how the cost of broadcasting B (n*p elements, independent
# of P) changes relative to the computation (m*n*p/P).
SIZES="
small       256   256   256
medium      512   512   512
large      1024  1024  1024
verylarge  1500  1500  1500
skew_tall  4096   128   128
skew_wide   128  1024  1024
uneven     1000   999   997
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

echo "size,m,n,p,P,trial,t_dist,t_compute,t_gather,t_total" > "$OUT"

echo "$SIZES" | while read -r label m n p; do
    [ -z "${label:-}" ] && continue
    case "$label" in \#*) continue;; esac
    if [ "$QUICK" -eq 1 ]; then
        case "$label" in verylarge|skew_tall) continue;; esac
    fi

    for np in $PROCS; do
        for trial in $(seq 1 "$TRIALS"); do
            # </dev/null is ESSENTIAL: mpirun reads stdin, and this loop is fed
            # by a pipe ("echo $SIZES | while read"). Without the redirect,
            # mpirun swallows the remaining size lines and every size after the
            # first is silently skipped.
            line=$(mpirun $FLAGS -np "$np" "$MPI" --gen "$m" "$n" "$p" "$SEED" \
                   --stats --no-output 2>&1 >/dev/null </dev/null \
                   | grep '^\[csv\]' | sed 's/^\[csv\] //')
            if [ -z "$line" ]; then
                echo "  WARNING: no timing for $label P=$np trial=$trial" >&2
                continue
            fi
            # [csv] P,m,n,p,t_dist,t_compute,t_gather,t_total
            td=$(echo "$line" | cut -d, -f5)
            tc=$(echo "$line" | cut -d, -f6)
            tg=$(echo "$line" | cut -d, -f7)
            tt=$(echo "$line" | cut -d, -f8)
            echo "$label,$m,$n,$p,$np,$trial,$td,$tc,$tg,$tt" >> "$OUT"
        done
        echo "  $label ${m}x${n}x${p}  P=$np done" >&2
    done
done

echo "" >&2
echo "results written to $OUT" >&2
echo "now run:  ./tools/analyze" >&2
