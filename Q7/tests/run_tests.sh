#!/bin/bash
# ---------------------------------------------------------------------------
# run_tests.sh -- correctness suite for Q7 (server log analytics).
#
# Two kinds of check:
#   1. against tests/small.expected, whose values were computed BY HAND from
#      tests/small.in (the PDF supplies no sample data for Q7)
#   2. MPI output vs the sequential reference, at P = 1, 2, 4, 8
#
# Outputs are compared with diff, byte for byte. Averages are printed to a
# fixed 6 decimal places by both programs, and every partial sum is an exact
# integer until the final division, so the comparison is exact.
#
# PASS/FAIL is printed by this script only; neither program under test has its
# own output format altered.
#
# Usage: bash tests/run_tests.sh          (from the Q7 project root)
# Env:   MPIRUN_FLAGS   extra mpirun flags, e.g. "--oversubscribe"
#        PROCS          process counts, default "1 2 4 8"
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="$ROOT/src/sequential"
MPI="$ROOT/src/q7_mpi"
GEN="$ROOT/tools/gen_log"
WORK="$(mktemp -d)"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"

pass=0; fail=0
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$SEQ" ] || [ ! -x "$MPI" ] || [ ! -x "$GEN" ]; then
    echo "ERROR: binaries not found. Run 'make' first." >&2
    exit 2
fi

# --- against the hand-computed expected output -----------------------------
check_expected() {
    local name="$1" input="$2" expected="$3"
    local ok=1 detail=""
    "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null
    diff -q "$WORK/seq.out" "$expected" >/dev/null 2>&1 || { ok=0; detail=" seq"; }
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/m.$p.out" 2>/dev/null
        diff -q "$WORK/m.$p.out" "$expected" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-44s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-44s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- file-based case: MPI vs sequential ------------------------------------
check_file() {
    local name="$1" input="$2"
    if ! "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null; then
        printf '  %-44s FAIL (sequential failed)\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/m.$p.out" 2>/dev/null
        diff -q "$WORK/seq.out" "$WORK/m.$p.out" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-44s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-44s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- generated case: MPI vs sequential -------------------------------------
# Both binaries share log_io.hpp, so --gen yields bit-identical logs.
check_gen() {
    local name="$1" N="$2" K="$3" S="$4" seed="${5:-7}"
    if ! "$SEQ" --gen "$N" "$K" "$S" "$seed" > "$WORK/seq.out" 2>/dev/null; then
        printf '  %-44s FAIL (sequential failed)\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" --gen "$N" "$K" "$S" "$seed" \
            > "$WORK/m.$p.out" 2>/dev/null </dev/null
        diff -q "$WORK/seq.out" "$WORK/m.$p.out" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-44s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-44s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- malformed input must fail cleanly -------------------------------------
check_rejects() {
    local name="$1" content="$2"
    printf '%b' "$content" > "$WORK/bad.in"
    "$SEQ" < "$WORK/bad.in" > "$WORK/bad.out" 2> "$WORK/bad.err"
    local code=$?
    if [ $code -ne 0 ] && [ -s "$WORK/bad.err" ]; then
        printf '  %-44s PASS\n' "$name"; pass=$((pass+1))
    else
        printf '  %-44s FAIL (expected non-zero exit + message)\n' "$name"; fail=$((fail+1))
    fi
}

echo "Q7 correctness suite  (process counts: $PROCS)"
echo "-----------------------------------------------------------------"

echo "  [hand-computed expected output]"
check_expected "small log, all fields + both tie-breaks" \
               "$ROOT/tests/small.in" "$ROOT/tests/small.expected"

echo "  [structural edge cases -- MPI vs sequential]"
printf '0 5 3\n'                          > "$WORK/empty.in"
check_file "N=0 (no records)"             "$WORK/empty.in"
printf '1 3 1\n100 7 9 1 200 50 500\n'    > "$WORK/one.in"
check_file "N=1 (single record)"          "$WORK/one.in"
printf '3 5 1\n0 1 1 1 200 10 100\n1 1 1 2 200 20 200\n2 1 1 3 200 30 300\n' > "$WORK/few.in"
check_file "N=3 < P (idle ranks at P=4,8)" "$WORK/few.in"
printf '4 0 2\n0 1 1 1 200 10 100\n1 2 2 2 404 20 200\n2 1 1 3 500 30 300\n3 2 2 4 301 40 400\n' > "$WORK/k0.in"
check_file "K=0 (no top-K rows)"          "$WORK/k0.in"
printf '2 99 2\n0 1 1 1 200 10 100\n1 2 2 2 200 20 200\n' > "$WORK/kbig.in"
check_file "K > distinct IDs"             "$WORK/kbig.in"
printf '3 2 1\n0 5 5 1 200 10 100\n1 5 5 2 200 20 200\n2 5 5 3 200 30 300\n' > "$WORK/one_srv.in"
check_file "single server, single endpoint" "$WORK/one_srv.in"
printf '3 2 1\n0 1 1 1 200 10 100\n60 2 2 2 200 20 200\n120 3 3 3 200 30 300\n' > "$WORK/tie.in"
check_file "three-way busiest-interval tie" "$WORK/tie.in"
printf '4 2 2\n0 1 1 1 100 10 100\n1 1 1 2 302 20 200\n2 2 2 3 451 30 300\n3 2 2 4 503 40 400\n' > "$WORK/status.in"
check_file "1xx status (outside 2xx-5xx buckets)" "$WORK/status.in"

echo "  [generated cases -- MPI vs sequential]"
check_gen "N=1000  K=5  S=8"              1000    5    8
check_gen "N=10000 K=10 S=64"             10000   10   64
check_gen "N=10001 K=10 S=64 (N odd)"     10001   10   64
check_gen "N=99991 K=7  S=13 (N prime)"   99991   7    13
check_gen "N=50000 K=3  S=1 (one server)" 50000   3    1
check_gen "N=50000 K=20 S=500 (many IDs)" 50000   20   500
check_gen "N=200000 K=10 S=32"            200000  10   32
check_gen "N=500000 K=10 S=128"           500000  10   128

echo "  [input validation]"
check_rejects "rejects truncated record"  '2 1 1\n0 1 1 1 200 10 100\n5 2\n'
check_rejects "rejects overlong input"    '1 1 1\n0 1 1 1 200 10 100\n0 1 1 1 200 10 100\n'
check_rejects "rejects missing header"    '\n'

echo "-----------------------------------------------------------------"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
