#!/bin/bash
# ---------------------------------------------------------------------------
# run_tests.sh -- correctness suite for Q6 (connected components).
#
# For every case: run the sequential reference, run the MPI program at
# P = 1, 2, 4, 8, and diff the outputs.
#
# PASS/FAIL is printed by this script only; neither program under test has its
# own output format altered.
#
# Test graphs are produced by ./tools/gen_graph (C++), so every case is
# reproducible from its seed.
#
# Usage: bash tests/run_tests.sh          (from the Q6 project root)
# Env:   MPIRUN_FLAGS   extra mpirun flags, e.g. "--oversubscribe"
#        PROCS          process counts, default "1 2 4 8"
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="$ROOT/src/sequential"
MPI="$ROOT/src/q6_mpi"
GEN="$ROOT/tools/gen_graph"
WORK="$(mktemp -d)"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"

pass=0; fail=0
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$SEQ" ] || [ ! -x "$MPI" ] || [ ! -x "$GEN" ]; then
    echo "ERROR: binaries not found. Run 'make' first." >&2
    exit 2
fi

check() {
    local name="$1" input="$2"
    if ! "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null; then
        printf '  %-38s FAIL (sequential failed)\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/mpi.$p.out" 2>/dev/null
        diff -q "$WORK/seq.out" "$WORK/mpi.$p.out" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-38s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-38s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

check_gen() {
    local name="$1"; shift
    "$GEN" "$@" -o "$WORK/case.in" || {
        printf '  %-38s FAIL (generator failed)\n' "$name"; fail=$((fail+1)); return; }
    check "$name" "$WORK/case.in"
}

echo "Q6 correctness suite  (process counts: $PROCS)"
echo "-----------------------------------------------------"

# The assignment sample, against the PDF's published expected output.
"$SEQ" < "$ROOT/tests/sample.in" > "$WORK/s.out"
if diff -q "$WORK/s.out" "$ROOT/tests/sample.expected" >/dev/null; then
    echo "  sequential vs PDF expected output    PASS"; pass=$((pass+1))
else
    echo "  sequential vs PDF expected output    FAIL"; fail=$((fail+1))
fi
check "A. PDF sample" "$ROOT/tests/sample.in"

check_gen "B. single vertex"             -v 1  -t isolated
check_gen "C. no edges"                  -v 50 -t isolated
check_gen "D. fully disconnected pairs"  -v 64 -t disconnected
check_gen "E. one component (path)"      -v 64 -t path
check_gen "F. multiple components"       -v 64 -t cluster -c 5 -e 200
check_gen "G. cross-process bridges"     -v 64 -t chain-blocks -c 8
check_gen "H. uneven split (V=97)"       -v 97 -t cluster -c 7 -e 300
check_gen "I. star (load imbalance)"     -v 64 -t star
check_gen "J. asymmetric adjacency"      -v 64 -t path --asymmetric
check_gen "K. V < P (idle ranks)"        -v 3  -t isolated
check_gen "L. random 5k/20k"             -v 5000 -e 20000 -s 11
check_gen "M. sparse (many components)"  -v 5000 -e 1000 -s 12
check_gen "N. long path 20k"             -v 20000 -t path
check_gen "O. shuffled path 20k (worst)" -v 20000 -t shuffled -s 9
check_gen "P. random 50k/400k"           -v 50000 -e 400000 -s 13

echo "-----------------------------------------------------"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
