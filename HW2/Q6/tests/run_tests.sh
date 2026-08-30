#!/bin/bash
# ---------------------------------------------------------------------------
# run_tests.sh -- correctness suite for Q6.
#
# For every test case: run the sequential reference, run the MPI program at
# P = 1,2,4,8, and diff. PASS/FAIL goes to this script's output only; the
# programs' own stdout is never altered.
#
# Usage:   bash tests/run_tests.sh
# Env:     MPIRUN_FLAGS  extra flags (e.g. "--oversubscribe") for oversubscribed
#                        or containerised environments.
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="$ROOT/src/sequential"
MPI="$ROOT/src/q6_mpi"
GEN="$ROOT/tools/gen_graph.py"
WORK="$(mktemp -d)"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"

pass=0; fail=0

trap 'rm -rf "$WORK"' EXIT

check() {
    local name="$1" input="$2"
    "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null
    if [ $? -ne 0 ]; then
        printf '  %-34s SEQ ERROR\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/mpi.$p.out" 2>/dev/null
        if ! diff -q "$WORK/seq.out" "$WORK/mpi.$p.out" > /dev/null 2>&1; then
            ok=0; detail="$detail P=$p"
        fi
    done
    if [ $ok -eq 1 ]; then
        printf '  %-34s PASS\n' "$name"; pass=$((pass+1))
    else
        printf '  %-34s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1))
    fi
}

echo "Q6 correctness suite  (process counts: $PROCS)"
echo "-----------------------------------------------------"

# --- A. the assignment sample, checked against the PDF's expected output ----
"$SEQ" < "$ROOT/tests/sample.in" > "$WORK/s.out"
if diff -q "$WORK/s.out" "$ROOT/tests/sample.expected" > /dev/null; then
    echo "  sequential vs PDF expected output  PASS"; pass=$((pass+1))
else
    echo "  sequential vs PDF expected output  FAIL"; fail=$((fail+1))
fi
check "A. PDF sample" "$ROOT/tests/sample.in"

# --- B..K generated cases --------------------------------------------------
gen() { python3 "$GEN" "$@"; }

gen -v 1  -t isolated                 -o "$WORK/b.in";  check "B. single vertex"            "$WORK/b.in"
gen -v 50 -t isolated                 -o "$WORK/c.in";  check "C. no edges"                 "$WORK/c.in"
gen -v 64 -t disconnected             -o "$WORK/d.in";  check "D. fully disconnected pairs" "$WORK/d.in"
gen -v 64 -t path                     -o "$WORK/e.in";  check "E. one component (path)"     "$WORK/e.in"
gen -v 64 -t cluster -c 5 -e 200      -o "$WORK/f.in";  check "F. multiple components"      "$WORK/f.in"
gen -v 64 -t chain-blocks -c 8        -o "$WORK/g.in";  check "G. cross-process bridges"    "$WORK/g.in"
gen -v 97 -t cluster -c 7 -e 300      -o "$WORK/h.in";  check "H. uneven split (V=97 prime)" "$WORK/h.in"
gen -v 64 -t star                     -o "$WORK/i.in";  check "I. star (load imbalance)"    "$WORK/i.in"
gen -v 64 -t path --asymmetric        -o "$WORK/j.in";  check "J. asymmetric adjacency"     "$WORK/j.in"
gen -v 3  -t isolated                 -o "$WORK/k.in";  check "K. V < P (idle ranks)"       "$WORK/k.in"
gen -v 5000  -e 20000 -t random -s 11 -o "$WORK/l.in";  check "L. random 5k/20k"            "$WORK/l.in"
gen -v 5000  -e 1000  -t random -s 12 -o "$WORK/m.in";  check "M. sparse (many components)" "$WORK/m.in"
gen -v 20000 -t path                  -o "$WORK/n.in";  check "N. long path 20k"            "$WORK/n.in"
gen -v 50000 -e 400000 -t random -s 13 -o "$WORK/o.in"; check "O. random 50k/400k"          "$WORK/o.in"

echo "-----------------------------------------------------"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
