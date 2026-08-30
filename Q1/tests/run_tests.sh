#!/bin/bash
# ---------------------------------------------------------------------------
# run_tests.sh -- correctness suite for Q1 (Row-Row matrix multiplication).
#
# Two kinds of check:
#   1. against the PDF's own worked examples, which have known expected output
#   2. MPI output vs the sequential reference, at P = 1, 2, 4, 8
#
# Integer arithmetic means the match must be EXACT; outputs are compared with
# diff, byte for byte, with no tolerance.
#
# PASS/FAIL is printed by this script only; neither program under test has its
# own output format altered.
#
# Usage: bash tests/run_tests.sh          (from the Q1 project root)
# Env:   MPIRUN_FLAGS   extra mpirun flags, e.g. "--oversubscribe"
#        PROCS          process counts, default "1 2 4 8"
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="$ROOT/src/sequential"
MPI="$ROOT/src/q1_mpi"
WORK="$(mktemp -d)"
FLAGS="${MPIRUN_FLAGS:-}"
PROCS="${PROCS:-1 2 4 8}"

pass=0; fail=0
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$SEQ" ] || [ ! -x "$MPI" ]; then
    echo "ERROR: binaries not found. Run 'make' first." >&2
    exit 2
fi

# Note on </dev/null below: mpirun reads stdin. It is harmless in these loops,
# but the redirect keeps the behaviour identical whether this script is run
# interactively, from a pipeline, or from a SLURM job.

# --- file-based case with published expected output ------------------------
check_expected() {
    local name="$1" input="$2" expected="$3"
    local ok=1 detail=""
    "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null
    diff -q "$WORK/seq.out" "$expected" >/dev/null 2>&1 || { ok=0; detail=" seq"; }
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/m.$p.out" 2>/dev/null
        diff -q "$WORK/m.$p.out" "$expected" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-42s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-42s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- file-based case: MPI vs sequential ------------------------------------
check_file() {
    local name="$1" input="$2"
    if ! "$SEQ" < "$input" > "$WORK/seq.out" 2>/dev/null; then
        printf '  %-42s FAIL (sequential failed)\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for p in $PROCS; do
        mpirun $FLAGS -np "$p" "$MPI" < "$input" > "$WORK/m.$p.out" 2>/dev/null
        diff -q "$WORK/seq.out" "$WORK/m.$p.out" >/dev/null 2>&1 || { ok=0; detail="$detail P=$p"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-42s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-42s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- generated case: MPI vs sequential -------------------------------------
# Both binaries share matrix_io.hpp, so --gen yields bit-identical matrices.
check_gen() {
    local name="$1" m="$2" n="$3" p="$4" seed="${5:-7}"
    if ! "$SEQ" --gen "$m" "$n" "$p" "$seed" > "$WORK/seq.out" 2>/dev/null; then
        printf '  %-42s FAIL (sequential failed)\n' "$name"; fail=$((fail+1)); return
    fi
    local ok=1 detail=""
    for np in $PROCS; do
        mpirun $FLAGS -np "$np" "$MPI" --gen "$m" "$n" "$p" "$seed" \
            > "$WORK/m.$np.out" 2>/dev/null </dev/null
        diff -q "$WORK/seq.out" "$WORK/m.$np.out" >/dev/null 2>&1 || { ok=0; detail="$detail P=$np"; }
    done
    if [ $ok -eq 1 ]; then printf '  %-42s PASS\n' "$name"; pass=$((pass+1))
    else printf '  %-42s FAIL (mismatch at:%s)\n' "$name" "$detail"; fail=$((fail+1)); fi
}

# --- malformed input must fail cleanly, not crash or produce output --------
check_rejects() {
    local name="$1" content="$2"
    printf '%b' "$content" > "$WORK/bad.in"
    "$SEQ" < "$WORK/bad.in" > "$WORK/bad.out" 2> "$WORK/bad.err"
    local code=$?
    if [ $code -ne 0 ] && [ -s "$WORK/bad.err" ] && [ ! -s "$WORK/bad.out" ]; then
        printf '  %-42s PASS\n' "$name"; pass=$((pass+1))
    else
        printf '  %-42s FAIL (expected non-zero exit + message)\n' "$name"; fail=$((fail+1))
    fi
}

echo "Q1 correctness suite  (process counts: $PROCS)"
echo "---------------------------------------------------------------"

echo "  [PDF worked examples -- known expected output]"
check_expected "PDF Example 1 (3x2 * 2x3, even split)" \
               "$ROOT/tests/pdf_ex1.in" "$ROOT/tests/pdf_ex1.expected"
check_expected "PDF Example 2 (4x2 * 2x2, uneven split)" \
               "$ROOT/tests/pdf_ex2.in" "$ROOT/tests/pdf_ex2.expected"

echo "  [file-based cases -- MPI vs sequential]"
check_file "small square 2x2 * 2x2"  "$ROOT/tests/small_square.in"
check_file "rectangular 2x3 * 3x4"   "$ROOT/tests/rect.in"
check_file "m=1 single row"          "$ROOT/tests/m1.in"
check_file "n=1 outer product"       "$ROOT/tests/n1.in"
check_file "p=1 single column"       "$ROOT/tests/p1.in"

echo "  [generated cases -- MPI vs sequential]"
check_gen "m divisible by P    (m=8)"        8    5   6
check_gen "m NOT divisible by P (m=10)"      10   5   6
check_gen "m NOT divisible by P (m=13)"      13   7   9
check_gen "m=1 (single row)"                 1    9   9
check_gen "n=1 (rank-1 outer product)"       20   1  11
check_gen "p=1 (single column)"              17  13   1
check_gen "1x1x1"                            1    1   1
check_gen "m < P (m=3, idle ranks at P=8)"   3    4   4
check_gen "m < P (m=5)"                      5    6   7
check_gen "tall skewed  m >> n (200x5)"      200  5   4
check_gen "wide skewed  n >> m (5x200)"      5  200   4
check_gen "wide output  p >> m (6x6x200)"    6    6 200
check_gen "square 64"                        64  64  64
check_gen "larger 200x150x120"               200 150 120
check_gen "prime dims 101x97x89"             101 97  89

echo "  [input validation]"
check_rejects "rejects truncated input"  '2 2 2\n1 2\n3 4\n5 6\n'
check_rejects "rejects overlong input"   '2 2 2\n1 2\n3 4\n5 6\n7 8\n9\n'
check_rejects "rejects missing header"   '\n'

echo "---------------------------------------------------------------"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
