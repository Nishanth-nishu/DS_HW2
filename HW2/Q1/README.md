# Q1 — Distributed Matrix Multiplication, Row-Row Method (MPI)

Distributed Systems, Home Work 2, Section 1 (Distributed Algorithms).

Compute `C = A × B` for `A` of size m×n and `B` of size n×p, using the
**Row-Row** method: matrix A is partitioned row-wise across processes, matrix B
is broadcast in full to every process, and each process computes its own rows
of C independently.

---

## 1. Problem statement

`C[i][j] = Σₖ A[i][k] · B[k][j]`, which the Row-Row method expresses as a
weighted sum of the **rows of B**:

```
c_i = a_i[0]·B[0,:] + a_i[1]·B[1,:] + ... + a_i[n-1]·B[n-1,:]
```

Example (PDF Example 1), A = 3×2, B = 2×3:

```
c1 = 1·(2,3,4) + 2·(1,0,−1) = (4, 3, 2)
c2 = 0·(2,3,4) + 3·(1,0,−1) = (3, 0, −3)
c3 = −1·(2,3,4) + 4·(1,0,−1) = (2, −3, −8)
```

`A(m×n) × B(n×p)` is defined only when A's column count equals B's row count.
Our input format states both as the single value `n`, so a mismatch cannot be
expressed (see §4).

## 2. Why rows of A are the unit of work

Row `i` of C depends on exactly two things: **row `i` of A**, and **all of B**.
It does not depend on any other row of A or of C. Rows of C are therefore
completely independent, and a process holding some rows of A plus a full copy
of B computes its rows of C with **no communication with any other worker** —
exactly as the PDF specifies.

The converse is why B cannot be split: the sum runs over `k = 0 … n−1`, so a
complete row of C needs *every* row of B. Replication is inherent to this
method, and is the cost that does **not** shrink as P grows.

## 3. Loop order — `i-k-j`, not `i-j-k`

The Row-Row formula is implemented as:

```cpp
for (i)                        // each row of C
  for (k) {                    // each row of B  = each term of the weighted sum
    a = A[i][k];               // the weight, loop-invariant in j
    for (j)                    // scale a whole row of B and add it
      C[i][j] += a * B[k][j];
  }
```

This is the method the PDF defines, and it is also the cache-friendly order:
`B[k][j]` and `C[i][j]` are both walked sequentially in row-major memory,
whereas the `i-j-k` inner-product order strides through `B[k][j]` by `p`
elements per step.

`rowRowMultiply` skips a term when the weight is zero (scaling a row of B by
zero and adding it is a no-op). Both binaries do this identically, so
correctness is unaffected, but note that runtime is therefore mildly
data-dependent.

## 4. Input and output format — **implementation choice**

The PDF prescribes no I/O format for Q1 (unlike Q4–Q6, which specify one). The
following is our choice and is not taken from the assignment:

**Input** (stdin)

```
m n p
<m*n integers>       matrix A, row-major
<n*p integers>       matrix B, row-major
```

Whitespace between values is irrelevant. Truncated or overlong input is
rejected with a message on stderr and a non-zero exit.

**Output** (stdout) — `m` lines, each of `p` space-separated integers.

**Generated mode.** The PDF permits the master to *read or generate* A and B.
`--gen M N P [SEED]` generates both matrices in-process from a deterministic
`splitmix64` stream, with entries in `[-9, 9]`. This is used for benchmarking so
that parsing a multi-megabyte text file does not dominate a measurement meant to
be about multiplication. The generator lives in `matrix_io.hpp`, shared by both
binaries, so `--gen` produces **bit-identical** matrices in each — without which
comparing their outputs would prove nothing.

## 5. Element types — **implementation choice**

`A`, `B` are `int`; `C` accumulates and is stored as `long long`. The PDF states
entries are integers but gives no magnitude bound, and with n = 1000 a 32-bit
accumulator can overflow. Widening C removes the risk. With the built-in
generator, `|C| ≤ 81n`, so overflow is provably impossible.

## 6. Row distribution, including the uneven case

The remainder goes to the **earliest** ranks, matching the PDF's Example 2
(m = 4, P = 3 → P0 gets 2 rows, P1 gets 1, P2 gets 1):

```
base = m / P            rem = m % P
count(r) = base + (r < rem ? 1 : 0)
start(r) = r*base + min(r, rem)
```

| m=10, P=4 | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| rows | 3 | 3 | 2 | 2 |
| start | 0 | 3 | 6 | 8 |

**Load balance.** The slowest process sets the pace, so the imbalance is
`1/⌈m/P⌉` in the worst case — severe for tiny m, under 1% for m = 1000 at P = 8.

**m < P** is handled: trailing ranks receive zero rows. They still take part in
every collective; skipping one would hang the job.

## 7. Data layout and MPI communication

Matrices are flat, **row-major**, 1-D arrays: `A[i][j] == A[i*n + j]`. A block
of consecutive rows is therefore a contiguous block of memory, which is exactly
what `Scatterv`/`Gatherv` consume — **no derived MPI datatypes are needed**.
(By contrast, a *column* is strided and would require `MPI_Type_vector`; this
is why Row-Row is the simpler decomposition.)

| Call | Purpose |
|---|---|
| `MPI_Bcast` (3 values) | dimensions m, n, p — nobody can size a buffer without them |
| `MPI_Scatterv` | rows of A; per-rank counts differ whenever `m % P != 0`, so the `v` variant is required |
| `MPI_Bcast` (n·p) | matrix B in full — every process needs every row of B |
| `MPI_Gatherv` | row-slices of C, per-rank counts again varying |
| `MPI_Barrier`, `MPI_Wtime`, `MPI_Reduce` | benchmarking only |

Counts and displacements are in **elements, not rows**:

```
Scatterv A:  sendcount[r] = count(r)*n    displ[r] = start(r)*n
Gatherv  C:  recvcount[r] = count(r)*p    displ[r] = start(r)*p
```

`Scatter`/`Gather` send the *same* count to every rank and are therefore
insufficient when `m % P != 0`; `Scatterv`/`Gatherv` take per-rank count and
displacement arrays.

**Deadlock safety.** Only collectives are used. Every rank executes an
identical, unconditional sequence of them, so the Send/Recv ordering deadlock
cannot arise, and no rank can skip a collective — the classic
`if (myRows > 0) MPI_Gatherv(...)` hang when `m < P` is avoided by design.

## 8. Complexity and memory

| | |
|---|---|
| Sequential | O(m·n·p) multiply-adds; O(n³) for square |
| Per process | O((m/P)·n·p) |
| Scatterv A | m·n elements leave root |
| Bcast B | n·p elements to **every** process, O(n·p·log P) in a tree |
| Gatherv C | m·p elements arrive at root |
| Root memory | m·n + n·p (int) + m·p (long long) |
| Worker memory | (m/P)·n + **n·p** + (m/P)·p |

Computation falls as 1/P; the broadcast of B does not. Speed-up is therefore
expected to be **sublinear and to worsen for "wide" shapes** where n·p is large
relative to m. The `t_dist`/`t_compute`/`t_gather` instrumentation measures this
rather than assuming it.

## 9. Build

```bash
module load hpcx-2.7.0/hpcx-ompi      # on the RCE cluster

cd src && make                        # both binaries
# or explicitly:
mpicxx -O2 -std=c++17 -o q1_mpi     q1_mpi.cpp
g++    -O2 -std=c++17 -o sequential sequential.cpp
```

## 10. Run

```bash
./src/sequential < input.txt > output.txt

mpirun -np 1 ./src/q1_mpi < input.txt > output.txt
mpirun -np 2 ./src/q1_mpi < input.txt > output.txt
mpirun -np 4 ./src/q1_mpi < input.txt > output.txt
mpirun -np 8 ./src/q1_mpi < input.txt > output.txt

# generated problem, no C printed, timings on stderr
mpirun -np 4 ./src/q1_mpi --gen 1024 1024 1024 --stats --no-output
```

Options: `--stats` (timings to **stderr**), `--no-output` (suppress C),
`--gen M N P [SEED]`. stdout always carries only matrix C.

## 11. SLURM (RCE)

```bash
sbatch slurm/job_q1.sh          # edit PROJECT_DIR inside first
squeue -u $USER                 # status
scontrol show job <JOB_ID>      # details
scancel <JOB_ID>                # cancel
cat job_<JOB_ID>.log            # output   (errors: job_<JOB_ID>.err)
```

The script requests `--nodes=2 --ntasks-per-node=4` = 8 tasks, so P = 1, 2, 4, 8
all run inside one allocation on the same hardware.

## 12. Correctness verification

```bash
bash tests/run_tests.sh
MPIRUN_FLAGS="--oversubscribe" bash tests/run_tests.sh   # oversubscribed nodes
```

Two kinds of check: against the PDF's own worked examples (known expected
output), and MPI vs the sequential reference at P = 1, 2, 4, 8. 22 cases:
both PDF examples · small square · rectangular · m=1 · n=1 · p=1 · 1×1×1 ·
m divisible by P · m not divisible by P (10, 13) · m < P (3, 5) · tall skew ·
wide skew · wide output · square 64 · 200×150×120 · prime dims 101×97×89.

Integer arithmetic means the match must be **exact**, and it is compared
byte-for-byte with `diff`.

## 13. Benchmarking

```bash
bash benchmark/run_bench.sh          # writes benchmark/results.csv
bash benchmark/run_bench.sh --quick  # skip the two largest sizes
python3 benchmark/analyze.py         # tables + plots
```

**What is timed.** `t_algo = t_compute + t_comm` is the headline figure, where
`t_comm = t_dist + t_gather`. Each value is the **maximum across ranks** (a
parallel phase ends when its slowest rank ends) and the **median over 5 trials**.
Compilation and file parsing are excluded; matrices are generated in-process and
`--no-output` suppresses printing C.

`S(P) = T1 / TP` uses the **MPI binary at P=1** as `T1`, so the speed-up
measures parallel scaling of one implementation rather than comparing two
different programs.

Existing `results.csv` is never overwritten — it is renamed with a timestamp.

## 14. Assumptions and limitations

1. **Input/output format is our choice** (§4); the PDF prescribes none for Q1.
2. **C is 64-bit** (§5) because the PDF does not bound entry magnitude.
3. **B is replicated** on every process — inherent to Row-Row. At 1500×1500 that
   is ~9 MB per process.
4. **32-bit MPI counts** are checked explicitly; `m·n`, `n·p`, `m·p` must each
   fit in `int`.
5. **Zero-weight skipping** makes runtime mildly data-dependent (§3).
6. **Root reads/generates the whole problem** before distributing, so root's
   memory bounds the largest solvable size.
7. `m = 0` or `p = 0` exits cleanly with empty output rather than erroring; the
   PDF does not define this case.

## 15. Project structure

```
Q1/
├── README.md
├── src/
│   ├── matrix_io.hpp       shared I/O, generator, Row-Row kernel
│   ├── sequential.cpp      sequential reference (oracle)
│   ├── q1_mpi.cpp          MPI implementation
│   └── Makefile
├── tests/
│   ├── pdf_ex1.in/.expected    PDF Example 1
│   ├── pdf_ex2.in/.expected    PDF Example 2
│   ├── small_square.in, rect.in, m1.in, n1.in, p1.in
│   └── run_tests.sh            22 cases at P=1,2,4,8
├── benchmark/
│   ├── run_bench.sh        writes results.csv
│   ├── analyze.py          tables + plots
│   └── plots/
├── slurm/
│   └── job_q1.sh
└── report/
    └── report_template.md
```
