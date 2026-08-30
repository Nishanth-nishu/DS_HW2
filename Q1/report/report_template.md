# Q1 — Distributed Matrix Multiplication (Row-Row Method)
### Distributed Systems, Home Work 2 — Report

> **NOTE — this is a template.** Every cell marked `TBD` must be filled from an
> actual run on the RCE cluster. Do not copy numbers from anywhere else.
> ```
> sbatch slurm/job_q1.sh
> ./tools/analyze > report/benchmark_results.md
> ```
> and paste the output into sections 12–15.

**Team:** TBD  **Members:** TBD  **Date:** TBD

---

## 1. Introduction

Matrix multiplication is a natural candidate for distributed execution because
the output rows are mutually independent. This report covers a Row-Row
decomposition implemented with MPI, its correctness verification against a
sequential reference, and its measured scaling behaviour.

## 2. Problem statement

Compute `C = A × B` with A of size m×n and B of size n×p, giving C of size m×p,
where `C[i][j] = Σₖ A[i][k]·B[k][j]`. The Row-Row method expresses each row of C
as a weighted sum of the rows of B:

```
c_i = a_i[0]·B[0,:] + a_i[1]·B[1,:] + … + a_i[n−1]·B[n−1,:]
```

A is partitioned row-wise across processes; B is broadcast in full; each process
computes its own rows of C without inter-worker communication; the master
gathers the row-slices.

## 3. Sequential approach

Triple loop in `i-k-j` order, matching the Row-Row formula: the `k` loop walks
the terms of the weighted sum and the `j` loop performs the scaled row addition.
`O(m·n·p)` multiply-adds. This order is also cache-friendly, since `B[k][j]` and
`C[i][j]` are traversed sequentially in row-major memory. Used as the
correctness oracle.

## 4. Row-Row parallel approach

Row `i` of C depends only on row `i` of A and on all of B, so rows of C are
independent. Each process therefore needs its own slice of A plus a **complete**
copy of B, and computes its slice of C in isolation. B cannot be partitioned,
because the sum runs over every row of B.

## 5. MPI communication

| Call | Purpose |
|---|---|
| `MPI_Bcast` (m, n, p) | dimensions, so every rank can size buffers |
| `MPI_Scatterv` | rows of A, per-rank counts |
| `MPI_Bcast` (n·p) | matrix B in full |
| `MPI_Gatherv` | row-slices of C |

Only collectives are used, so the Send/Recv ordering deadlock cannot occur, and
every rank executes an identical unconditional sequence of calls.

## 6. Data distribution

Row-major flat arrays (`A[i][j] == A[i*n+j]`) make a block of consecutive rows a
contiguous memory block, so `Scatterv`/`Gatherv` work directly with no derived
datatypes. Counts and displacements are in elements:
`sendcount[r] = count(r)·n`, `displ[r] = start(r)·n` for A, and the same with
`p` for C.

## 7. Uneven row handling

`base = m/P`, `rem = m%P`, `count(r) = base + (r < rem)`, remainder to the
earliest ranks — matching the PDF's Example 2 (m=4, P=3 → 2, 1, 1).
`Scatter`/`Gather` cannot express varying counts, hence the `v` variants.
Ranks with zero rows (when m < P) still join every collective.

## 8. Correctness verification

22 cases at P = 1, 2, 4, 8, compared byte-for-byte: both PDF worked examples
against their published expected output, and everything else against the
sequential reference. Integer arithmetic means the match is exact.

Result: **TBD** — paste the suite output here.

## 9. Experimental setup

| | |
|---|---|
| Cluster / partition | RCE, `debug` |
| Allocation | `--nodes=2 --ntasks-per-node=4` (8 tasks, one job for all P) |
| MPI module | `hpcx-2.7.0/hpcx-ompi` |
| Compiler flags | `mpicxx -O2 -std=c++17` |
| Trials per configuration | 5, median reported |
| Timed region | `t_algo = t_compute + t_comm` (excludes generation and printing) |
| Aggregation across ranks | maximum (slowest rank) |

## 10. Matrix sizes

| Label | m × n × p | Purpose |
|:--|:--|:--|
| Small | 256 × 256 × 256 | |
| Medium | 512 × 512 × 512 | |
| Large | 1024 × 1024 × 1024 | |
| Very large | 1500 × 1500 × 1500 | |
| skew_tall | 4096 × 128 × 128 | m ≫ n: small B, favourable to Row-Row |
| skew_wide | 128 × 1024 × 1024 | n·p ≫ m: large broadcast, unfavourable |
| uneven | 1000 × 999 × 997 | m not divisible by any of P = 2, 4, 8 |

## 11. Process counts

P = 1, 2, 4, 8.

## 12. Timing results

Paste the runtime table from `./tools/analyze`.

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | TBD | TBD | TBD | TBD |
| Medium | TBD | TBD | TBD | TBD |
| Large | TBD | TBD | TBD | TBD |
| Very large | TBD | TBD | TBD | TBD |

## 13. Speed-up  S(P) = T₁ / T_P

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | 1.00 | TBD | TBD | TBD |
| Medium | 1.00 | TBD | TBD | TBD |
| Large | 1.00 | TBD | TBD | TBD |
| Very large | 1.00 | TBD | TBD | TBD |

## 14. Efficiency  E(P) = S(P) / P

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | 100% | TBD | TBD | TBD |
| Medium | 100% | TBD | TBD | TBD |
| Large | 100% | TBD | TBD | TBD |
| Very large | 100% | TBD | TBD | TBD |

## 15. Graphs

- `benchmark/plots/time.svg` — execution time vs P
- `benchmark/plots/speedup.svg` — speed-up vs P, with the ideal line
- `benchmark/plots/efficiency.svg` — efficiency vs P
- `benchmark/plots/comm_fraction.svg` — communication share vs P

## 16. Communication vs computation

`t_comm` as a percentage of `t_algo`: **TBD**

Discuss against the measured numbers:

- **Computation** is `O((m/P)·n·p)` and falls as 1/P.
- **Broadcasting B** moves `n·p` elements to *every* process. This cost does not
  fall with P — it typically rises, roughly `log P` in a tree implementation.
  It is the main obstacle to linear speed-up, and it is the reason the
  `skew_wide` shape should scale worse than `skew_tall` at the same total work.
  Compare those two rows and report what was actually observed.
- **Scatterv A / Gatherv C** move `m·n` and `m·p` elements in total; unlike the
  broadcast these are split across ranks, but root remains a serialisation point.
- **Synchronization**: each collective is an implicit barrier; the phase ends
  when the slowest rank finishes.
- **Load balance**: with `m % P != 0`, some ranks hold one extra row. Report the
  `rows/rank min…max` spread from `--stats`; the relative imbalance is
  `1/⌈m/P⌉`, negligible at m = 1000 and significant only for tiny m.
- **Memory bandwidth**: multiplication is memory-bound at these sizes; several
  ranks on one node share bandwidth, so per-rank throughput can fall as P rises
  within a node even with no communication at all. Note whether the P=1→2 and
  P=4→8 steps behave differently, since they may cross a node boundary.

## 17. Scalability

Fill from measurement. Points to address: which size scales best and why;
where efficiency drops below 50%; whether any configuration is slower at higher
P and whether `t_dist` explains it; and how the skewed shapes compare with the
square ones at similar total work.

## 18. Complexity

Sequential `O(m·n·p)`; per process `O((m/P)·n·p)`; communication `m·n` scattered,
`n·p` broadcast to all, `m·p` gathered.

## 19. Memory analysis

Root: `m·n + n·p` ints plus `m·p` long longs. Each worker: `(m/P)·n + n·p` ints
plus `(m/P)·p` long longs. **B is replicated on every process** — at
1500×1500 that is ~9 MB per rank, ~72 MB across 8 ranks. This replication is
inherent to the Row-Row method and is the main memory limitation.

## 20. Conclusion

TBD — summarise measured trends only. Do not claim scaling behaviour the data
does not show.
