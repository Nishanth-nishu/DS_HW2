# Q7 — Large-Scale Server Log Analytics
### Distributed Systems, Home Work 2 — Report

> **NOTE — this is a template.** Every cell marked `TBD` must be filled from an
> actual run on the RCE cluster. Do not copy numbers from anywhere else.
> ```
> sbatch slurm/job_q7.sh
> ./tools/analyze > report/benchmark_results.md
> ```
> and paste the output into sections 11–14.

**Team:** TBD  **Members:** TBD  **Date:** TBD

---

## 1. Introduction

Server log analytics is a natural fit for distributed execution: the records are
independent, and every required statistic is a fold that combines associatively.
This report covers an MPI implementation, its correctness verification against a
sequential reference, and its measured scaling behaviour.

## 2. Problem statement

Given `N` log records of the form
`timestamp server_id endpoint_id user_id status_code response_time bytes_sent`,
compute request totals, response-time statistics, byte totals, status-class
counts, top-K servers and endpoints, and the busiest 60-second interval. A
request is successful when `status_code < 400`; the interval ID is
`timestamp / 60`.

## 3. Points the PDF leaves open

`S` in the header is never defined; no sample data is supplied; field types,
number formatting, `BUSIEST_INTERVAL` tie-breaking, the fewer-than-K case and
`N = 0` are all unspecified. Each was resolved as a documented implementation
choice — see README §2. The expected output used for verification was computed
by hand, since the PDF provides no sample.

## 4. Sequential approach

A single pass over the records maintaining scalar counters and three hash maps
(server → count and response sum, endpoint → count and byte total, interval →
count), then a sort of the distinct IDs for top-K selection.
**O(N + D log D)** time, **O(N + D)** memory.

## 5. Parallel approach

Every required statistic is a commutative, associative fold, so each rank
aggregates its own slice with no communication and the partial results combine
exactly. Top-K selection is the exception: a server ranked 3rd on every rank
could still be 1st globally, so ranks exchange full per-key counts and only the
root selects top-K after merging.

## 6. MPI communication

| Call | Purpose |
|---|---|
| `MPI_Bcast` | header N, K, S |
| `MPI_Scatterv` | records, per-rank counts |
| `MPI_Reduce` (SUM) | nine scalar counters and sums |
| `MPI_Reduce` (MIN/MAX) | min and max response time |
| `MPI_Gather` / `MPI_Gatherv` | per-key `(id, count, extra)` triples |

Only collectives; no Send/Recv, so the ordering deadlock cannot occur. Ranks
with zero records contribute the MIN/MAX identities and join every collective.

## 7. Data distribution

Contiguous record blocks, remainder to the earliest ranks:
`count(r) = N/P + (r < N%P)`, `start(r) = r·(N/P) + min(r, N%P)`. Handles
`N % P != 0` and `N < P`. Records are a POD struct in a contiguous array, so
`MPI_BYTE` moves them with no derived datatype.

## 8. Merging per-key results

Each rank flattens its map to `(id, count, extra)` triples; `Gatherv` collects
them; root sums key by key. Communication is O(P × D) rather than O(N), and no
assumption is made about ID range or density — unlike a dense array reduction
indexed by ID.

## 9. Correctness verification

20 cases at P = 1, 2, 4, 8: a hand-computed fixture exercising every output
field and both tie-breaks; structural edge cases (`N=0`, `N=1`, `N<P`, `K=0`,
`K` > distinct IDs, single server, three-way interval tie, 1xx status);
generated logs with prime and odd `N`, 1 and 500 servers, up to 500,000
records; and three input-validation cases. Comparison is byte-for-byte.

Result: **TBD** — paste the suite output here.

## 10. Experimental setup

| | |
|---|---|
| Cluster / partition | RCE, `debug` |
| Allocation | `--nodes=2 --ntasks-per-node=4` (8 tasks, one job for all P) |
| MPI module | `hpcx-2.7.0/hpcx-ompi` |
| Compiler flags | `mpicxx -O2 -std=c++17` |
| Trials per configuration | 5, median reported |
| Timed region | `t_algo = t_compute + t_comm` (excludes generation and printing) |
| Aggregation across ranks | maximum (slowest rank) |

## 11. Input sizes

| Label | N | K | S | Purpose |
|:--|--:|--:|--:|:--|
| Small | 100,000 | 10 | 32 | |
| Medium | 1,000,000 | 10 | 64 | |
| Large | 5,000,000 | 10 | 128 | |
| Very large | 10,000,000 | 10 | 128 | |
| keys_few | 1,000,000 | 10 | 4 | low key cardinality: cheap merge |
| keys_many | 1,000,000 | 10 | 5000 | high key cardinality: expensive merge |
| topk_large | 1,000,000 | 100 | 128 | larger K, same data |

All regenerable via `./tools/gen_log N K S SEED`.

## 12. Timing results

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

- **Computation** is O(N/P) and falls with P.
- **`t_dist`** distributes N records once. Root is a serialisation point: it
  holds and sends the whole log, and that cost does not fall with P.
- **`t_reduce`** has two parts. The scalar `Reduce` is 11 values — negligible.
  The per-key `Gatherv` is O(P × D) and **grows** with P. Compare `keys_few`
  against `keys_many` at the same N: if the merge matters, those two rows
  should diverge as P rises. Report what was actually observed.
- **Synchronization**: each collective is an implicit barrier; a phase ends
  when its slowest rank finishes.
- **Load balance**: records are split evenly to within one, so imbalance is
  negligible for large N. Report the `records/rank min…max` spread from
  `--stats`.
- **Memory bandwidth**: the aggregation is a streaming pass over records and is
  memory-bound; ranks sharing a node share bandwidth, so per-rank throughput
  can fall as P rises within a node even with no communication. Note whether
  the P=1→2 and P=4→8 steps differ, since they may cross a node boundary.
- **Why speed-up is not linear**: distribution cost is not parallelised, the
  merge grows with P, and the per-record work is small enough that the fixed
  costs are a meaningful fraction at small N.

## 17. Scalability

Fill from measurement. Which N first shows real speed-up; where efficiency
drops below 50%; whether any configuration is slower at higher P and whether
`t_dist` or `t_reduce` explains it; and how key cardinality changes the picture.

## 18. Complexity

Sequential O(N + D log D); per process O(N/P); communication N records
scattered, 11 scalars reduced, O(P × D) triples gathered.

## 19. Memory analysis

Root holds the whole log before scattering — at 10,000,000 records × 40 bytes,
~400 MB — plus the merged maps, O(D). Each worker holds O(N/P) records and its
own local maps. Root's memory is the binding constraint on problem size.

## 20. Conclusion

TBD — summarise measured trends only. Do not claim scaling behaviour the data
does not show.
