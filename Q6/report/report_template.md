# Q6 — Connected Components of a Large Graph
### Distributed Systems, Home Work 2 — Report

> **NOTE — this is a template.** Every cell marked `TBD` must be filled from an
> actual run on the RCE cluster. Do not copy numbers from anywhere else.
> Generate the tables with:
> ```
> sbatch slurm/job_q6.sh
> ./tools/analyze > report/benchmark_results.md
> ```
> and paste the output into sections 8–11 below.

**Team:** TBD  **Members:** TBD  **Date:** TBD

---

## 1. Problem statement

Identify the connected components of a large undirected graph supplied as an
adjacency list, distributing the vertices and their adjacency lists across MPI
processes and using message passing to assign each vertex a component ID. The
component ID must be the **minimum vertex ID** in that component, and output
must be sorted ascending by vertex ID.

Constraints: `1 ≤ V ≤ 10^5`, `0 ≤ E ≤ 10^6`, vertices 0-indexed.

## 2. Approach

A single local traversal is insufficient, because a component may span several
processes: each rank would find only locally-correct fragments. We therefore
use **iterative minimum-label propagation**. Every vertex holds a label that
only ever decreases; each round, a rank extracts everything its own edges imply
using a union-find, and one `MPI_Allreduce(MPI_MIN)` merges every rank's
knowledge. Iteration continues until the label array stops changing.

## 3. Sequential algorithm

BFS from each not-yet-visited vertex, seeds scanned in increasing order, so the
seed of a component is automatically its minimum. **O(V + E)** time and memory.
Used as the correctness oracle.

## 4. MPI algorithm

```
comp[v] = v                                for all v
repeat:
    reset union-find (union-by-minimum-root)
    for v: if comp[v] != v: unite(v, comp[v])       # re-seed prior knowledge
    for each locally owned u, each neighbour w: unite(u, w)
    for v: comp[v] = find(v)
    MPI_Allreduce(comp, V, MPI_INT, MPI_MIN)
    stop when comp is unchanged
```

Union-by-minimum-root makes `find(v)` the smallest known ID in v's set
directly. Re-seeding from `comp[]` lets remotely-discovered links compose
transitively with local edges.

## 5. Data distribution

Contiguous vertex blocks, `first(r) = (r·V)/P`; integer arithmetic handles
`V` not divisible by `P` and `V < P`. Each rank receives only its own vertices'
adjacency lists via two `MPI_Scatterv` calls. The O(E) data is partitioned; the
O(V) label array is replicated (400 KB at V = 10^5).

## 6. Communication pattern

| Call | Purpose |
|---|---|
| `MPI_Bcast` | broadcast `V` |
| `MPI_Scatter` / `MPI_Scatterv` | distribute counts, degrees, neighbour entries |
| `MPI_Allreduce(MPI_MIN)` | merge labels; one per round |

No point-to-point communication, therefore no possibility of deadlock. No final
gather (results are already replicated) and no extra collective for convergence
detection (post-`Allreduce` arrays are bit-identical on all ranks).

## 7. Correctness

*Soundness:* a label is always the ID of a genuinely connected vertex, so
disconnected components can never merge. *Monotonicity:* labels never increase.
*Termination:* the label vector is non-increasing, bounded below, and strictly
decreases each non-final round. *Fixpoint:* at convergence labels agree across
every edge, so they are constant per component; that constant is a member and
is `≤` every member, hence the minimum. ∎

**Verification.** `./tools/run_tests` compares MPI output against the
sequential oracle at P = 1, 2, 4, 8 across 15 cases (sample, V=1, E=0, fully
disconnected, single component, multiple components, cross-process bridges,
V=97 prime, star, asymmetric adjacency, V<P, and larger random/path graphs).

Result: **TBD** — paste the suite output here.

## 8. Experimental setup

| | |
|---|---|
| Cluster / partition | RCE, `debug` |
| Allocation | `--nodes=2 --ntasks-per-node=4` (8 tasks, one job for all P) |
| MPI module | `hpcx-2.7.0/hpcx-ompi` |
| Compiler flags | `mpicxx -O2 -std=c++17` |
| Trials per configuration | 5, median reported |
| Timed region | `t_algo = t_compute + t_comm` (excludes file I/O and distribution) |
| Aggregation across ranks | maximum (slowest rank) |

## 9. Input sizes

| Label | V | E | Topology | Seed |
|:--|--:|--:|:--|--:|
| Small | 10,000 | 100,000 | random | 101 |
| Medium | 50,000 | 500,000 | random | 102 |
| Large | 100,000 | 1,000,000 | random | 103 |
| Very large | 100,000 | 1,000,000 | cluster (16 blobs) | 104 |
| path | 100,000 | 99,999 | path | 105 |
| shuffled | 100,000 | 99,999 | permuted path | 106 |
| chain | 100,000 | ~150,000 | chain-blocks | 107 |

All regenerable via `./tools/gen_graph` with the seeds above.

## 10. Execution time, speed-up, efficiency

Paste `./tools/analyze` output here.

**Runtime — median `t_algo` (s)**

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | TBD | TBD | TBD | TBD |
| Medium | TBD | TBD | TBD | TBD |
| Large | TBD | TBD | TBD | TBD |
| Very large | TBD | TBD | TBD | TBD |

**Speed-up  S(P) = T₁ / T_P**

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | 1.00 | TBD | TBD | TBD |
| Medium | 1.00 | TBD | TBD | TBD |
| Large | 1.00 | TBD | TBD | TBD |
| Very large | 1.00 | TBD | TBD | TBD |

**Efficiency  E(P) = S(P) / P**

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Small | 100% | TBD | TBD | TBD |
| Medium | 100% | TBD | TBD | TBD |
| Large | 100% | TBD | TBD | TBD |
| Very large | 100% | TBD | TBD | TBD |

## 11. Plots

- `benchmark/plots/time.svg` — execution time vs P
- `benchmark/plots/speedup.svg` — speed-up vs P, with the ideal line
- `benchmark/plots/efficiency.svg`
- `benchmark/plots/comm_fraction.svg`

## 12. Communication vs computation

`t_comm` as a percentage of `t_algo`:

| Input size | P=1 | P=2 | P=4 | P=8 |
|:--|:-:|:-:|:-:|:-:|
| Large | TBD | TBD | TBD | TBD |

Discuss using the measured values:

- **Local graph processing.** Only the edge-union loop is O(E/P); the
  union-find reset, re-seed and collapse are O(V) and replicated on every rank.
- **Communication of component IDs.** One `Allreduce` of `V` ints per round —
  400 KB at V = 10^5, independent of `P` in volume but rising in latency with
  `log P`.
- **Cross-process edges.** Handled implicitly: a rank writes labels for remote
  vertices, and the `Allreduce` publishes them.
- **Synchronization.** One implicit barrier per round at the `Allreduce`; the
  round ends when the slowest rank finishes.
- **Convergence rounds.** Measured, deterministic, and machine-independent.
- **Load balancing.** Vertices are split evenly, but *edges* are not: report
  the `edges/rank min…max` spread from `--stats`. Skewed-degree graphs (star)
  are the worst case.
- **Why speed-up is not linear.** The replicated O(V) work and the O(V)
  `Allreduce` do not shrink with `P` (Amdahl). At the assignment's constraints
  the computation is small in absolute terms, so per-round communication and
  synchronization can dominate at higher `P`.

## 13. Scalability discussion

Fill from measurement. Points to address: which size first shows real speed-up;
where efficiency falls below 50%; whether any configuration is *slower* at
higher `P`, and whether `t_comm` explains it; and how round count varies with
topology and vertex numbering.

## 14. Conclusion

TBD — summarise measured trends only. Do not claim scaling behaviour that the
data does not show.
