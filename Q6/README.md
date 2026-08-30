# Q6 — Connected Components of a Large Graph (MPI)

Distributed Systems, Home Work 2, Section 2 (Graph Algorithms).

Identify the connected components of a large undirected graph. Each vertex is
labelled with the **minimum vertex ID** in its component. The graph is
distributed across MPI processes; component labels are resolved by
message passing.

---

## 1. Problem

**Input** (stdin)

```
V
k v1 v2 ... vk        <- line i (0-indexed) lists the k neighbours of vertex i
...                      V such lines
```

**Output** (stdout) — exactly `V` lines, sorted ascending by vertex ID:

```
vertex_id component_id
```

**Constraints** — `1 ≤ V ≤ 10^5`, `0 ≤ E ≤ 10^6`, vertices 0-indexed.

**Sample**

| input | output |
|---|---|
| `5`<br>`1 1`<br>`1 0`<br>`2 3 4`<br>`1 2`<br>`1 2` | `0 0`<br>`1 0`<br>`2 2`<br>`3 2`<br>`4 2` |

Components are `{0,1}` → label `0`, and `{2,3,4}` → label `2`.

---

## 2. Algorithm

### 2.1 Sequential reference (`src/sequential.cpp`)

BFS from each not-yet-visited vertex, scanning seeds in increasing order.
Because seeds are scanned in increasing order, the seed of a component is
automatically its minimum: if a smaller vertex shared the component it would
have been seeded earlier and would already have labelled this one.

Complexity **O(V + E)** time, **O(V + E)** memory.

### 2.2 MPI algorithm (`src/q6_mpi.cpp`) — iterative minimum-label propagation

A single local BFS is *not* sufficient, because a component may span several
processes. Instead every vertex carries a label that only ever decreases:

```
comp[v] = v                                for all v

repeat:
    # local phase -- no communication
    reset union-find
    for v: if comp[v] != v: unite(v, comp[v])       # re-seed prior knowledge
    for each locally owned u, each neighbour w: unite(u, w)
    for v: comp[v] = find(v)

    # global phase -- one collective
    MPI_Allreduce(comp, V, MPI_INT, MPI_MIN)

    if comp unchanged since last round: stop
```

The union-find uses **union-by-minimum-root**, so `find(v)` *is* the smallest
ID in v's known set — no separate minimum pass is needed.

**Why re-seeding matters.** `comp[v] = c` records "v is connected to c",
knowledge that may have arrived from another rank. Feeding it back into a fresh
union-find lets this round's local edges compose transitively with it. Because
the label array is replicated, every rank sees *every* link discovered anywhere,
so a chain of P links collapses in **one** round rather than propagating one hop
per round. This is why the measured round count stays small.

### 2.3 Correctness

*Invariant (soundness).* `comp[v]` is always the ID of a vertex genuinely
connected to `v`. True initially (`comp[v] = v`); `unite(u,w)` is applied only
to real edges or to pairs already known connected; `MPI_MIN` selects one rank's
value, each valid. **Consequence: disconnected components can never merge**,
since a shared label would have to lie in the intersection of two disjoint
components.

*Monotonicity.* `comp[v]` never increases — union-find roots are minima and
`MPI_MIN` cannot raise a value.

*Termination.* The global label vector is non-increasing and bounded below by
0, and every non-final round strictly decreases at least one entry.

*Fixpoint.* At convergence `comp[u] == comp[w]` for every edge (each edge is
owned by some rank, whose union-find would otherwise have changed a label).
So `comp` is constant on each component. That constant is a member of the
component (soundness) and is `≤` every member (monotonicity, since
`comp[v] ≤ v`), hence it is exactly the minimum. ∎

### 2.4 Data distribution

Vertices are split into contiguous blocks, `first(r) = (r·V)/P`. Integer
arithmetic handles `V` not divisible by `P` and `V < P` (trailing ranks get
zero vertices and simply contribute nothing, while still joining every
collective). Contiguous blocks are preferred over round-robin because they keep
graph locality, so more edges stay internal.

Each rank receives **only its own vertices' adjacency lists** via two
`MPI_Scatterv` calls (degrees, then neighbour entries). The O(E) data — the
dominant term — is partitioned. The O(V) label array is replicated: at
`V ≤ 10^5` that is 400 KB per rank, which buys a single collective in place of
a ghost-vertex exchange protocol.

### 2.5 MPI calls used, and why

| Call | Purpose |
|---|---|
| `MPI_Bcast` | broadcast `V` so every rank can size its arrays |
| `MPI_Scatterv` ×2 | distribute degrees and neighbour entries; per-rank counts differ, so the `v`-variant is required |
| `MPI_Scatter` | tell each rank how many neighbour entries it is about to receive |
| `MPI_Allreduce(MPI_MIN)` | the algorithmic core: merges all ranks' labels and leaves every rank identical |
| `MPI_Barrier`, `MPI_Wtime`, `MPI_Reduce(MPI_MAX)` | benchmarking only |

**No point-to-point calls are used.** Every rank executes an identical sequence
of collectives, so deadlock is structurally impossible.

**No final gather is needed**: after convergence every rank already holds the
complete, identical result, so rank 0 simply prints it.

**Convergence needs no extra collective**: after `MPI_Allreduce` every rank
holds a bit-identical array, so the test `comp == prev` yields the same boolean
on all ranks and they exit together.

### 2.6 Complexity

Let `R` = rounds, `E_local ≈ E/P`.

| | |
|---|---|
| Sequential | O(V + E) time, O(V + E) memory |
| MPI computation | O(R · (V + E/P) · α(V)) |
| MPI communication | R × one `Allreduce` of V ints |
| Memory per rank | 3V ints (≈1.2 MB at V=10^5) + O(E/P) |

Only the `E/P` term shrinks with `P`. The O(V) union-find reset/re-seed/collapse
and the `Allreduce` do not, so **speed-up is expected to be sublinear and to
flatten**; the `t_comm` / `t_compute` instrumentation is there to measure
exactly that.

---

## 3. Build

The programs and the tools (generator, analyzer) are C++; there is **no Python
dependency**. Orchestration stays in shell — running the suites, collecting
timings, and submitting to SLURM.

```bash
module load hpcx-2.7.0/hpcx-ompi      # on the RCE cluster

make                 # builds both programs and all four tools
make test            # build, then run the correctness suite
make bench           # build, then run the benchmark
make analyze         # regenerate tables and SVG plots from results.csv
make clean
```

| Component | Language | Purpose |
|---|---|---|
| `src/sequential`, `src/q6_mpi` | C++ | the programs |
| `tools/gen_graph` (`gen_graph.cpp`) | C++ | seeded generator, 8 topologies |
| `tools/analyze` (`analyze.cpp`) | C++ | tables + SVG plots from `results.csv` |
| `tests/run_tests.sh` | shell | runs the correctness suite |
| `benchmark/run_bench.sh` | shell | runs the timing sweep |
| `slurm/job_q6.sh` | shell | cluster submission |

## 4. Run

```bash
./src/sequential < input.txt > output.txt

mpirun -np 1 ./src/q6_mpi < input.txt > output.txt
mpirun -np 2 ./src/q6_mpi < input.txt > output.txt
mpirun -np 4 ./src/q6_mpi < input.txt > output.txt
mpirun -np 8 ./src/q6_mpi < input.txt > output.txt
```

`--stats` adds timing and round-count diagnostics **on stderr**; stdout always
carries only the required `V` output lines.

```bash
mpirun -np 4 ./src/q6_mpi --stats < input.txt > output.txt 2> timing.txt
```

## 5. SLURM (RCE)

```bash
sbatch slurm/job_q6.sh          # edit PROJECT_DIR inside first
squeue -u $USER                 # status
scontrol show job <JOB_ID>      # details
scancel <JOB_ID>                # cancel
cat job_<JOB_ID>.log            # output   (errors: job_<JOB_ID>.err)
```

The script requests `--nodes=2 --ntasks-per-node=4` = 8 tasks, so P = 1, 2, 4, 8
all run inside one allocation, on the same hardware, making the timings
comparable.

## 6. Correctness testing

```bash
bash tests/run_tests.sh
MPIRUN_FLAGS="--oversubscribe" bash tests/run_tests.sh   # oversubscribed nodes
PROCS="1 2" bash tests/run_tests.sh                      # fewer process counts
```

For each case the suite runs the sequential reference and the MPI program at
P = 1, 2, 4, 8 and diffs the outputs. PASS/FAIL appears only in the suite's own
output; program output format is never altered.

17 cases: PDF sample · single vertex · no edges · fully disconnected · single
component (path) · multiple components · cross-process bridges · uneven split
(V=97 prime) · star (load imbalance) · asymmetric adjacency · V < P (idle
ranks) · random 5k/20k · sparse many-components · long path 20k · random
50k/400k · shuffled path (worst-case round count).

## 7. Dataset generation

All datasets are reproducible from a fixed seed:

```bash
./tools/gen_graph -v 100000 -e 1000000 -t random -s 103 -o big.in
./tools/gen_graph -v 100000 -t path -o path.in
./tools/gen_graph -v 100000 -t chain-blocks -c 16 -o chain.in
```

Topologies: `random`, `path`, `shuffled`, `star`, `disconnected`, `isolated`,
`cluster`, `chain-blocks`. `--asymmetric` writes each edge into only one
adjacency list. The generator uses splitmix64, so a given seed reproduces the
same graph on any platform.

## 8. Benchmarking

```bash
bash benchmark/run_bench.sh            # writes benchmark/results.csv
bash benchmark/run_bench.sh --quick    # skip the out-of-constraint xl_* sizes
./tools/analyze                        # tables + SVG plots (C++)
```

Plots are emitted as **SVG**, written directly by `analyze.cpp` with no
plotting library. SVG embeds in a Markdown or HTML report as is, and converts
to PNG with any standard converter if a raster image is required.

**What is timed.** `t_algo = t_compute + t_comm` is the headline figure: the
union-find work plus the `Allreduce`, excluding file reading and distribution.
`t_input` and `t_total` are reported separately so nothing is hidden. Each
value is the **maximum across ranks** (a parallel phase ends when its slowest
rank ends) and the **median over 5 trials** (robust against one slow run).

`S(P) = T1 / TP` uses the **MPI binary at P=1** as `T1`, so the speed-up
measures parallel scaling of one implementation rather than comparing two
different programs.

## 9. Assumptions and limitations

1. **Undirectedness.** Every pair `(i, vⱼ)` in the input is treated as an
   undirected edge. The PDF does not guarantee the adjacency lists are listed
   symmetrically, so the sequential reference inserts each listed pair in both
   directions, and the MPI version unites each pair symmetrically. A one-sided
   listing is therefore still a real edge in both implementations.
2. **Duplicates and self-loops** are accepted and do not affect the result
   (union-find operations are idempotent).
3. **Input validation.** Negative `k`, neighbour IDs outside `[0,V)`, and
   truncated input are rejected with a message on stderr and a non-zero exit.
4. **Rank 0 reads the whole file** (~12 MB at maximum size) before scattering.
   Scaling far beyond `V = 10^5` would want parallel I/O instead.
5. **Replicated label array** costs O(V) memory per rank and an O(V) `Allreduce`
   per round. This is deliberate at the assignment's scale; for much larger `V`
   a ghost-vertex exchange (`MPI_Alltoallv` over boundary vertices only) would
   reduce both, at considerably more complexity.
6. **32-bit MPI counts.** Guarded explicitly; the assignment limits are two
   orders of magnitude below the threshold.
7. **Round count depends on vertex numbering, not just graph size.** A path
   graph with sequential IDs converges in 2–3 rounds; the same path with
   randomly permuted IDs takes 8–9, because block partitioning no longer
   matches graph locality.

## 10. Project structure

```
Q6/
├── README.md
├── Makefile                builds everything
├── src/
│   ├── sequential.cpp      BFS reference implementation (oracle)
│   └── q6_mpi.cpp          MPI implementation
├── tools/
│   ├── harness.hpp         statistics, CSV, SVG plotting (C++)
│   ├── gen_graph.cpp       seeded generator, 8 topologies
│   └── analyze.cpp         tables + SVG plots
├── tests/
│   ├── sample.in           the PDF sample
│   ├── sample.expected     the PDF expected output
│   └── run_tests.sh        17 cases at P=1,2,4,8
├── benchmark/
│   ├── run_bench.sh        timing sweep, writes results.csv
│   └── plots/              (produced by tools/analyze)
├── slurm/
│   └── job_q6.sh           compile + test + benchmark in one allocation
└── report/
    └── report_template.md  report skeleton to fill from measured results
```
