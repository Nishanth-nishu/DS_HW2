# Q7 — Large-Scale Server Log Analytics (MPI)

Distributed Systems, Home Work 2, Section 3 (Real-World Applications).

Compute a set of aggregate statistics over a large server access log, using MPI
to distribute the records across processes. Both a sequential and an MPI
implementation are provided, along with a reproducible dataset generator.

---

## 1. Problem statement

**Input**

```
N K S
timestamp server_id endpoint_id user_id status_code response_time bytes_sent
... N such lines
```

A request is **successful** if `status_code < 400`. The interval ID is
`timestamp / 60`.

**Required computations** (from the PDF)

- total, successful and failed requests
- average, minimum and maximum response time
- total bytes sent, and counts of 2xx, 3xx, 4xx and 5xx responses
- top-K servers by request count, with average response time
- top-K endpoints by request count, with total bytes
- busiest 60-second interval

**Output** — exactly the keys listed in the PDF, in the order given, with
`TOP_SERVERS` and `TOP_ENDPOINTS` sections following. Top-K results are sorted
by decreasing count, then increasing ID.

## 2. Points the PDF leaves open

These are **implementation choices**, not requirements taken from the
assignment:

1. **`S` is never defined.** The header is `N K S`; `N` is the record count and
   `K` the top-K parameter, but `S` has no stated meaning (most likely the
   server count). Both programs **parse `S` and do not depend on it** —
   aggregation uses the IDs actually present in the data — so the result is
   correct whatever was intended. The generator uses it as the number of
   distinct servers.
2. **No sample input or output is supplied.** Section 3 refers to "the provided
   samples", but the PDF gives none for Q7. `tests/small.expected` was
   therefore computed **by hand** from `tests/small.in` and is used as the
   oracle; everything else is verified against the sequential implementation.
3. **Field types are not stated.** All seven fields are treated as integers
   (`response_time` in milliseconds), consistent with the rest of the
   assignment. `timestamp` and `bytes_sent` are 64-bit because they exceed or
   accumulate beyond 2^31.
4. **Number formatting is not specified.** `AVERAGE_RESPONSE_TIME` and the
   per-server average print with **6 decimal places**; all counts, byte totals
   and min/max response times print as integers.
5. **`BUSIEST_INTERVAL` tie-breaking is not specified** (top-K tie-breaking
   *is*). Ties are broken by **smallest interval ID**, consistent with the
   "then increasing ID" rule used elsewhere.
6. **Fewer than K distinct IDs**: all of them are printed. `K = 0` prints none.
7. **`N = 0`**: averages print as `0.000000` and min/max as `0`.
8. Status codes outside 200–599 are counted in `TOTAL_REQUESTS` and in
   successful/failed, but in none of the four `STATUS_nXX` buckets, since the
   PDF defines only those four.

## 3. Why this parallelises cleanly

Every required statistic is a **commutative and associative fold** over the
records — a count, a sum, a minimum or a maximum — either globally or per key.
So each rank can aggregate its own slice with no communication at all, and the
partial results combine exactly. No record is examined by more than one rank,
and no second pass is needed.

The one step that **cannot** be done per rank is top-K selection: a server
ranked 3rd on every rank could still be 1st globally. So the ranks exchange
full per-key counts, and only the **root** selects top-K, after merging.

## 4. Parallel structure

```
root      reads or generates the log
Bcast     header values N, K, S
Scatterv  the records (per-rank counts differ when N % P != 0)
local     each rank aggregates its own slice -- NO communication
Reduce    scalars: SUM for counts and sums, MIN and MAX for response time
Gatherv   per-rank (id, count, extra) triples for servers, endpoints, intervals
root      merges the maps, selects top-K, writes the report
```

### Record distribution

Contiguous blocks, remainder to the earliest ranks:

```
base = N / P     rem = N % P
count(r) = base + (r < rem ? 1 : 0)
start(r) = r*base + min(r, rem)
```

This handles `N` not divisible by `P`, and `N < P` (trailing ranks receive zero
records and still join every collective — skipping one would hang the job).

Records are a contiguous array of a POD struct, so `MPI_Scatterv` moves them
with `MPI_BYTE` and no derived datatype is required. On a homogeneous cluster
this is exact; `MPI_Type_create_struct` would be the portable alternative for
heterogeneous systems.

### Merging the per-key maps

Each rank flattens its map into `(id, count, extra)` triples; `MPI_Gatherv`
brings them to root, which sums entries key by key. Communication is
**O(P × distinct_keys)**, not O(N) — for server logs the number of distinct
servers, endpoints and intervals is far smaller than the record count.

Gathering per key, rather than reducing a dense array indexed by ID, means **no
assumption is made about ID range or density**. A log with server IDs in the
billions works exactly as well as one with IDs 0..63.

### MPI calls, and why

| Call | Purpose |
|---|---|
| `MPI_Bcast` | header values N, K, S — needed before any rank can size buffers |
| `MPI_Scatterv` | records; per-rank counts differ when `N % P != 0` |
| `MPI_Reduce` (SUM) | the nine scalar counters and sums |
| `MPI_Reduce` (MIN/MAX) | min and max response time |
| `MPI_Gather` / `MPI_Gatherv` | per-key triples for the three maps |
| `MPI_Barrier`, `MPI_Wtime` | benchmarking only |

**Deadlock safety.** Only collectives are used, and every rank executes an
identical unconditional sequence of them. A rank with zero records contributes
`minResponse = +inf` and `maxResponse = -inf`, the identities for `MPI_MIN` and
`MPI_MAX`, so idle ranks cannot corrupt the result either.

## 5. Complexity and memory

| | |
|---|---|
| Sequential | O(N) to accumulate + O(D log D) to sort D distinct IDs |
| Per process | O(N/P) accumulate |
| Scatterv | N records distributed |
| Reduce | 11 scalars, constant |
| Gatherv | O(P × D) triples |
| Root memory | O(N) before scattering, plus O(D) merged |
| Worker memory | O(N/P) + O(D_local) |

The aggregation is O(N/P) and shrinks with P; the map merge grows as O(P × D).
For small D (few servers) the merge is negligible and scaling should be good;
for large D it becomes the limiting term. The `keys_few` / `keys_many` benchmark
rows are designed to measure exactly this.

## 6. Build

The programs and the tools (generator, analyzer) are C++; there is **no Python
dependency**. Orchestration stays in shell.

```bash
module load hpcx-2.7.0/hpcx-ompi      # on the RCE cluster

make                 # builds both programs and both tools
make test            # correctness suite
make bench           # timing sweep
make analyze         # tables + SVG plots
make clean
```

| Component | Language | Purpose |
|---|---|---|
| `src/sequential`, `src/q7_mpi` | C++ | the programs |
| `tools/gen_log` (`gen_log.cpp`) | C++ | seeded log generator |
| `tools/analyze` (`analyze.cpp`) | C++ | tables + SVG plots |
| `tests/run_tests.sh` | shell | correctness suite |
| `benchmark/run_bench.sh` | shell | timing sweep |
| `slurm/job_q7.sh` | shell | cluster submission |

## 7. Run

```bash
./src/sequential < log.txt > report.txt

mpirun -np 1 ./src/q7_mpi < log.txt > report.txt
mpirun -np 2 ./src/q7_mpi < log.txt > report.txt
mpirun -np 4 ./src/q7_mpi < log.txt > report.txt
mpirun -np 8 ./src/q7_mpi < log.txt > report.txt

# generated log, no report printed, timings on stderr
mpirun -np 4 ./src/q7_mpi --gen 1000000 10 64 --stats --no-output
```

Options: `--stats` (timings to **stderr**), `--no-output` (suppress the
report), `--gen N K S [SEED]`. stdout always carries only the required report.

## 8. Dataset generation

```bash
./tools/gen_log 1000000 10 64 2024 > log.txt
```

Deterministic splitmix64, so the same `(N, K, S, seed)` reproduces the same log
on any platform, in the tool and in both programs' `--gen` mode.

Derived parameters (implementation choices): servers in `[0, S)`, endpoints in
`[0, 4S)`, users in `[0, 100S)`, timestamps spread over a 24-hour window from a
fixed epoch base, status codes ~80% 2xx / 8% 3xx / 8% 4xx / 4% 5xx,
`response_time` in `[1, 1000]`, `bytes_sent` in `[100, 100099]`.

## 9. SLURM (RCE)

```bash
sbatch slurm/job_q7.sh          # edit PROJECT_DIR inside first
squeue -u $USER                 # status
scontrol show job <JOB_ID>      # details
scancel <JOB_ID>                # cancel
cat job_<JOB_ID>.log            # output   (errors: job_<JOB_ID>.err)
```

`--nodes=2 --ntasks-per-node=4` = 8 tasks, so P = 1, 2, 4, 8 all run inside one
allocation on the same hardware.

## 10. Correctness verification

```bash
bash tests/run_tests.sh
MPIRUN_FLAGS="--oversubscribe" bash tests/run_tests.sh
PROCS="1 2" bash tests/run_tests.sh
```

20 cases at P = 1, 2, 4, 8: the hand-computed fixture (which exercises every
output field plus both tie-breaks), then structural edge cases — `N=0`, `N=1`,
`N < P`, `K=0`, `K` greater than the distinct ID count, a single server, a
three-way busiest-interval tie, a 1xx status outside the four buckets — then
generated logs including prime and odd `N`, one server, 500 servers, and up to
500,000 records. Finally three input-validation cases.

Comparison is byte-for-byte. Every partial sum is an exact integer until the
final division, and both programs format to a fixed 6 decimal places, so the
match is exact rather than approximate.

## 11. Benchmarking

```bash
bash benchmark/run_bench.sh            # writes benchmark/results.csv
bash benchmark/run_bench.sh --quick    # skip the two largest sizes
./tools/analyze                        # tables + SVG plots
```

**What is timed.** `t_algo = t_compute + t_comm`, where
`t_comm = t_dist + t_reduce`. Each value is the **maximum across ranks** and the
**median over 5 trials**. Logs are generated in-process and `--no-output`
suppresses the report, so neither file parsing nor printing pollutes the
measurement.

`S(P) = T1 / TP` uses the **MPI binary at P=1** as `T1`, so the speed-up
measures parallel scaling of one implementation rather than comparing two
different programs.

Existing `results.csv` is never overwritten — it is renamed with a timestamp.

## 12. Assumptions and limitations

1. `S` is parsed but unused (§2.1).
2. Field types, number formatting, tie-breaking and the `N=0` case are
   documented choices (§2.3–2.7).
3. **Root reads or generates the whole log** before scattering, so root's
   memory bounds the largest solvable size. At 10,000,000 records × 40 bytes
   that is ~400 MB. Parallel I/O would be the next step for larger logs.
4. **32-bit MPI counts** are checked explicitly: `N × sizeof(Record)` must fit
   in `int`, which caps a single scatter at roughly 53 million records.
5. Merge cost is O(P × D); very high key cardinality shifts the bottleneck from
   aggregation to the merge.
6. `MPI_BYTE` transfer of a POD struct assumes a homogeneous cluster.

## 13. Project structure

```
Q7/
├── Makefile
├── README.md
├── src/
│   ├── log_io.hpp          record type, parser, generator, aggregation, output
│   ├── sequential.cpp      sequential reference (oracle)
│   └── q7_mpi.cpp          MPI implementation
├── tools/
│   ├── harness.hpp         statistics, CSV, SVG plotting (C++)
│   ├── gen_log.cpp         seeded log generator
│   └── analyze.cpp         tables + SVG plots
├── tests/
│   ├── small.in            hand-built log
│   ├── small.expected      hand-computed expected output
│   └── run_tests.sh        20 cases at P=1,2,4,8
├── benchmark/
│   ├── run_bench.sh        timing sweep, writes results.csv
│   └── plots/              (produced by tools/analyze)
├── slurm/
│   └── job_q7.sh
└── report/
    └── report_template.md
```
