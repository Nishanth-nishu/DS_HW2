# Distributed Systems — Home Work 2

MPI implementations for the assigned questions, each self-contained.

```
HW2/
├── Q1/    Distributed Matrix Multiplication — Row-Row Method   (Section 1)
├── Q6/    Connected Components of a Large Graph                (Section 2)
└── Q7/    Large-Scale Server Log Analytics                     (Section 3)
```

Each directory has the same layout and its own full `README.md`:

```
<Q>/
├── README.md               problem, algorithm, correctness argument, assumptions
├── Makefile                builds programs and tools
├── src/                    sequential reference + MPI implementation
├── tools/                  test runner, benchmark runner, analyzer, generator
├── tests/                  input fixtures and expected outputs
├── benchmark/              results.csv and SVG plots land here
├── slurm/                  SLURM job script for the RCE cluster
└── report/                 report template to fill from measured results
```

## No Python

All programs and tools are C++, built by `make`. Orchestration stays in shell.

| Component | Language | Purpose |
|---|---|---|
| `src/sequential`, `src/*_mpi` | C++ | the implementations |
| `tools/analyze` | C++ | Markdown tables + SVG charts from `results.csv` |
| `tools/gen_graph` (Q6), `tools/gen_matrix` (Q1), `tools/gen_log` (Q7) | C++ | reproducible seeded input generation |
| `tests/run_tests.sh` | shell | correctness suite: MPI vs sequential at P = 1, 2, 4, 8 |
| `benchmark/run_bench.sh` | shell | timing sweep, writes `benchmark/results.csv` |
| `slurm/job_*.sh` | shell | cluster submission |

Plots are emitted as **SVG**, written directly by `analyze.cpp` — no plotting
library, no matplotlib. SVG embeds in a Markdown or HTML report as is, and
converts to PNG with any standard converter if a raster is required.

`tools/harness.hpp` is the analyzer's shared infrastructure: medians, CSV
parsing, Markdown table emission, and the SVG chart writer.

## Quick start (either question)

```bash
module load hpcx-2.7.0/hpcx-ompi

cd Q1                 # or Q6
make                  # builds programs and tools
make test             # correctness at P = 1, 2, 4, 8
make bench            # timing runs -> benchmark/results.csv
make analyze          # tables + SVG plots
```

Run the tools from the question's project root; they use relative paths.
`MPIRUN_FLAGS` and `PROCS` override the launcher flags and process counts.

## On the cluster

```bash
# edit PROJECT_DIR at the top of the job script first
sbatch Q1/slurm/job_q1.sh
sbatch Q6/slurm/job_q6.sh
sbatch Q7/slurm/job_q7.sh

squeue -u $USER                     # status
scontrol show job <JOB_ID>          # details
scancel <JOB_ID>                    # cancel
cat job_<JOB_ID>.log                # output   (errors: job_<JOB_ID>.err)
```

Each job script requests `--nodes=2 --ntasks-per-node=4` = 8 tasks, so
P = 1, 2, 4, 8 all run inside **one** allocation, on the same hardware — which
is what makes the speed-up figures comparable.

## Current status

| | Q1 (Row-Row matmul) | Q6 (Connected components) | Q7 (Log analytics) |
|---|---|---|---|
| Sequential reference | done | done | done |
| MPI implementation | done | done | done |
| Correctness suite | 25 cases, all pass | 17 cases, all pass | 20 cases, all pass |
| Verified against | both PDF examples, exactly | PDF sample, exactly | hand-computed fixture |
| Benchmark harness | done | done | done |
| SLURM script | done | done | done |
| Tooling | C++ + shell orchestration | C++ + shell orchestration | C++ + shell orchestration |
| README | done | done | done |
| Report | **template — needs your numbers** | **template — needs your numbers** | **template — needs your numbers** |

**Q7 note:** the PDF supplies no sample data for Q7 and never defines `S` in the
`N K S` header. The expected output in `Q7/tests/small.expected` was computed by
hand; `S` is parsed but not relied upon. See `Q7/README.md` §2 for the full list
of points the PDF leaves open.

## No benchmark numbers are included

`results.csv` files and plots are deliberately absent. They must be produced by
your own runs on the RCE cluster. Every `TBD` in the report templates marks a
figure that has to come from a real measurement.
