# Distributed Systems — Home Work 2

MPI implementations for the two assigned questions, each self-contained.

```
HW2/
├── Q1/    Distributed Matrix Multiplication — Row-Row Method   (Section 1)
└── Q6/    Connected Components of a Large Graph                (Section 2)
```

Each directory has the same layout and its own full `README.md`:

```
<Q>/
├── README.md               problem, algorithm, correctness argument, assumptions
├── src/                    sequential reference + MPI implementation + Makefile
├── tests/                  fixtures and the correctness suite (P = 1,2,4,8)
├── tools/                  dataset generator (Q6 only; Q1 generates in-process)
├── benchmark/              benchmark harness, analysis script, plots
├── slurm/                  SLURM job script for the RCE cluster
└── report/                 report template to fill from measured results
```

## Quick start (either question)

```bash
module load hpcx-2.7.0/hpcx-ompi

cd Q1/src && make && cd ..          # or Q6
bash tests/run_tests.sh             # correctness at P = 1, 2, 4, 8
```

## On the cluster

```bash
# edit PROJECT_DIR at the top of the job script first
sbatch Q1/slurm/job_q1.sh
sbatch Q6/slurm/job_q6.sh

squeue -u $USER                     # status
scontrol show job <JOB_ID>          # details
scancel <JOB_ID>                    # cancel
cat job_<JOB_ID>.log                # output   (errors: job_<JOB_ID>.err)
```

Each job script requests `--nodes=2 --ntasks-per-node=4` = 8 tasks, so
P = 1, 2, 4, 8 all run inside **one** allocation, on the same hardware — which
is what makes the speed-up figures comparable.

## Current status

| | Q1 (Row-Row matmul) | Q6 (Connected components) |
|---|---|---|
| Sequential reference | done | done |
| MPI implementation | done | done |
| Correctness suite | 22 cases, all pass at P=1,2,4,8 | 16 cases, all pass at P=1,2,4,8 |
| Matches PDF worked examples | both examples, exactly | sample, exactly |
| Benchmark harness | done | done |
| SLURM script | done | done |
| README | done | done |
| Report | **template only — needs your measurements** | **template only — needs your measurements** |

## No benchmark numbers are included

`results.csv` files and plots are deliberately absent. They must be produced by
your own runs on the RCE cluster. Every `TBD` in the report templates marks a
figure that has to come from a real measurement.
