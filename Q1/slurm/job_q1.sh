#!/bin/bash
#SBATCH --job-name=q1-rowrow-matmul
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err
#SBATCH --partition=debug
# ---------------------------------------------------------------------------
# Q1: Row-Row matrix multiplication -- correctness suite + benchmark.
#
# The allocation is 2 nodes x 4 tasks = 8 tasks, so P = 1, 2, 4, 8 all fit in a
# SINGLE job. Running every process count in one allocation means all four data
# points come from the same hardware under the same conditions, which is what
# makes the speed-up numbers comparable.
#
# Submit:   sbatch slurm/job_q1.sh
# Status:   squeue -u $USER
# Details:  scontrol show job <JOB_ID>
# Cancel:   scancel <JOB_ID>
# Output:   cat job_<JOB_ID>.log      (errors: job_<JOB_ID>.err)
#
# NOTE: edit PROJECT_DIR below to your own path before submitting.
#
# Memory note: at 1500x1500 with 4-byte ints, B is ~9 MB and is replicated on
# every process; C is 8-byte and ~18 MB on root. --mem-per-cpu=4G is ample.
# ---------------------------------------------------------------------------

PROJECT_DIR="$HOME/DS_HW2/Q1"

module load hpcx-2.7.0/hpcx-ompi

# This cluster's mpirun miscounts available slots/cpu-bindings once more than
# one rank lands on a node (fails at P>=4 with a core-binding error); disabling
# process binding avoids it and does not affect correctness.
export MPIRUN_FLAGS="--bind-to none"

cd "$PROJECT_DIR" || { echo "PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }

echo "========================================="
echo "SLURM Job ID   : $SLURM_JOB_ID"
echo "Nodes          : $SLURM_NNODES"
echo "Total tasks    : $SLURM_NTASKS"
echo "Node list      : $SLURM_NODELIST"
echo "Started        : $(date)"
echo "========================================="

# ---------------------------------------------------------------- compile ---
echo ""
echo "--- Compiling ---"
make all || { echo "build failed"; exit 1; }
echo "Build OK"

# ----------------------------------------------------------- correctness ---
# The suite uses mpirun so it can launch varying process counts from inside
# the job. If your site requires srun, replace the mpirun calls in
# tests/run_tests.sh and benchmark/run_bench.sh with "srun -n <P>".
echo ""
echo "--- Correctness suite (P = 1 2 4 8) ---"
bash tests/run_tests.sh || echo "WARNING: correctness suite reported failures"

# ------------------------------------------------------------- benchmark ---
echo ""
echo "--- Benchmark ---"
# Drop --quick to include the verylarge (1500^3) and skew_tall sizes.
bash benchmark/run_bench.sh --quick

echo ""
echo "--- Analysis ---"
./tools/analyze | tee report/benchmark_results.md

echo ""
echo "========================================="
echo "Finished       : $(date)"
echo "Results        : benchmark/results.csv"
echo "Tables/plots   : report/benchmark_results.md , benchmark/plots/"
echo "========================================="
