#!/bin/bash
#SBATCH --job-name=q7-log-analytics
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err
#SBATCH --partition=debug
# ---------------------------------------------------------------------------
# Q7: Server log analytics -- correctness suite + benchmark, one allocation.
#
# The allocation is 2 nodes x 4 tasks = 8 tasks, so P = 1, 2, 4, 8 all fit
# inside a SINGLE job. Running every process count in one allocation means all
# four data points come from the same hardware under the same conditions,
# which is what makes the speed-up numbers comparable.
#
# Submit:   sbatch slurm/job_q7.sh
# Status:   squeue -u $USER
# Details:  scontrol show job <JOB_ID>
# Cancel:   scancel <JOB_ID>
# Output:   cat job_<JOB_ID>.log     (errors: job_<JOB_ID>.err)
#
# NOTE: edit PROJECT_DIR below to your own path before submitting.
# ---------------------------------------------------------------------------

# Memory note: root holds the whole log before scattering. At 10,000,000
# records x 40 bytes that is ~400 MB on rank 0; --mem-per-cpu=4G covers it.

PROJECT_DIR="$HOME/DS_HW2/Q7"

module load hpcx-2.7.0/hpcx-ompi

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
# Uses mpirun so the suite can launch varying process counts from inside the
# job. If your site requires srun instead, set:
#     change mpiCommand() in tools/harness.hpp to emit "srun -n <P>" -- it is
#     the single place the launcher is named.
echo ""
echo "--- Correctness suite (P = 1 2 4 8) ---"
bash tests/run_tests.sh || echo "WARNING: correctness suite reported failures"

# ------------------------------------------------------------- benchmark ---
echo ""
echo "--- Benchmark ---"
# Drop --quick to include the 5M and 10M record sizes (needs more memory and
# time; check the --mem-per-cpu and --time above first).
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
