#!/bin/bash
#SBATCH --job-name=q6-conncomp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err
#SBATCH --partition=debug
# ---------------------------------------------------------------------------
# Q6: Connected Components -- correctness suite + benchmark, one allocation.
#
# The allocation is 2 nodes x 4 tasks = 8 tasks, so P = 1, 2, 4, 8 all fit
# inside a SINGLE job. Running every process count in one allocation means all
# four data points come from the same hardware under the same conditions,
# which is what makes the speed-up numbers comparable.
#
# Submit:   sbatch slurm/job_q6.sh
# Status:   squeue -u $USER
# Details:  scontrol show job <JOB_ID>
# Cancel:   scancel <JOB_ID>
# Output:   cat job_<JOB_ID>.log     (errors: job_<JOB_ID>.err)
#
# NOTE: edit PROJECT_DIR below to your own path before submitting.
# ---------------------------------------------------------------------------

PROJECT_DIR="$HOME/Q6"        # <-- EDIT THIS

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
mpicxx -O2 -std=c++17 -o src/q6_mpi     src/q6_mpi.cpp     || { echo "MPI build failed";        exit 1; }
g++    -O2 -std=c++17 -o src/sequential src/sequential.cpp || { echo "sequential build failed"; exit 1; }
echo "Build OK"

# ----------------------------------------------------------- correctness ---
# Uses mpirun so the suite can launch varying process counts from inside the
# job. If your site requires srun instead, set:
#     export MPIRUN_FLAGS=""   and edit tests/run_tests.sh to call srun -n
echo ""
echo "--- Correctness suite (P = 1 2 4 8) ---"
bash tests/run_tests.sh || echo "WARNING: correctness suite reported failures"

# ------------------------------------------------------------- benchmark ---
echo ""
echo "--- Benchmark ---"
# Drop --quick to include the out-of-constraint xl_* scaling study
# (needs more memory and time; check the --mem-per-cpu and --time above first).
bash benchmark/run_bench.sh --quick

echo ""
echo "--- Analysis ---"
python3 benchmark/analyze.py | tee report/benchmark_results.md

echo ""
echo "========================================="
echo "Finished       : $(date)"
echo "Results        : benchmark/results.csv"
echo "Tables/plots   : report/benchmark_results.md , benchmark/plots/"
echo "========================================="
