// q1_mpi.cpp
// ---------------------------------------------------------------------------
// Q1: Distributed Matrix Multiplication -- ROW-ROW METHOD (MPI)
//
// Method (as defined in the assignment PDF):
//     c_i = a_i[0]*B[0,:] + a_i[1]*B[1,:] + ... + a_i[n-1]*B[n-1,:]
// Each row of C is a weighted sum of the ROWS of B, with the weights taken
// from the corresponding row of A. Rows of C are therefore independent, so a
// process holding some rows of A plus a full copy of B can compute the
// corresponding rows of C with NO communication with any other worker.
//
// Parallel structure:
//     root      reads/generates A and B
//     Bcast     dimensions m, n, p
//     Scatterv  rows of A (varying counts: m need not be divisible by P)
//     Bcast     the whole of B to every process
//     local     each rank computes its own row-slice of C, independently
//     Gatherv   row-slices of C collected on root, in order
//
// Only collective communication is used, so the classic Send/Recv ordering
// deadlock cannot occur. Every rank executes an identical, unconditional
// sequence of collectives -- including ranks that receive zero rows.
//
// Build: mpicxx -O2 -std=c++17 -o q1_mpi q1_mpi.cpp
//
// Run (problem on stdin, read by root):
//     mpirun -np 4 ./q1_mpi < input.txt > output.txt
//
// Run (problem generated in-process; identical matrices to sequential):
//     mpirun -np 4 ./q1_mpi --gen 1000 1000 1000 [SEED] > output.txt
//
// Options:
//     --stats      timing to stderr (stdout carries only matrix C)
//     --no-output  suppress C, for timing runs
// ---------------------------------------------------------------------------

#include <mpi.h>

#include <cstring>
#include <vector>

#include "matrix_io.hpp"

// ---------------------------------------------------------------------------
// Row distribution.
//
// The remainder is given to the EARLIEST ranks, matching the PDF's Example 2
// (m = 4, P = 3  ->  P0 gets 2 rows, P1 gets 1, P2 gets 1):
//     base = m / P,  rem = m % P
//     count(r) = base + (r < rem ? 1 : 0)
//     start(r) = r*base + min(r, rem)
//
// This also handles m < P correctly: trailing ranks get count 0. Such ranks
// still take part in every collective -- skipping one would hang the job.
// ---------------------------------------------------------------------------
static inline long long rowCount(int r, long long m, int P) {
    return m / P + (r < m % P ? 1 : 0);
}
static inline long long rowStart(int r, long long m, int P) {
    long long base = m / P, rem = m % P;
    return r * base + (r < rem ? r : rem);
}

[[noreturn]] static void mpiDie(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // ----------------------------- options ------------------------------
    bool stats = false, noOutput = false, useGen = false;
    long long gm = 0, gn = 0, gp = 0;
    uint64_t seed = 12345;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--stats") == 0) {
            stats = true;
        } else if (std::strcmp(argv[i], "--no-output") == 0) {
            noOutput = true;
        } else if (std::strcmp(argv[i], "--gen") == 0) {
            if (i + 3 >= argc) mpiDie("--gen needs M N P [SEED]");
            useGen = true;
            gm = atoll(argv[i + 1]);
            gn = atoll(argv[i + 2]);
            gp = atoll(argv[i + 3]);
            i += 3;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                seed = strtoull(argv[++i], nullptr, 10);
        } else {
            mpiDie("unknown option");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    // =====================================================================
    // STEP 1: root obtains A and B
    // =====================================================================
    Problem prob;
    if (rank == 0) {
        if (useGen) {
            if (gm < 0 || gn < 0 || gp < 0)
                mpiDie("dimensions must be non-negative");
            prob.m = gm; prob.n = gn; prob.p = gp;
            generateMatrix(prob.A, gm * gn, seed);
            generateMatrix(prob.B, gn * gp, seed ^ 0xA5A5A5A5A5A5A5A5ULL);
        } else {
            readProblem(stdin, prob);
        }
    }

    // =====================================================================
    // STEP 2: broadcast the dimensions
    //   Nobody can size a buffer without these, so this must precede every
    //   other transfer. One Bcast of 3 values; the cost is pure latency.
    // =====================================================================
    long long dims[3] = { prob.m, prob.n, prob.p };
    MPI_Bcast(dims, 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    const long long m = dims[0], n = dims[1], p = dims[2];

    // MPI-3 collective counts are int. Guard explicitly rather than letting a
    // silent overflow corrupt the transfer.
    if (m * n > 2147483647LL || n * p > 2147483647LL || m * p > 2147483647LL)
        mpiDie("matrix too large for 32-bit MPI counts");

    if (m == 0 || p == 0) {          // nothing to compute; exit collectively
        MPI_Finalize();
        return 0;
    }

    // =====================================================================
    // STEP 3: counts and displacements for Scatterv (A) and Gatherv (C)
    //   Units are ELEMENTS, not rows: a row of A is n wide, a row of C is p.
    //   displs must be the running prefix sum of the counts, so that the
    //   pieces tile the buffer exactly.
    // =====================================================================
    const long long myRows  = rowCount(rank, m, P);
    const long long myStart = rowStart(rank, m, P);

    std::vector<int> aCounts(P), aDispls(P), cCounts(P), cDispls(P);
    for (int r = 0; r < P; r++) {
        const long long cnt = rowCount(r, m, P);
        const long long st  = rowStart(r, m, P);
        aCounts[r] = static_cast<int>(cnt * n);
        aDispls[r] = static_cast<int>(st  * n);
        cCounts[r] = static_cast<int>(cnt * p);
        cDispls[r] = static_cast<int>(st  * p);
    }

    // =====================================================================
    // STEP 4: distribute the rows of A  (MPI_Scatterv)
    //   Scatterv rather than Scatter because per-rank counts differ whenever
    //   m % P != 0. Buffers are sized max(1, ...) so that a rank with zero
    //   rows still has a valid (never dereferenced) pointer to pass.
    // =====================================================================
    std::vector<Elem> myA(static_cast<size_t>(myRows * n > 0 ? myRows * n : 1));
    MPI_Scatterv(rank == 0 ? prob.A.data() : nullptr,
                 aCounts.data(), aDispls.data(), MPI_INT,
                 myA.data(), static_cast<int>(myRows * n), MPI_INT,
                 0, MPI_COMM_WORLD);

    // =====================================================================
    // STEP 5: broadcast B in full  (MPI_Bcast)
    //   Row i of C is a combination of EVERY row of B, so no process can
    //   compute a complete row of C from a partial B. Replication is inherent
    //   to the Row-Row method -- and it is the cost that does not shrink as P
    //   grows, which is what limits the speed-up.
    // =====================================================================
    std::vector<Elem> B(static_cast<size_t>(n * p > 0 ? n * p : 1));
    if (rank == 0 && n * p > 0)
        std::memcpy(B.data(), prob.B.data(), sizeof(Elem) * (size_t)(n * p));
    MPI_Bcast(B.data(), static_cast<int>(n * p), MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_after_dist = MPI_Wtime();

    // =====================================================================
    // STEP 6: local computation -- no communication whatsoever
    // =====================================================================
    std::vector<Acc> myC(static_cast<size_t>(myRows * p > 0 ? myRows * p : 1));
    rowRowMultiply(myA.data(), B.data(), myC.data(), myRows, n, p);

    const double t_after_compute = MPI_Wtime();

    // =====================================================================
    // STEP 7: gather the row-slices of C  (MPI_Gatherv)
    //   Mirror image of the scatter, in units of p. Because every rank uses
    //   the same start/count formula, slices land in the right place and C is
    //   reassembled in row order with no sorting or tagging.
    // =====================================================================
    std::vector<Acc> C;
    if (rank == 0) C.resize(static_cast<size_t>(m * p));
    MPI_Gatherv(myC.data(), static_cast<int>(myRows * p), MPI_LONG_LONG,
                rank == 0 ? C.data() : nullptr,
                cCounts.data(), cDispls.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    const double t_after_gather = MPI_Wtime();

    // =====================================================================
    // STEP 8: output -- root only, matrix C alone on stdout
    // =====================================================================
    if (rank == 0 && !noOutput) writeMatrix(stdout, C.data(), m, p);

    const double t_end = MPI_Wtime();

    // =====================================================================
    // Instrumentation. stderr only, and only with --stats, so that the
    // required program output is never contaminated.
    //   t_dist    = Bcast(dims) + Scatterv(A) + Bcast(B)
    //   t_compute = the local Row-Row multiply
    //   t_gather  = Gatherv(C)
    //   t_comm    = t_dist + t_gather
    //   t_algo    = t_compute + t_comm   <- the headline parallel figure
    // A parallel phase ends when its slowest rank ends, so we report the MAX.
    // =====================================================================
    if (stats) {
        double loc[5] = { t_after_dist    - t_start,
                          t_after_compute - t_after_dist,
                          t_after_gather  - t_after_compute,
                          t_end           - t_start,
                          static_cast<double>(myRows) };
        double mx[5];
        MPI_Reduce(loc, mx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double mnRows;
        MPI_Reduce(&loc[4], &mnRows, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            const double comm = mx[0] + mx[2];
            const double algo = mx[1] + comm;
            fprintf(stderr, "[mpi] P=%d m=%lld n=%lld p=%lld\n", P, m, n, p);
            fprintf(stderr, "[mpi] rows/rank  = min %.0f max %.0f\n",
                    mnRows, mx[4]);
            fprintf(stderr, "[mpi] t_dist     = %.3f ms (Bcast dims + "
                            "Scatterv A + Bcast B)\n", mx[0] * 1e3);
            fprintf(stderr, "[mpi] t_compute  = %.3f ms\n", mx[1] * 1e3);
            fprintf(stderr, "[mpi] t_gather   = %.3f ms\n", mx[2] * 1e3);
            fprintf(stderr, "[mpi] t_comm     = %.3f ms\n", comm * 1e3);
            fprintf(stderr, "[mpi] t_algo     = %.3f ms\n", algo * 1e3);
            fprintf(stderr, "[mpi] t_total    = %.3f ms\n", mx[3] * 1e3);
            fprintf(stderr, "[mpi] comm_pct   = %.1f %%\n",
                    algo > 0 ? 100.0 * comm / algo : 0.0);
            fprintf(stderr, "[csv] %d,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f\n",
                    P, m, n, p, mx[0], mx[1], mx[2], mx[3]);
        }
    }

    MPI_Finalize();
    return 0;
}
