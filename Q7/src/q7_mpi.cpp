// q7_mpi.cpp
// ---------------------------------------------------------------------------
// Q7: Large-Scale Server Log Analytics -- MPI IMPLEMENTATION
//
// Parallel structure:
//     root      reads/generates the log
//     Bcast     header values N, K, S
//     Scatterv  the records (varying counts: N need not divide by P)
//     local     each rank accumulates its own partial aggregate, with NO
//               communication -- every required statistic is decomposable
//     Reduce    scalars combined with SUM / MIN / MAX
//     Gatherv   per-rank (id, count, extra) triples for servers, endpoints and
//               intervals; root merges them and selects top-K
//     root      writes the report
//
// WHY THIS DECOMPOSES CLEANLY
// Every required statistic is either a commutative-associative fold over the
// records (counts, sums, min, max) or a per-key fold of the same kind. So a
// rank can aggregate its own slice independently and the partial results
// combine exactly. Nothing needs a second pass over the data, and no record is
// examined by more than one rank.
//
// The one thing that CANNOT be done per-rank is top-K selection: a server that
// is 3rd on every rank could still be 1st globally. So the ranks reduce full
// per-key counts, and only the ROOT selects top-K, after merging.
//
// Only collectives are used, so the Send/Recv ordering deadlock cannot occur.
// Every rank executes an identical unconditional sequence of them, including
// ranks that receive zero records.
//
// Build: mpicxx -O2 -std=c++17 -o q7_mpi q7_mpi.cpp
// Run  : mpirun -np 4 ./q7_mpi < input.txt > output.txt
//        mpirun -np 4 ./q7_mpi --gen 1000000 10 64 > output.txt
// Options: --stats (timings to stderr), --no-output (suppress the report)
// ---------------------------------------------------------------------------

#include <mpi.h>

#include <cstring>

#include "log_io.hpp"

// Record distribution: contiguous blocks, remainder to the earliest ranks.
//   base = N / P,  rem = N % P
// Handles N not divisible by P, and N < P (trailing ranks get zero records,
// and still take part in every collective).
static inline long long recCount(int r, long long N, int P) {
    return N / P + (r < N % P ? 1 : 0);
}
static inline long long recStart(int r, long long N, int P) {
    const long long base = N / P, rem = N % P;
    return r * base + (r < rem ? r : rem);
}

[[noreturn]] static void mpiDie(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Collect one per-key map from every rank onto the root and merge it.
//
// Each rank flattens its map into triples (id, count, extra). Gatherv brings
// them to root, which sums the entries key by key.
//
// Communication is O(P * distinct_keys), not O(N): for server logs the number
// of distinct servers/endpoints/intervals is far smaller than the record
// count, so this is cheap. Using per-key GATHER rather than a dense array
// reduction means no assumption is made about ID range or density.
// ---------------------------------------------------------------------------
static void gatherMerge(const std::unordered_map<long long, Pair2>& local,
                        std::unordered_map<long long, Pair2>& merged,
                        int rank, int P) {
    std::vector<long long> flat;
    flat.reserve(local.size() * 3);
    for (const auto& kv : local) {
        flat.push_back(kv.first);
        flat.push_back(kv.second.count);
        flat.push_back(kv.second.extra);
    }

    int myCount = static_cast<int>(flat.size());
    std::vector<int> counts(P), displs(P);
    MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    long long totalLL = 0;
    if (rank == 0) {
        for (int r = 0; r < P; r++) {
            displs[r] = static_cast<int>(totalLL);
            totalLL += counts[r];
        }
        if (totalLL > 2147483647LL)
            mpiDie("too many distinct keys for a 32-bit MPI count");
    }

    std::vector<long long> all;
    if (rank == 0) all.resize(static_cast<size_t>(totalLL));

    MPI_Gatherv(flat.data(), myCount, MPI_LONG_LONG,
                rank == 0 ? all.data() : nullptr,
                counts.data(), displs.data(), MPI_LONG_LONG, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        merged.clear();
        merged.reserve(all.size() / 3 + 1);
        for (size_t i = 0; i + 2 < all.size(); i += 3) {
            Pair2& p = merged[all[i]];
            p.count += all[i + 1];
            p.extra += all[i + 2];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    bool stats = false, noOutput = false, useGen = false;
    long long gN = 0, gK = 0, gS = 0;
    uint64_t seed = 12345;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--stats") == 0) stats = true;
        else if (std::strcmp(argv[i], "--no-output") == 0) noOutput = true;
        else if (std::strcmp(argv[i], "--gen") == 0) {
            if (i + 3 >= argc) mpiDie("--gen needs N K S [SEED]");
            useGen = true;
            gN = atoll(argv[i + 1]); gK = atoll(argv[i + 2]); gS = atoll(argv[i + 3]);
            i += 3;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                seed = strtoull(argv[++i], nullptr, 10);
        } else {
            mpiDie("unknown option");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    // ------------------- root obtains the log -------------------
    LogInput in;
    if (rank == 0) {
        if (useGen) {
            if (gN < 0 || gK < 0) mpiDie("N and K must be non-negative");
            in.N = gN; in.K = gK; in.S = gS;
            generateLog(in.records, gN, gS, seed);
        } else {
            readLog(stdin, in);
        }
    }

    // ------------------- broadcast the header -------------------
    long long hdr[3] = { in.N, in.K, in.S };
    MPI_Bcast(hdr, 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    const long long N = hdr[0], K = hdr[1];

    // ------------------- distribute the records -------------------
    // Records are a contiguous array of a POD struct, so MPI_BYTE moves them
    // directly. On a homogeneous cluster this is exact; MPI_Type_create_struct
    // would be the portable alternative for heterogeneous systems.
    const long long myN = recCount(rank, N, P);

    std::vector<int> counts(P), displs(P);
    long long runningBytes = 0;
    for (int r = 0; r < P; r++) {
        const long long cnt = recCount(r, N, P);
        const long long st  = recStart(r, N, P);
        const long long cb  = cnt * static_cast<long long>(sizeof(Record));
        const long long db  = st  * static_cast<long long>(sizeof(Record));
        if (cb > 2147483647LL || db > 2147483647LL)
            mpiDie("log too large for 32-bit MPI byte counts; reduce N");
        counts[r] = static_cast<int>(cb);
        displs[r] = static_cast<int>(db);
        runningBytes += cb;
    }
    (void)runningBytes;

    std::vector<Record> myRecs(static_cast<size_t>(myN > 0 ? myN : 1));
    MPI_Scatterv(rank == 0 ? in.records.data() : nullptr,
                 counts.data(), displs.data(), MPI_BYTE,
                 myRecs.data(),
                 static_cast<int>(myN * static_cast<long long>(sizeof(Record))),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) { in.records.clear(); in.records.shrink_to_fit(); }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_after_dist = MPI_Wtime();

    // ------------------- local aggregation, no communication -------------------
    Aggregate agg;
    accumulate(agg, myRecs.data(), myN);

    const double t_after_compute = MPI_Wtime();

    // ------------------- combine scalars -------------------
    // Sums, then min, then max. A rank with zero records contributes
    // minResponse = +inf and maxResponse = -inf, which are the identities for
    // MPI_MIN and MPI_MAX, so idle ranks cannot corrupt the result.
    const Scalars& L = agg.scal;
    long long sendSum[9] = { L.total, L.successful, L.failed, L.sumResponse,
                             L.totalBytes, L.s2xx, L.s3xx, L.s4xx, L.s5xx };
    long long recvSum[9] = {0};
    MPI_Reduce(sendSum, recvSum, 9, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long long gMin = 0, gMax = 0;
    MPI_Reduce(&agg.scal.minResponse, &gMin, 1, MPI_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&agg.scal.maxResponse, &gMax, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // ------------------- combine the per-key maps -------------------
    std::unordered_map<long long, Pair2> gServers, gEndpoints, gIntervals;
    gatherMerge(agg.servers,   gServers,   rank, P);
    gatherMerge(agg.endpoints, gEndpoints, rank, P);
    gatherMerge(agg.intervals, gIntervals, rank, P);

    const double t_after_reduce = MPI_Wtime();

    // ------------------- report -------------------
    if (rank == 0) {
        Scalars g;
        g.total = recvSum[0]; g.successful = recvSum[1]; g.failed = recvSum[2];
        g.sumResponse = recvSum[3]; g.totalBytes = recvSum[4];
        g.s2xx = recvSum[5]; g.s3xx = recvSum[6];
        g.s4xx = recvSum[7]; g.s5xx = recvSum[8];
        g.minResponse = gMin; g.maxResponse = gMax;

        // Top-K is selected only here, after merging: a server ranked 3rd on
        // every rank could still be 1st globally, so per-rank top-K would be
        // wrong.
        const std::vector<Entry> topServers   = topK(gServers, K);
        const std::vector<Entry> topEndpoints = topK(gEndpoints, K);
        long long busyId = 0, busyCount = 0;
        busiestInterval(gIntervals, busyId, busyCount);

        if (!noOutput)
            writeReport(stdout, g, topServers, topEndpoints, busyId, busyCount);
    }

    const double t_end = MPI_Wtime();

    // ------------------- instrumentation (stderr, --stats only) -------------------
    if (stats) {
        double loc[5] = { t_after_dist    - t_start,
                          t_after_compute - t_after_dist,
                          t_after_reduce  - t_after_compute,
                          t_end           - t_start,
                          static_cast<double>(myN) };
        double mx[5];
        MPI_Reduce(loc, mx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double mnRecs;
        MPI_Reduce(&loc[4], &mnRecs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            const double comm = mx[0] + mx[2];
            const double algo = mx[1] + comm;
            fprintf(stderr, "[mpi] P=%d N=%lld K=%lld\n", P, N, K);
            fprintf(stderr, "[mpi] records/rank = min %.0f max %.0f\n", mnRecs, mx[4]);
            fprintf(stderr, "[mpi] t_dist     = %.3f ms (Bcast hdr + Scatterv records)\n", mx[0] * 1e3);
            fprintf(stderr, "[mpi] t_compute  = %.3f ms (local aggregation)\n", mx[1] * 1e3);
            fprintf(stderr, "[mpi] t_reduce   = %.3f ms (Reduce scalars + Gatherv maps)\n", mx[2] * 1e3);
            fprintf(stderr, "[mpi] t_comm     = %.3f ms\n", comm * 1e3);
            fprintf(stderr, "[mpi] t_algo     = %.3f ms\n", algo * 1e3);
            fprintf(stderr, "[mpi] t_total    = %.3f ms\n", mx[3] * 1e3);
            fprintf(stderr, "[mpi] comm_pct   = %.1f %%\n",
                    algo > 0 ? 100.0 * comm / algo : 0.0);
            fprintf(stderr, "[csv] %d,%lld,%lld,%.6f,%.6f,%.6f,%.6f\n",
                    P, N, K, mx[0], mx[1], mx[2], mx[3]);
        }
    }

    MPI_Finalize();
    return 0;
}
