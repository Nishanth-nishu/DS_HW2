// q6_mpi.cpp
// ---------------------------------------------------------------------------
// Q6: Connected Components of a Large Graph  --  MPI IMPLEMENTATION
//
// Algorithm: iterative minimum-label propagation.
//   comp[v] = v initially.
//   Each round, every rank extracts everything its OWN edges imply, using a
//   union-find with union-by-minimum-root, then a single MPI_Allreduce(MPI_MIN)
//   merges every rank's knowledge. Repeat until the label array stops changing.
//
// Distribution:
//   Vertices are split into contiguous blocks; each rank receives only the
//   adjacency lists of its own vertices (the O(E) data is partitioned).
//   The O(V) label array is replicated -- at V <= 1e5 that is 400 KB, which
//   buys a single collective in place of a ghost-exchange protocol.
//
// Correctness sketch:
//   Invariant  comp[v] is always the ID of a vertex genuinely connected to v,
//              so disconnected components can never merge.
//   Monotone   comp[v] never increases (DSU roots are minima; MPI_MIN).
//   Fixpoint   at convergence comp[u] == comp[w] for every edge, so comp is
//              constant on each component; that constant is a member and is
//              <= every member, hence it is the minimum.
//
// Build:  mpicxx -O2 -std=c++17 -o q6_mpi q6_mpi.cpp
// Run:    mpirun -np 4 ./q6_mpi < input.txt > output.txt
//         mpirun -np 4 ./q6_mpi --stats < input.txt > output.txt
//
// stdout carries ONLY the required "vertex_id component_id" lines.
// All timing/diagnostic output goes to stderr, and only with --stats.
// ---------------------------------------------------------------------------

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

// ===========================================================================
// Fast integer reader (rank 0 only)
// ===========================================================================
class Reader {
    std::vector<char> buf;
    size_t pos = 0;
public:
    void slurp(FILE* f) {
        size_t used = 0;
        buf.resize(1 << 20);
        while (true) {
            if (used == buf.size()) buf.resize(buf.size() * 2);
            size_t got = fread(buf.data() + used, 1, buf.size() - used, f);
            used += got;
            if (got == 0) break;
        }
        buf.resize(used);
    }
    bool nextInt(long long& out) {
        while (pos < buf.size() &&
               (buf[pos] == ' ' || buf[pos] == '\n' ||
                buf[pos] == '\r' || buf[pos] == '\t')) pos++;
        if (pos >= buf.size()) return false;
        bool neg = false;
        if (buf[pos] == '-') { neg = true; pos++; }
        if (pos >= buf.size() || buf[pos] < '0' || buf[pos] > '9') return false;
        long long v = 0;
        while (pos < buf.size() && buf[pos] >= '0' && buf[pos] <= '9')
            v = v * 10 + (buf[pos++] - '0');
        out = neg ? -v : v;
        return true;
    }
};

// Abort ALL ranks. Calling exit() on one rank only would hang the others
// inside their next collective.
[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);   // unreachable; tells the compiler this path never returns
}

// ===========================================================================
// STAGE 4: block partition of the vertex range
//   Integer arithmetic handles V not divisible by P, and V < P (trailing
//   ranks legitimately receive zero vertices).
// ===========================================================================
static inline int blockFirst(int r, int V, int P) {
    return static_cast<int>((static_cast<long long>(r) * V) / P);
}

// ===========================================================================
// Union-find over global vertex IDs, union-by-minimum-root.
//   The root of a set IS the smallest ID in it, so find(v) needs no extra
//   minimum pass. Path compression keeps it near-linear.
// ===========================================================================
class DSU {
    std::vector<int> parent;
public:
    explicit DSU(int n) : parent(n) {}
    void reset() { std::iota(parent.begin(), parent.end(), 0); }

    int find(int x) {
        int r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) { int nx = parent[x]; parent[x] = r; x = nx; }
        return r;
    }
    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else         parent[ra] = rb;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    bool stats = false;
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--stats") == 0) stats = true;

    double t_start = MPI_Wtime();

    // =======================================================================
    // STAGE 5: read on rank 0, distribute adjacency lists
    // =======================================================================
    int V = 0;
    std::vector<int> rootDeg;     // rank 0 only: degree of every vertex
    std::vector<int> rootFlat;    // rank 0 only: all neighbour entries, in order

    if (rank == 0) {
        Reader in;
        in.slurp(stdin);
        long long Vll;
        if (!in.nextInt(Vll)) fatal("could not read V");
        if (Vll < 0 || Vll > (1LL << 30)) fatal("V out of supported range");
        V = static_cast<int>(Vll);

        rootDeg.assign(V, 0);
        rootFlat.reserve(static_cast<size_t>(V) * 2 + 1);
        for (int i = 0; i < V; i++) {
            long long k;
            if (!in.nextInt(k)) fatal("unexpected end of input reading k");
            if (k < 0) fatal("negative neighbour count k");
            rootDeg[i] = static_cast<int>(k);
            for (long long j = 0; j < k; j++) {
                long long w;
                if (!in.nextInt(w)) fatal("unexpected end of input reading neighbours");
                if (w < 0 || w >= V) fatal("neighbour id out of range [0,V)");
                rootFlat.push_back(static_cast<int>(w));
            }
        }
        // NOTE: unlike the sequential reference, we do NOT symmetrise here.
        // The DSU unites (u,w) symmetrically by construction, so a one-sided
        // listing is already handled -- and skipping symmetrisation halves the
        // memory and the volume we have to scatter.
    }

    // Everyone needs V before allocating anything.
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (V == 0) {                    // nothing to do; still exit collectively
        MPI_Finalize();
        return 0;
    }

    const int myFirst = blockFirst(rank, V, P);
    const int myCount = blockFirst(rank + 1, V, P) - myFirst;

    // ---- scatter the degrees (counts differ per rank -> Scatterv) ----
    std::vector<int> vCounts(P), vDispls(P);
    if (rank == 0) {
        for (int r = 0; r < P; r++) {
            vDispls[r] = blockFirst(r, V, P);
            vCounts[r] = blockFirst(r + 1, V, P) - vDispls[r];
        }
    }
    std::vector<int> myDeg(myCount > 0 ? myCount : 1, 0);
    MPI_Scatterv(rank == 0 ? rootDeg.data() : nullptr,
                 vCounts.data(), vDispls.data(), MPI_INT,
                 myDeg.data(), myCount, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- scatter the neighbour entries ----
    // Byte counts per rank = sum of the degrees of that rank's vertices.
    std::vector<int> eCounts(P), eDispls(P);
    if (rank == 0) {
        long long run = 0;
        for (int r = 0; r < P; r++) {
            long long sum = 0;
            for (int v = vDispls[r]; v < vDispls[r] + vCounts[r]; v++)
                sum += rootDeg[v];
            if (run > 2147483647LL || sum > 2147483647LL)
                fatal("edge entries exceed the 32-bit MPI count limit");
            eDispls[r] = static_cast<int>(run);
            eCounts[r] = static_cast<int>(sum);
            run += sum;
        }
    }
    int myEdgeCount = 0;
    MPI_Scatter(eCounts.data(), 1, MPI_INT, &myEdgeCount, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    std::vector<int> myFlat(myEdgeCount > 0 ? myEdgeCount : 1);
    MPI_Scatterv(rank == 0 ? rootFlat.data() : nullptr,
                 eCounts.data(), eDispls.data(), MPI_INT,
                 myFlat.data(), myEdgeCount, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 can drop the full graph now: each rank owns its slice.
    if (rank == 0) {
        rootDeg.clear();  rootDeg.shrink_to_fit();
        rootFlat.clear(); rootFlat.shrink_to_fit();
    }

    // Local CSR offsets from the local degrees.
    std::vector<long long> myOff(myCount + 1, 0);
    for (int i = 0; i < myCount; i++) myOff[i + 1] = myOff[i] + myDeg[i];

    MPI_Barrier(MPI_COMM_WORLD);
    double t_after_input = MPI_Wtime();

    // =======================================================================
    // STAGES 6-8: local DSU phase, Allreduce, convergence detection
    // =======================================================================
    std::vector<int> comp(V), prev(V);
    std::iota(comp.begin(), comp.end(), 0);      // comp[v] = v
    prev = comp;

    DSU dsu(V);
    int rounds = 0;
    double t_compute = 0.0, t_comm = 0.0;

    while (true) {
        double c0 = MPI_Wtime();

        // ---- STAGE 6: local phase (no communication at all) ----
        dsu.reset();

        // Re-seed with what we already know. comp[v] = c means "v is connected
        // to c" (proved by the invariant), so this is sound, and it lets this
        // round's local edges compose transitively with knowledge that arrived
        // from other ranks in the previous Allreduce.
        for (int v = 0; v < V; v++)
            if (comp[v] != v) dsu.unite(v, comp[v]);

        // Every local edge. u is owned by this rank; w may be anywhere. We
        // write labels for remote vertices freely -- that write IS the message,
        // published to their owners by the Allreduce below.
        for (int i = 0; i < myCount; i++) {
            const int u = myFirst + i;
            for (long long e = myOff[i]; e < myOff[i + 1]; e++)
                dsu.unite(u, myFlat[e]);
        }

        // Collapse to roots. find(v) <= comp[v] always, because comp[v] was
        // united with v above, so this can only ever lower a label.
        for (int v = 0; v < V; v++) comp[v] = dsu.find(v);

        double c1 = MPI_Wtime();
        t_compute += c1 - c0;

        // ---- STAGE 7: global phase, one collective ----
        MPI_Allreduce(MPI_IN_PLACE, comp.data(), V, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);

        double c2 = MPI_Wtime();
        t_comm += c2 - c1;
        rounds++;

        // ---- STAGE 8: convergence ----
        // After the Allreduce every rank holds a bit-identical array, so this
        // comparison yields the same boolean everywhere. No extra collective
        // is needed to agree on termination.
        if (comp == prev) break;
        prev = comp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_after_compute = MPI_Wtime();

    // =======================================================================
    // STAGE 9: output
    //   Every rank already holds the complete, identical result, so there is
    //   no gather to perform. Rank 0 simply prints.
    // =======================================================================
    if (rank == 0) {
        std::vector<char> out;
        out.reserve(static_cast<size_t>(V) * 14 + 16);
        char tmp[16];
        for (int v = 0; v < V; v++) {
            int n = snprintf(tmp, sizeof(tmp), "%d", v);
            out.insert(out.end(), tmp, tmp + n);
            out.push_back(' ');
            n = snprintf(tmp, sizeof(tmp), "%d", comp[v]);
            out.insert(out.end(), tmp, tmp + n);
            out.push_back('\n');
        }
        fwrite(out.data(), 1, out.size(), stdout);
    }

    double t_end = MPI_Wtime();

    // =======================================================================
    // Instrumentation (stderr, only with --stats). A parallel phase ends when
    // its slowest rank ends, so we report the MAX across ranks.
    // =======================================================================
    if (stats) {
        double loc[4] = { t_after_input - t_start, t_compute, t_comm,
                          t_end - t_start };
        double mx[4];
        MPI_Reduce(loc, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        long long myEdges = myEdgeCount, minE = 0, maxE = 0;
        MPI_Reduce(&myEdges, &minE, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&myEdges, &maxE, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double algo = mx[1] + mx[2];
            fprintf(stderr, "[mpi] P=%d V=%d rounds=%d\n", P, V, rounds);
            fprintf(stderr, "[mpi] t_input    = %.3f ms\n", mx[0] * 1e3);
            fprintf(stderr, "[mpi] t_compute  = %.3f ms\n", mx[1] * 1e3);
            fprintf(stderr, "[mpi] t_comm     = %.3f ms\n", mx[2] * 1e3);
            fprintf(stderr, "[mpi] t_algo     = %.3f ms  (compute+comm)\n",
                    algo * 1e3);
            fprintf(stderr, "[mpi] t_total    = %.3f ms\n", mx[3] * 1e3);
            fprintf(stderr, "[mpi] comm_pct   = %.1f %%\n",
                    algo > 0 ? 100.0 * mx[2] / algo : 0.0);
            fprintf(stderr, "[mpi] edges/rank = min %lld max %lld\n", minE, maxE);
            fprintf(stderr, "[csv] %d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                    P, V, rounds, mx[0], mx[1], mx[2], mx[3]);
        }
    }

    // Silence an unused-variable warning when --stats is off.
    (void)t_after_compute;

    MPI_Finalize();
    return 0;
}
