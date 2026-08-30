// sequential.cpp
// ---------------------------------------------------------------------------
// Q6: Connected Components of a Large Graph  --  SEQUENTIAL REFERENCE
//
// Purpose: correctness oracle for the MPI implementation. Not parallel.
//
// Algorithm: BFS from every not-yet-visited vertex, scanning seeds in
// increasing vertex order. Because seeds are scanned in increasing order,
// the seed of a component is automatically the minimum vertex ID in it.
//
// Complexity: O(V + E) time, O(V + E) memory.
//
// Build:  g++ -O2 -std=c++17 -o sequential sequential.cpp
// Run:    ./sequential < input.txt > output.txt
//         ./sequential --stats < input.txt > output.txt   (timings -> stderr)
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// ---------------------------------------------------------------------------
// Fast input: slurp all of stdin, then parse integers by hand.
// At the assignment limits the file holds ~2e6 integers; iostream/scanf per
// token is measurably slower and would pollute the timing comparison.
// ---------------------------------------------------------------------------
class Reader {
    std::vector<char> buf;
    size_t pos = 0;
public:
    void slurp(FILE* f) {
        const size_t CHUNK = 1 << 20;
        size_t used = 0;
        buf.resize(CHUNK);
        while (true) {
            if (used == buf.size()) buf.resize(buf.size() * 2);
            size_t got = fread(buf.data() + used, 1, buf.size() - used, f);
            used += got;
            if (got == 0) break;
        }
        buf.resize(used);
    }
    // Returns false at end of input. Accepts an optional leading '-' so that
    // malformed negative values are *parsed* and can then be range-checked,
    // rather than silently mis-read.
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

static void fail(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

int main(int argc, char** argv) {
    bool stats = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--stats") == 0) stats = true;

    auto t0 = std::chrono::steady_clock::now();

    // ---------------- read + build CSR ----------------
    Reader in;
    in.slurp(stdin);

    long long Vll;
    if (!in.nextInt(Vll)) fail("could not read V");
    if (Vll < 0) fail("V is negative");
    const int V = static_cast<int>(Vll);

    // Pass 1: read the file exactly as given into (src, dst) pair form.
    std::vector<int> esrc, edst;
    for (int i = 0; i < V; i++) {
        long long k;
        if (!in.nextInt(k)) fail("unexpected end of input while reading degrees");
        if (k < 0) fail("negative neighbour count k");
        for (long long j = 0; j < k; j++) {
            long long w;
            if (!in.nextInt(w)) fail("unexpected end of input while reading neighbours");
            if (w < 0 || w >= V) fail("neighbour id out of range [0,V)");
            esrc.push_back(i);
            edst.push_back(static_cast<int>(w));
        }
    }

    // Pass 2: build a SYMMETRIC CSR. The problem states the graph is
    // undirected, but the PDF does not guarantee that the adjacency lists are
    // listed symmetrically. We therefore insert every listed pair in BOTH
    // directions. Without this, a one-sided listing "u lists w but w does not
    // list u" would be traversable only one way and BFS could split a genuine
    // component. This also keeps the reference in agreement with the MPI
    // version, which unions each listed pair regardless of direction.
    // Duplicates and self-loops are left in place: both are harmless here.
    const size_t M = esrc.size();
    std::vector<long long> off(V + 1, 0);
    for (size_t e = 0; e < M; e++) { off[esrc[e] + 1]++; off[edst[e] + 1]++; }
    for (int i = 0; i < V; i++) off[i + 1] += off[i];

    std::vector<int> flat(off[V]);
    {
        std::vector<long long> cur(off.begin(), off.end() - 1);
        for (size_t e = 0; e < M; e++) {
            flat[cur[esrc[e]]++] = edst[e];
            flat[cur[edst[e]]++] = esrc[e];
        }
    }
    esrc.clear(); esrc.shrink_to_fit();
    edst.clear(); edst.shrink_to_fit();

    auto t1 = std::chrono::steady_clock::now();

    // ---------------- BFS over components ----------------
    const int UNSET = -1;
    std::vector<int> comp(V, UNSET);
    std::vector<int> queue;
    queue.reserve(V);

    for (int s = 0; s < V; s++) {
        if (comp[s] != UNSET) continue;   // already claimed by a smaller seed
        comp[s] = s;                      // s is the minimum of its component
        queue.clear();
        queue.push_back(s);
        for (size_t qi = 0; qi < queue.size(); qi++) {
            int u = queue[qi];
            for (long long e = off[u]; e < off[u + 1]; e++) {
                int w = flat[e];
                if (comp[w] == UNSET) {
                    comp[w] = s;
                    queue.push_back(w);
                }
            }
        }
    }

    auto t2 = std::chrono::steady_clock::now();

    // ---------------- output ----------------
    // Manual formatting into one buffer: V up to 1e5 lines, and we want the
    // oracle to be fast enough that it is never the bottleneck in testing.
    {
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

    auto t3 = std::chrono::steady_clock::now();

    if (stats) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr, "[seq] V=%d E_entries=%zu\n", V, flat.size());
        fprintf(stderr, "[seq] t_input   = %.3f ms\n", ms(t0, t1));
        fprintf(stderr, "[seq] t_compute = %.3f ms\n", ms(t1, t2));
        fprintf(stderr, "[seq] t_output  = %.3f ms\n", ms(t2, t3));
        fprintf(stderr, "[seq] t_total   = %.3f ms\n", ms(t0, t3));
    }
    return 0;
}
