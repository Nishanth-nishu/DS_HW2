// matrix_io.hpp
// ---------------------------------------------------------------------------
// Shared helpers for Q1 (Row-Row distributed matrix multiplication).
//
// Both sequential.cpp and q1_mpi.cpp include this file, so that:
//   * the input parser is identical, and
//   * generated matrices are BIT-IDENTICAL for a given (m,n,p,seed).
// If the two programs generated matrices differently, comparing their outputs
// would prove nothing.
// ---------------------------------------------------------------------------

#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Storage decision: matrices are flat, ROW-MAJOR, 1-D arrays.
//     A is m x n   ->  A[i][j] == A[i*n + j]
//     B is n x p   ->  B[k][j] == B[k*p + j]
//     C is m x p   ->  C[i][j] == C[i*p + j]
// A block of consecutive ROWS is therefore a contiguous block of memory,
// which is exactly what MPI_Scatterv/MPI_Gatherv need -- no derived datatypes.
//
// Element types:
//   A, B  : int      (the PDF states entries are integers)
//   C     : long long
// The PDF does not bound the magnitude of the entries. With n = 1000 and
// unbounded ints the accumulator can overflow 32 bits, so C accumulates and is
// stored in 64 bits. This is an IMPLEMENTATION CHOICE, documented in README.
// ---------------------------------------------------------------------------
using Elem = int;
using Acc  = long long;

// ===========================================================================
// Fast whitespace-separated integer reader
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
        else if (buf[pos] == '+') pos++;
        if (pos >= buf.size() || buf[pos] < '0' || buf[pos] > '9') return false;
        long long v = 0;
        while (pos < buf.size() && buf[pos] >= '0' && buf[pos] <= '9')
            v = v * 10 + (buf[pos++] - '0');
        out = neg ? -v : v;
        return true;
    }
};

// ===========================================================================
// Deterministic generator (splitmix64). Used so that large benchmark inputs
// need not be written to disk: the same (m,n,p,seed) reproduces the same
// matrices in every program, on every machine.
// Entries fall in [-9, 9], matching the small magnitudes in the PDF examples
// and guaranteeing no overflow: |C| <= 9*9*n = 81n.
// ===========================================================================
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline void generateMatrix(std::vector<Elem>& out, long long count,
                           uint64_t seed) {
    out.resize(static_cast<size_t>(count));
    uint64_t st = seed;
    for (long long i = 0; i < count; i++)
        out[static_cast<size_t>(i)] =
            static_cast<Elem>(static_cast<int>(splitmix64(st) % 19) - 9);
}

// ===========================================================================
// INPUT FORMAT  (implementation choice -- the PDF prescribes none for Q1)
//
//     m n p
//     <m*n integers>      matrix A, row-major
//     <n*p integers>      matrix B, row-major
//
// Whitespace (spaces or newlines) between values is irrelevant.
//
// Because the three dimensions are given in ONE header, A's column count and
// B's row count are the same value n by construction -- an inconsistent pair
// cannot be expressed. We still validate that exactly m*n and n*p values
// follow, so truncated or overlong input is rejected rather than misread.
// ===========================================================================
struct Problem {
    long long m = 0, n = 0, p = 0;
    std::vector<Elem> A, B;
};

inline void die(const std::string& msg) {
    fprintf(stderr, "ERROR: %s\n", msg.c_str());
    exit(1);
}

// Reads a problem from `f`. Returns false only on a well-formed empty stream.
inline void readProblem(FILE* f, Problem& prob) {
    Reader in;
    in.slurp(f);
    long long m, n, p;
    if (!in.nextInt(m) || !in.nextInt(n) || !in.nextInt(p))
        die("could not read the header line \"m n p\"");
    if (m < 0 || n < 0 || p < 0) die("dimensions must be non-negative");
    prob.m = m; prob.n = n; prob.p = p;

    prob.A.resize(static_cast<size_t>(m * n));
    for (long long i = 0; i < m * n; i++) {
        long long v;
        if (!in.nextInt(v))
            die("matrix A is truncated: expected " + std::to_string(m * n) +
                " values, got " + std::to_string(i));
        prob.A[static_cast<size_t>(i)] = static_cast<Elem>(v);
    }
    prob.B.resize(static_cast<size_t>(n * p));
    for (long long i = 0; i < n * p; i++) {
        long long v;
        if (!in.nextInt(v))
            die("matrix B is truncated: expected " + std::to_string(n * p) +
                " values, got " + std::to_string(i));
        prob.B[static_cast<size_t>(i)] = static_cast<Elem>(v);
    }
    long long extra;
    if (in.nextInt(extra))
        die("input contains more values than the header m n p describes");
}

// ===========================================================================
// ROW-ROW multiply kernel.
//
// Computes rows [0, rows) of C from `Asub` (rows x n) and B (n x p) as the
// PDF defines the method:
//     c_i = a_i[0]*B[0,:] + a_i[1]*B[1,:] + ... + a_i[n-1]*B[n-1,:]
//
// The loop order is i-k-j, NOT i-j-k:
//   * the k loop walks the terms of the weighted sum,
//   * the j loop performs "scale a whole row of B and add it",
//   * A[i][k] is loop-invariant in j, so it is loaded once into a register,
//   * B[k][j] and C[i][j] are both walked sequentially in memory.
// This is both the method the PDF names and the cache-friendly order.
// ===========================================================================
inline void rowRowMultiply(const Elem* Asub, const Elem* B, Acc* Csub,
                           long long rows, long long n, long long p) {
    for (long long i = 0; i < rows * p; i++) Csub[i] = 0;

    for (long long i = 0; i < rows; i++) {
        const Elem* arow = Asub + i * n;
        Acc*        crow = Csub + i * p;
        for (long long k = 0; k < n; k++) {
            const Elem a = arow[k];              // the weight
            if (a == 0) continue;                // skip a no-op row addition
            const Elem* brow = B + k * p;        // the whole row of B
            for (long long j = 0; j < p; j++)
                crow[j] += static_cast<Acc>(a) * brow[j];
        }
    }
}

// ===========================================================================
// OUTPUT FORMAT (implementation choice): m lines, each of p space-separated
// integers -- the natural rendering of matrix C, matching how the PDF displays
// its example results.
// ===========================================================================
inline void writeMatrix(FILE* f, const Acc* C, long long m, long long p) {
    std::vector<char> out;
    out.reserve(static_cast<size_t>(m) * p * 4 + 16);
    char tmp[24];
    for (long long i = 0; i < m; i++) {
        for (long long j = 0; j < p; j++) {
            int len = snprintf(tmp, sizeof(tmp), "%lld", C[i * p + j]);
            out.insert(out.end(), tmp, tmp + len);
            out.push_back(j + 1 < p ? ' ' : '\n');
        }
        if (p == 0) out.push_back('\n');
    }
    fwrite(out.data(), 1, out.size(), f);
}

#endif  // MATRIX_IO_HPP
