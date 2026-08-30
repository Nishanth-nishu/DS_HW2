// gen_matrix.cpp
// ---------------------------------------------------------------------------
// Writes a Q1 problem file in our input format:
//
//     m n p
//     <m*n integers>      matrix A, row-major
//     <n*p integers>      matrix B, row-major
//
// The MPI and sequential programs can generate matrices internally with
// --gen, which is what the benchmark uses. This tool exists for the cases
// where an actual FILE is wanted: building test fixtures, inspecting an input
// by hand, or feeding the programs through stdin as the PDF's examples do.
//
// It reuses generateMatrix() from src/matrix_io.hpp, so a file produced here
// with seed S is identical to what --gen produces with the same seed.
//
// Build: g++ -O2 -std=c++17 -Isrc -o gen_matrix gen_matrix.cpp
// Run  : ./tools/gen_matrix M N P [SEED] > input.txt
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "matrix_io.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s M N P [SEED] > input.txt\n\n"
                     "Writes an m x n matrix A and an n x p matrix B in the "
                     "Q1 input format.\nEntries are in [-9, 9]; the same SEED "
                     "reproduces the same file exactly.\n", argv[0]);
        return 2;
    }
    const long long m = std::atoll(argv[1]);
    const long long n = std::atoll(argv[2]);
    const long long p = std::atoll(argv[3]);
    const uint64_t seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 12345;

    if (m < 0 || n < 0 || p < 0) {
        std::fprintf(stderr, "ERROR: dimensions must be non-negative\n");
        return 1;
    }

    std::vector<Elem> A, B;
    generateMatrix(A, m * n, seed);
    generateMatrix(B, n * p, seed ^ 0xA5A5A5A5A5A5A5A5ULL);

    std::string out;
    out.reserve(static_cast<size_t>((m * n + n * p) * 4 + 32));
    char tmp[24];

    int len = std::snprintf(tmp, sizeof(tmp), "%lld %lld %lld\n", m, n, p);
    out.append(tmp, len);

    auto emit = [&](const std::vector<Elem>& M, long long rows, long long cols) {
        for (long long i = 0; i < rows; i++) {
            for (long long j = 0; j < cols; j++) {
                len = std::snprintf(tmp, sizeof(tmp), "%d",
                                    M[static_cast<size_t>(i * cols + j)]);
                out.append(tmp, len);
                out.push_back(j + 1 < cols ? ' ' : '\n');
            }
            if (cols == 0) out.push_back('\n');
        }
    };
    emit(A, m, n);
    emit(B, n, p);

    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
