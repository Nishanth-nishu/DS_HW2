// sequential.cpp
// ---------------------------------------------------------------------------
// Q1: Distributed Matrix Multiplication, Row-Row Method
//     -- SEQUENTIAL REFERENCE IMPLEMENTATION
//
// Computes C = A x B on a single process, using the Row-Row formulation:
//     c_i = a_i[0]*B[0,:] + a_i[1]*B[1,:] + ... + a_i[n-1]*B[n-1,:]
//
// This program is the correctness oracle for the MPI version. It is also the
// basis for the "absolute speed-up" figure, though the reported speed-up
// S(P) = T1/TP uses the MPI binary at P=1 (see README section 8).
//
// Complexity: O(m*n*p) multiply-adds, O(m*n + n*p + m*p) memory.
//
// Build: g++ -O2 -std=c++17 -o sequential sequential.cpp
//
// Run (read a problem from stdin):
//     ./sequential < input.txt > output.txt
//
// Run (generate the problem in-process, for large benchmarks):
//     ./sequential --gen M N P [SEED] > output.txt
//
// Options:
//     --stats      timing to stderr (never to stdout)
//     --no-output  suppress the matrix C; for timing runs where writing
//                  millions of numbers would dominate the measurement
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstring>
#include <vector>

#include "matrix_io.hpp"

int main(int argc, char** argv) {
    bool stats = false, noOutput = false, useGen = false;
    long long gm = 0, gn = 0, gp = 0;
    uint64_t seed = 12345;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--stats") == 0) {
            stats = true;
        } else if (std::strcmp(argv[i], "--no-output") == 0) {
            noOutput = true;
        } else if (std::strcmp(argv[i], "--gen") == 0) {
            if (i + 3 >= argc) die("--gen needs M N P [SEED]");
            useGen = true;
            gm = atoll(argv[i + 1]);
            gn = atoll(argv[i + 2]);
            gp = atoll(argv[i + 3]);
            i += 3;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                seed = strtoull(argv[++i], nullptr, 10);
        } else {
            die(std::string("unknown option: ") + argv[i]);
        }
    }

    auto clk = []() { return std::chrono::steady_clock::now(); };
    auto ms  = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    auto t0 = clk();

    // ---------------- obtain the problem ----------------
    Problem prob;
    if (useGen) {
        if (gm < 0 || gn < 0 || gp < 0) die("dimensions must be non-negative");
        prob.m = gm; prob.n = gn; prob.p = gp;
        // Distinct seeds for A and B so the two matrices are not identical
        // when they happen to have the same shape.
        generateMatrix(prob.A, gm * gn, seed);
        generateMatrix(prob.B, gn * gp, seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    } else {
        readProblem(stdin, prob);
    }

    // Dimension validity: C = A(m x n) * B(n x p) requires A's column count to
    // equal B's row count. Our input format states both as the single value n,
    // so a mismatch is unrepresentable; this check documents the requirement
    // and guards against a future format change.
    if (static_cast<long long>(prob.A.size()) != prob.m * prob.n)
        die("matrix A does not contain m*n elements");
    if (static_cast<long long>(prob.B.size()) != prob.n * prob.p)
        die("matrix B does not contain n*p elements");

    auto t1 = clk();

    // ---------------- compute ----------------
    std::vector<Acc> C(static_cast<size_t>(prob.m * prob.p));
    rowRowMultiply(prob.A.data(), prob.B.data(), C.data(),
                   prob.m, prob.n, prob.p);

    auto t2 = clk();

    // ---------------- output ----------------
    if (!noOutput) writeMatrix(stdout, C.data(), prob.m, prob.p);

    auto t3 = clk();

    if (stats) {
        fprintf(stderr, "[seq] m=%lld n=%lld p=%lld\n", prob.m, prob.n, prob.p);
        fprintf(stderr, "[seq] t_input   = %.3f ms\n", ms(t0, t1));
        fprintf(stderr, "[seq] t_compute = %.3f ms\n", ms(t1, t2));
        fprintf(stderr, "[seq] t_output  = %.3f ms\n", ms(t2, t3));
        fprintf(stderr, "[seq] t_total   = %.3f ms\n", ms(t0, t3));
    }
    return 0;
}
