// sequential.cpp
// ---------------------------------------------------------------------------
// Q7: Large-Scale Server Log Analytics -- SEQUENTIAL REFERENCE
//
// Single-process implementation. Serves as the correctness oracle for the MPI
// version and as the baseline for the absolute-speed-up figure.
//
// Complexity: O(N) to accumulate, plus O(D log D) to sort the D distinct
// server/endpoint IDs for the top-K selection. Memory O(N + D).
//
// Build: g++ -O2 -std=c++17 -o sequential sequential.cpp
// Run  : ./sequential < input.txt > output.txt
//        ./sequential --gen N K S [SEED] > output.txt
// Options: --stats (timings to stderr), --no-output (suppress the report)
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstring>

#include "log_io.hpp"

int main(int argc, char** argv) {
    bool stats = false, noOutput = false, useGen = false;
    long long gN = 0, gK = 0, gS = 0;
    uint64_t seed = 12345;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--stats") == 0) stats = true;
        else if (std::strcmp(argv[i], "--no-output") == 0) noOutput = true;
        else if (std::strcmp(argv[i], "--gen") == 0) {
            if (i + 3 >= argc) logDie("--gen needs N K S [SEED]");
            useGen = true;
            gN = atoll(argv[i + 1]); gK = atoll(argv[i + 2]); gS = atoll(argv[i + 3]);
            i += 3;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                seed = strtoull(argv[++i], nullptr, 10);
        } else {
            logDie(std::string("unknown option: ") + argv[i]);
        }
    }

    auto clk = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    auto t0 = clk();

    LogInput in;
    if (useGen) {
        if (gN < 0 || gK < 0) logDie("N and K must be non-negative");
        in.N = gN; in.K = gK; in.S = gS;
        generateLog(in.records, gN, gS, seed);
    } else {
        readLog(stdin, in);
    }

    auto t1 = clk();

    Aggregate agg;
    accumulate(agg, in.records.data(), static_cast<long long>(in.records.size()));

    const std::vector<Entry> topServers   = topK(agg.servers, in.K);
    const std::vector<Entry> topEndpoints = topK(agg.endpoints, in.K);
    long long busyId = 0, busyCount = 0;
    busiestInterval(agg.intervals, busyId, busyCount);

    auto t2 = clk();

    if (!noOutput)
        writeReport(stdout, agg.scal, topServers, topEndpoints, busyId, busyCount);

    auto t3 = clk();

    if (stats) {
        fprintf(stderr, "[seq] N=%lld K=%lld S=%lld\n", in.N, in.K, in.S);
        fprintf(stderr, "[seq] distinct servers=%zu endpoints=%zu intervals=%zu\n",
                agg.servers.size(), agg.endpoints.size(), agg.intervals.size());
        fprintf(stderr, "[seq] t_input   = %.3f ms\n", ms(t0, t1));
        fprintf(stderr, "[seq] t_compute = %.3f ms\n", ms(t1, t2));
        fprintf(stderr, "[seq] t_output  = %.3f ms\n", ms(t2, t3));
        fprintf(stderr, "[seq] t_total   = %.3f ms\n", ms(t0, t3));
    }
    return 0;
}
