// gen_log.cpp
// ---------------------------------------------------------------------------
// Writes a Q7 log file in the PDF's input format:
//
//     N K S
//     timestamp server_id endpoint_id user_id status_code response_time bytes_sent
//     ... N such lines
//
// The programs can generate a log internally with --gen, which is what the
// benchmark uses. This tool exists for the cases where an actual FILE is
// wanted: building test fixtures, inspecting input by hand, or feeding the
// programs through stdin.
//
// It reuses generateLog() from src/log_io.hpp, so a file produced here with
// seed S is identical to what --gen produces with the same seed.
//
// Build: g++ -O2 -std=c++17 -Isrc -o gen_log gen_log.cpp
// Run  : ./tools/gen_log N K S [SEED] > log.txt
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "log_io.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s N K S [SEED] > log.txt\n\n"
            "  N     number of log records\n"
            "  K     top-K parameter carried in the header\n"
            "  S     server count (the PDF does not define S; the generator\n"
            "        uses it as the number of distinct servers)\n"
            "  SEED  optional, default 12345; the same seed reproduces the\n"
            "        same file exactly\n", argv[0]);
        return 2;
    }
    const long long N = std::atoll(argv[1]);
    const long long K = std::atoll(argv[2]);
    const long long S = std::atoll(argv[3]);
    const uint64_t seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 12345;

    if (N < 0 || K < 0) { std::fprintf(stderr, "ERROR: N and K must be non-negative\n"); return 1; }

    std::vector<Record> recs;
    generateLog(recs, N, S, seed);

    std::string out;
    out.reserve(static_cast<size_t>(N) * 48 + 32);
    char tmp[128];
    int len = std::snprintf(tmp, sizeof(tmp), "%lld %lld %lld\n", N, K, S);
    out.append(tmp, len);
    for (long long i = 0; i < N; i++) {
        const Record& r = recs[static_cast<size_t>(i)];
        len = std::snprintf(tmp, sizeof(tmp), "%lld %d %d %d %d %d %lld\n",
                            r.timestamp, r.server_id, r.endpoint_id, r.user_id,
                            r.status_code, r.response_time, r.bytes_sent);
        out.append(tmp, len);
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
