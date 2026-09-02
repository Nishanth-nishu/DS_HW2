// log_io.hpp
// ---------------------------------------------------------------------------
// Shared definitions for Q7 (Large-Scale Server Log Analytics).
//
// Included by BOTH sequential.cpp and q7_mpi.cpp so that parsing, generation,
// aggregation semantics and output formatting are identical. If the two
// programs formatted numbers differently, comparing their outputs would prove
// nothing.
// ---------------------------------------------------------------------------

#ifndef LOG_IO_HPP
#define LOG_IO_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Record layout.
//
// A plain POD struct in a contiguous vector, so a block of consecutive records
// is a contiguous block of memory and can be moved with MPI_Scatterv using
// MPI_BYTE -- no derived datatype needed.
//
// FIELD TYPES ARE AN IMPLEMENTATION CHOICE: the PDF does not state them. All
// seven fields are treated as integers, consistent with the rest of the
// assignment and with typical log data (response_time in milliseconds).
// bytes_sent and timestamp are 64-bit because they accumulate/exceed 2^31.
// ===========================================================================
struct Record {
    long long timestamp;
    int       server_id;
    int       endpoint_id;
    int       user_id;
    int       status_code;
    int       response_time;
    long long bytes_sent;
};

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

inline void logDie(const std::string& msg) {
    fprintf(stderr, "ERROR: %s\n", msg.c_str());
    exit(1);
}

// ===========================================================================
// INPUT FORMAT (from the PDF)
//
//     N K S
//     timestamp server_id endpoint_id user_id status_code response_time bytes_sent
//     ... N such lines
//
// NOTE ON S: the PDF never defines what S means. It is most likely the number
// of servers. This implementation PARSES it but does not depend on it --
// aggregation uses the IDs actually present in the data -- so the result is
// correct regardless of the intended meaning.
// ===========================================================================
struct LogInput {
    long long           N = 0, K = 0, S = 0;
    std::vector<Record> records;
};

inline void readLog(FILE* f, LogInput& in) {
    Reader rd;
    rd.slurp(f);
    long long N, K, S;
    if (!rd.nextInt(N) || !rd.nextInt(K) || !rd.nextInt(S))
        logDie("could not read the header line \"N K S\"");
    if (N < 0) logDie("N is negative");
    if (K < 0) logDie("K is negative");
    in.N = N; in.K = K; in.S = S;

    in.records.resize(static_cast<size_t>(N));
    for (long long i = 0; i < N; i++) {
        long long ts, sid, eid, uid, st, rt, by;
        if (!rd.nextInt(ts) || !rd.nextInt(sid) || !rd.nextInt(eid) ||
            !rd.nextInt(uid) || !rd.nextInt(st) || !rd.nextInt(rt) ||
            !rd.nextInt(by))
            logDie("log truncated: expected " + std::to_string(N) +
                   " records, record " + std::to_string(i) + " is incomplete");
        Record& r = in.records[static_cast<size_t>(i)];
        r.timestamp     = ts;
        r.server_id     = static_cast<int>(sid);
        r.endpoint_id   = static_cast<int>(eid);
        r.user_id       = static_cast<int>(uid);
        r.status_code   = static_cast<int>(st);
        r.response_time = static_cast<int>(rt);
        r.bytes_sent    = by;
    }
    long long extra;
    if (rd.nextInt(extra))
        logDie("input contains more values than the header N K S describes");
}

// ===========================================================================
// Deterministic generator (splitmix64).
//
// Used so large benchmark inputs need not be written to disk: the same
// (N, K, S, seed) reproduces the same log in every program on every platform.
//
// Derived parameters (implementation choices, documented in README):
//   servers   in [0, S)
//   endpoints in [0, 4S)
//   users     in [0, 100S)
//   timestamps spread over a 24-hour window from a fixed epoch base
//   status codes drawn ~ 80% 2xx, 8% 3xx, 8% 4xx, 4% 5xx
//   response_time in [1, 1000]   bytes_sent in [100, 100099]
// ===========================================================================
inline uint64_t logSplitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline void generateLog(std::vector<Record>& out, long long N, long long S,
                        uint64_t seed) {
    const long long servers   = S > 0 ? S : 1;
    const long long endpoints = servers * 4;
    const long long users     = servers * 100;
    const long long base      = 1700000000LL;   // fixed epoch base
    const long long window    = 86400LL;        // 24 hours

    out.resize(static_cast<size_t>(N));
    uint64_t st = seed;
    for (long long i = 0; i < N; i++) {
        Record& r = out[static_cast<size_t>(i)];
        r.timestamp   = base + static_cast<long long>(logSplitmix64(st) % (uint64_t)window);
        r.server_id   = static_cast<int>(logSplitmix64(st) % (uint64_t)servers);
        r.endpoint_id = static_cast<int>(logSplitmix64(st) % (uint64_t)endpoints);
        r.user_id     = static_cast<int>(logSplitmix64(st) % (uint64_t)users);

        const uint64_t bucket = logSplitmix64(st) % 100;
        if (bucket < 80)      r.status_code = 200 + static_cast<int>(logSplitmix64(st) % 4);
        else if (bucket < 88) r.status_code = 301 + static_cast<int>(logSplitmix64(st) % 3);
        else if (bucket < 96) r.status_code = 400 + static_cast<int>(logSplitmix64(st) % 5);
        else                  r.status_code = 500 + static_cast<int>(logSplitmix64(st) % 4);

        r.response_time = 1 + static_cast<int>(logSplitmix64(st) % 1000);
        r.bytes_sent    = 100 + static_cast<long long>(logSplitmix64(st) % 100000);
    }
}

// ===========================================================================
// Aggregation
//
// Scalars plus three maps. The maps are keyed by the IDs actually observed, so
// no assumption is made about ID density or range.
//
// A request is SUCCESSFUL if status_code < 400 (stated in the PDF).
// Interval ID is timestamp / 60 (stated in the PDF).
// ===========================================================================
struct Scalars {
    long long total = 0, successful = 0, failed = 0;
    long long sumResponse = 0;
    long long minResponse = std::numeric_limits<long long>::max();
    long long maxResponse = std::numeric_limits<long long>::min();
    long long totalBytes = 0;
    long long s2xx = 0, s3xx = 0, s4xx = 0, s5xx = 0;
};

struct Pair2 { long long count = 0, extra = 0; };   // extra = sum(resp) or sum(bytes)

struct Aggregate {
    Scalars scal;
    std::unordered_map<long long, Pair2> servers;    // id -> (count, sum response)
    std::unordered_map<long long, Pair2> endpoints;  // id -> (count, total bytes)
    std::unordered_map<long long, Pair2> intervals;  // id -> (count, unused)
};

inline void accumulate(Aggregate& a, const Record* recs, long long n) {
    for (long long i = 0; i < n; i++) {
        const Record& r = recs[i];
        Scalars& s = a.scal;
        s.total++;
        if (r.status_code < 400) s.successful++; else s.failed++;
        s.sumResponse += r.response_time;
        if (r.response_time < s.minResponse) s.minResponse = r.response_time;
        if (r.response_time > s.maxResponse) s.maxResponse = r.response_time;
        s.totalBytes += r.bytes_sent;
        if (r.status_code >= 200 && r.status_code < 300) s.s2xx++;
        else if (r.status_code >= 300 && r.status_code < 400) s.s3xx++;
        else if (r.status_code >= 400 && r.status_code < 500) s.s4xx++;
        else if (r.status_code >= 500 && r.status_code < 600) s.s5xx++;

        Pair2& sv = a.servers[r.server_id];
        sv.count++; sv.extra += r.response_time;

        Pair2& ep = a.endpoints[r.endpoint_id];
        ep.count++; ep.extra += r.bytes_sent;

        a.intervals[r.timestamp / 60].count++;
    }
}

// ===========================================================================
// Top-K selection
//
// The PDF: "sorted by decreasing count, then increasing ID".
//
// IMPLEMENTATION CHOICE: if fewer than K distinct IDs exist, all of them are
// printed (the PDF does not say). K = 0 prints none.
// ===========================================================================
struct Entry { long long id, count, extra; };

inline std::vector<Entry> topK(const std::unordered_map<long long, Pair2>& m,
                               long long K) {
    std::vector<Entry> v;
    v.reserve(m.size());
    for (const auto& kv : m)
        v.push_back({kv.first, kv.second.count, kv.second.extra});
    std::sort(v.begin(), v.end(), [](const Entry& a, const Entry& b) {
        if (a.count != b.count) return a.count > b.count;   // decreasing count
        return a.id < b.id;                                 // then increasing ID
    });
    if (K >= 0 && static_cast<long long>(v.size()) > K)
        v.resize(static_cast<size_t>(K));
    return v;
}

// Busiest 60-second interval.
// IMPLEMENTATION CHOICE: the PDF specifies tie-breaking for top-K but not for
// the busiest interval; we break ties by SMALLEST interval ID, consistent with
// the "then increasing ID" rule used everywhere else.
inline void busiestInterval(const std::unordered_map<long long, Pair2>& m,
                            long long& outId, long long& outCount) {
    outId = 0; outCount = 0;
    bool first = true;
    for (const auto& kv : m) {
        const long long id = kv.first, c = kv.second.count;
        if (first || c > outCount || (c == outCount && id < outId)) {
            outId = id; outCount = c; first = false;
        }
    }
}

// ===========================================================================
// OUTPUT
//
// Exactly the format given in the PDF. Number formatting is an implementation
// choice (the PDF does not specify it):
//   AVERAGE_RESPONSE_TIME and the per-server average -> 6 decimal places
//   all counts, byte totals, min/max response time    -> integers
//
// IMPLEMENTATION CHOICE for N = 0: averages print as 0.000000 and min/max as
// 0, since no value exists. The PDF does not define this case.
// ===========================================================================
inline void writeReport(FILE* f, const Scalars& s,
                        const std::vector<Entry>& servers,
                        const std::vector<Entry>& endpoints,
                        long long busyId, long long busyCount) {
    const double avg = s.total > 0
                           ? static_cast<double>(s.sumResponse) / static_cast<double>(s.total)
                           : 0.0;
    const long long mn = s.total > 0 ? s.minResponse : 0;
    const long long mx = s.total > 0 ? s.maxResponse : 0;

    fprintf(f, "TOTAL_REQUESTS %lld\n", s.total);
    fprintf(f, "SUCCESSFUL_REQUESTS %lld\n", s.successful);
    fprintf(f, "FAILED_REQUESTS %lld\n", s.failed);
    fprintf(f, "AVERAGE_RESPONSE_TIME %.6f\n", avg);
    fprintf(f, "MIN_RESPONSE_TIME %lld\n", mn);
    fprintf(f, "MAX_RESPONSE_TIME %lld\n", mx);
    fprintf(f, "TOTAL_BYTES %lld\n", s.totalBytes);
    fprintf(f, "STATUS_2XX %lld\n", s.s2xx);
    fprintf(f, "STATUS_3XX %lld\n", s.s3xx);
    fprintf(f, "STATUS_4XX %lld\n", s.s4xx);
    fprintf(f, "STATUS_5XX %lld\n", s.s5xx);
    fprintf(f, "BUSIEST_INTERVAL %lld %lld\n", busyId, busyCount);

    fprintf(f, "TOP_SERVERS\n");
    for (const Entry& e : servers) {
        const double a = e.count > 0
                             ? static_cast<double>(e.extra) / static_cast<double>(e.count)
                             : 0.0;
        fprintf(f, "%lld %lld %.6f\n", e.id, e.count, a);
    }
    fprintf(f, "TOP_ENDPOINTS\n");
    for (const Entry& e : endpoints)
        fprintf(f, "%lld %lld %lld\n", e.id, e.count, e.extra);
}

#endif  // LOG_IO_HPP
