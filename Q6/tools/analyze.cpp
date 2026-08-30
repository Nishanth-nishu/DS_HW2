// analyze.cpp
// ---------------------------------------------------------------------------
// Q6 analysis tool (C++ replacement for the Python script).
//
// Reads benchmark/results.csv and writes, on stdout, the tables the report
// format asks for: runtime, speed-up, efficiency, communication share, and
// convergence round counts. Writes three SVG charts into benchmark/plots/.
//
// SVG is generated directly (see harness.hpp) so no plotting library is
// needed; it embeds in a report as is, and converts to PNG if a raster is
// required.
//
// The value reported per configuration is the MEDIAN over trials. T1 is the
// MPI binary at P=1, not the sequential program, so the speed-up measures
// parallel scaling of one implementation.
//
// Build: g++ -O2 -std=c++17 -o analyze analyze.cpp
// Run  : ./tools/analyze [results.csv]     (from the Q6 project root)
// ---------------------------------------------------------------------------

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "harness.hpp"

using harness::fmt;
using harness::median;
using harness::Series;

struct Key {
    std::string ds;
    int         P;
    bool operator<(const Key& o) const {
        return ds != o.ds ? ds < o.ds : P < o.P;
    }
};

struct Agg { double input = 0, compute = 0, comm = 0, total = 0, algo = 0, rounds = 0; };

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "benchmark/results.csv";

    auto rows = harness::readCsv(path);
    if (rows.empty()) {
        std::fprintf(stderr,
                     "no usable results at %s\nRun ./tools/run_bench first "
                     "(from the Q6 project root).\n", path.c_str());
        return 1;
    }

    std::map<Key, std::vector<double>> vIn, vComp, vComm, vTot, vRounds;
    std::vector<std::string> dsets;
    std::vector<int> procs;

    for (auto& r : rows) {
        Key k{r["dataset"], std::atoi(r["P"].c_str())};
        vIn[k].push_back(std::atof(r["t_input"].c_str()));
        vComp[k].push_back(std::atof(r["t_compute"].c_str()));
        vComm[k].push_back(std::atof(r["t_comm"].c_str()));
        vTot[k].push_back(std::atof(r["t_total"].c_str()));
        vRounds[k].push_back(std::atof(r["rounds"].c_str()));
        if (std::find(dsets.begin(), dsets.end(), k.ds) == dsets.end())
            dsets.push_back(k.ds);
        if (std::find(procs.begin(), procs.end(), k.P) == procs.end())
            procs.push_back(k.P);
    }
    std::sort(procs.begin(), procs.end());

    std::map<Key, Agg> agg;
    for (auto& kv : vComp) {
        const Key& k = kv.first;
        Agg a;
        a.input   = median(vIn[k]);
        a.compute = median(vComp[k]);
        a.comm    = median(vComm[k]);
        a.total   = median(vTot[k]);
        a.rounds  = median(vRounds[k]);
        a.algo    = a.compute + a.comm;
        agg[k] = a;
    }
    auto get = [&](const std::string& d, int p, Agg& out) {
        auto it = agg.find(Key{d, p});
        if (it == agg.end()) return false;
        out = it->second;
        return true;
    };

    std::printf("# Q6 Benchmark Analysis - Connected Components\n");
    std::printf("\nSource: `%s`  |  process counts:", path.c_str());
    for (int p : procs) std::printf(" %d", p);
    std::printf("  |  statistic: median over trials\n");
    std::printf("\n`t_algo` = `t_compute` + `t_comm` (excludes file reading and "
                "distribution).\n");

    std::vector<std::string> head{"Input size"};
    for (int p : procs) head.push_back("P=" + std::to_string(p));

    {
        std::vector<std::vector<std::string>> body;
        for (auto& d : dsets) {
            std::vector<std::string> row{d};
            for (int p : procs) { Agg a; row.push_back(get(d, p, a) ? fmt(a.algo, 4) : "--"); }
            body.push_back(row);
        }
        harness::printTable("Runtime - median `t_algo` (seconds)", head, body);
    }

    std::map<Key, double> speed;
    {
        std::vector<std::vector<std::string>> body;
        for (auto& d : dsets) {
            std::vector<std::string> row{d};
            Agg base; const bool haveBase = get(d, 1, base);
            for (int p : procs) {
                Agg a;
                if (haveBase && get(d, p, a) && a.algo > 0) {
                    const double v = base.algo / a.algo;
                    speed[Key{d, p}] = v;
                    row.push_back(fmt(v, 2));
                } else row.push_back("--");
            }
            body.push_back(row);
        }
        harness::printTable("Speed-up  S(P) = T1 / TP", head, body);
    }

    {
        std::vector<std::vector<std::string>> body;
        for (auto& d : dsets) {
            std::vector<std::string> row{d};
            for (int p : procs) {
                auto it = speed.find(Key{d, p});
                row.push_back(it == speed.end() ? "--" : fmt(100.0 * it->second / p, 1) + "%");
            }
            body.push_back(row);
        }
        harness::printTable("Efficiency  E(P) = S(P) / P", head, body);
    }

    {
        std::vector<std::vector<std::string>> body;
        for (auto& d : dsets) {
            std::vector<std::string> row{d};
            for (int p : procs) {
                Agg a;
                row.push_back((get(d, p, a) && a.algo > 0)
                                  ? fmt(100.0 * a.comm / a.algo, 1) + "%" : "--");
            }
            body.push_back(row);
        }
        harness::printTable("Communication share - `t_comm` as % of `t_algo`",
                            head, body,
                            "The Allreduce moves V ints per round regardless of "
                            "P, while the per-rank edge work falls as 1/P.");
    }

    {
        std::vector<std::vector<std::string>> body;
        for (auto& d : dsets) {
            std::vector<std::string> row{d};
            for (int p : procs) {
                Agg a;
                row.push_back(get(d, p, a) ? fmt(a.rounds, 0) : "--");
            }
            body.push_back(row);
        }
        harness::printTable("Convergence rounds", head, body,
                            "Deterministic and machine-independent: these "
                            "depend on the graph and the partition, not on "
                            "timing.");
    }

    std::system("mkdir -p benchmark/plots");

    auto seriesOf = [&](bool useSpeed, int mode) {
        // mode 0 = algo time, 1 = comm fraction, 2 = efficiency
        std::vector<Series> out;
        for (auto& d : dsets) {
            Series s; s.name = d;
            for (int p : procs) {
                if (useSpeed) {
                    auto it = speed.find(Key{d, p});
                    if (it == speed.end()) continue;
                    s.x.push_back(p);
                    s.y.push_back(mode == 2 ? it->second / p : it->second);
                } else {
                    Agg a;
                    if (!get(d, p, a)) continue;
                    s.x.push_back(p);
                    s.y.push_back(mode == 1 ? (a.algo > 0 ? a.comm / a.algo : 0.0)
                                            : a.algo);
                }
            }
            if (!s.x.empty()) out.push_back(s);
        }
        return out;
    };

    harness::svgPlot("benchmark/plots/time.svg", "Q6 - Execution time vs processes",
                     "Number of MPI processes (P)", "t_algo (s, log scale)",
                     seriesOf(false, 0), true, false);
    harness::svgPlot("benchmark/plots/speedup.svg", "Q6 - Speed-up vs processes",
                     "Number of MPI processes (P)", "Speed-up  S(P)",
                     seriesOf(true, 0), false, true);
    harness::svgPlot("benchmark/plots/efficiency.svg", "Q6 - Efficiency vs processes",
                     "Number of MPI processes (P)", "Efficiency  E(P)",
                     seriesOf(true, 2), false, false);
    harness::svgPlot("benchmark/plots/comm_fraction.svg",
                     "Q6 - Communication share of algorithm time",
                     "Number of MPI processes (P)", "t_comm / t_algo",
                     seriesOf(false, 1), false, false);

    std::printf("\n### Plots written\n\n");
    std::printf("- `benchmark/plots/time.svg`\n");
    std::printf("- `benchmark/plots/speedup.svg`\n");
    std::printf("- `benchmark/plots/efficiency.svg`\n");
    std::printf("- `benchmark/plots/comm_fraction.svg`\n");
    return 0;
}
