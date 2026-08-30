// analyze.cpp
// ---------------------------------------------------------------------------
// Q1 analysis tool (C++ replacement for the Python script).
//
// Reads benchmark/results.csv and writes, on stdout, the tables the report
// format asks for:
//   * runtime            median t_algo, seconds
//   * speed-up           S(P) = T1 / TP
//   * efficiency         E(P) = S(P) / P
//   * communication      t_comm as a percentage of t_algo
//   * phase breakdown    t_dist / t_compute / t_gather
//
// And writes four SVG charts into benchmark/plots/. SVG is generated directly
// (see harness.hpp) so no plotting library is needed; it embeds in a report as
// is, and converts to PNG with any standard tool if a raster is required.
//
// The value reported per configuration is the MEDIAN over trials, which is
// robust against a single slow run caused by another job sharing the node.
//
// T1 is the MPI binary at P=1, not the sequential program, so the speed-up
// measures parallel scaling of one implementation rather than comparing two
// different programs.
//
// Build: g++ -O2 -std=c++17 -o analyze analyze.cpp
// Run  : ./tools/analyze [results.csv]      (from the Q1 project root)
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
    std::string size;
    int         P;
    bool operator<(const Key& o) const {
        return size != o.size ? size < o.size : P < o.P;
    }
};

struct Agg {
    double dist = 0, compute = 0, gather = 0, total = 0, comm = 0, algo = 0;
};

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "benchmark/results.csv";

    auto rows = harness::readCsv(path);
    if (rows.empty()) {
        std::fprintf(stderr,
                     "no usable results at %s\nRun ./tools/run_bench first "
                     "(from the Q1 project root).\n", path.c_str());
        return 1;
    }

    // ---- gather raw samples ------------------------------------------------
    std::map<Key, std::vector<double>> vDist, vComp, vGath, vTot;
    std::vector<std::string> sizes;
    std::map<std::string, std::string> shape;
    std::vector<int> procs;

    for (auto& r : rows) {
        Key k{r["size"], std::atoi(r["P"].c_str())};
        vDist[k].push_back(std::atof(r["t_dist"].c_str()));
        vComp[k].push_back(std::atof(r["t_compute"].c_str()));
        vGath[k].push_back(std::atof(r["t_gather"].c_str()));
        vTot[k].push_back(std::atof(r["t_total"].c_str()));
        if (std::find(sizes.begin(), sizes.end(), k.size) == sizes.end()) {
            sizes.push_back(k.size);
            shape[k.size] = r["m"] + "x" + r["n"] + "x" + r["p"];
        }
        if (std::find(procs.begin(), procs.end(), k.P) == procs.end())
            procs.push_back(k.P);
    }
    std::sort(procs.begin(), procs.end());

    // ---- medians -----------------------------------------------------------
    std::map<Key, Agg> agg;
    for (auto& kv : vComp) {
        const Key& k = kv.first;
        Agg a;
        a.dist    = median(vDist[k]);
        a.compute = median(vComp[k]);
        a.gather  = median(vGath[k]);
        a.total   = median(vTot[k]);
        a.comm    = a.dist + a.gather;
        a.algo    = a.compute + a.comm;
        agg[k] = a;
    }
    auto get = [&](const std::string& s, int p, Agg& out) {
        auto it = agg.find(Key{s, p});
        if (it == agg.end()) return false;
        out = it->second;
        return true;
    };

    // ---- header ------------------------------------------------------------
    std::printf("# Q1 Benchmark Analysis - Row-Row Matrix Multiplication\n");
    std::printf("\nSource: `%s`  |  process counts:", path.c_str());
    for (int p : procs) std::printf(" %d", p);
    std::printf("  |  statistic: median over trials\n");
    std::printf("\n`t_algo` = `t_compute` + `t_comm`, where `t_comm` = "
                "`t_dist` (Bcast dims + Scatterv A + Bcast B) + `t_gather` "
                "(Gatherv C).\n");

    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes) body.push_back({s, shape[s]});
        harness::printTable("Matrix sizes", {"Label", "m x n x p"}, body);
    }

    std::vector<std::string> head{"Input size"};
    for (int p : procs) head.push_back("P=" + std::to_string(p));

    // ---- runtime -----------------------------------------------------------
    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes) {
            std::vector<std::string> row{s};
            for (int p : procs) {
                Agg a;
                row.push_back(get(s, p, a) ? fmt(a.algo, 4) : "--");
            }
            body.push_back(row);
        }
        harness::printTable("Runtime - median `t_algo` (seconds)", head, body);
    }

    // ---- speed-up ----------------------------------------------------------
    std::map<Key, double> speed;
    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes) {
            std::vector<std::string> row{s};
            Agg base;
            const bool haveBase = get(s, 1, base);
            for (int p : procs) {
                Agg a;
                if (haveBase && get(s, p, a) && a.algo > 0) {
                    const double v = base.algo / a.algo;
                    speed[Key{s, p}] = v;
                    row.push_back(fmt(v, 2));
                } else {
                    row.push_back("--");
                }
            }
            body.push_back(row);
        }
        harness::printTable("Speed-up  S(P) = T1 / TP", head, body);
    }

    // ---- efficiency --------------------------------------------------------
    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes) {
            std::vector<std::string> row{s};
            for (int p : procs) {
                auto it = speed.find(Key{s, p});
                row.push_back(it == speed.end()
                                  ? "--"
                                  : fmt(100.0 * it->second / p, 1) + "%");
            }
            body.push_back(row);
        }
        harness::printTable("Efficiency  E(P) = S(P) / P", head, body);
    }

    // ---- communication share ----------------------------------------------
    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes) {
            std::vector<std::string> row{s};
            for (int p : procs) {
                Agg a;
                row.push_back((get(s, p, a) && a.algo > 0)
                                  ? fmt(100.0 * a.comm / a.algo, 1) + "%"
                                  : "--");
            }
            body.push_back(row);
        }
        harness::printTable(
            "Communication share - `t_comm` as % of `t_algo`", head, body,
            "Rising with P is expected: the Bcast of B moves n*p elements to "
            "every process regardless of P, while the computation per process "
            "falls as 1/P.");
    }

    // ---- phase breakdown ---------------------------------------------------
    {
        std::vector<std::vector<std::string>> body;
        for (auto& s : sizes)
            for (int p : procs) {
                Agg a;
                if (!get(s, p, a)) continue;
                body.push_back({s, std::to_string(p), fmt(a.dist, 4),
                                fmt(a.compute, 4), fmt(a.gather, 4),
                                fmt(a.algo, 4)});
            }
        harness::printTable(
            "Phase breakdown (seconds, median)",
            {"Size", "P", "t_dist", "t_compute", "t_gather", "t_algo"}, body);
    }

    // ---- plots -------------------------------------------------------------
    std::system("mkdir -p benchmark/plots");

    auto build = [&](double (*pick)(const Agg&, double), bool useSpeed) {
        std::vector<Series> out;
        for (auto& s : sizes) {
            Series ser;
            ser.name = s + " (" + shape[s] + ")";
            for (int p : procs) {
                if (useSpeed) {
                    auto it = speed.find(Key{s, p});
                    if (it == speed.end()) continue;
                    ser.x.push_back(p);
                    ser.y.push_back(pick(Agg{}, it->second));
                } else {
                    Agg a;
                    if (!get(s, p, a)) continue;
                    ser.x.push_back(p);
                    ser.y.push_back(pick(a, 0.0));
                }
            }
            if (!ser.x.empty()) out.push_back(ser);
        }
        return out;
    };

    harness::svgPlot("benchmark/plots/time.svg",
                     "Q1 - Execution time vs processes",
                     "Number of MPI processes (P)", "t_algo (s, log scale)",
                     build([](const Agg& a, double) { return a.algo; }, false),
                     true, false);

    harness::svgPlot("benchmark/plots/speedup.svg",
                     "Q1 - Speed-up vs processes",
                     "Number of MPI processes (P)", "Speed-up  S(P)",
                     build([](const Agg&, double s) { return s; }, true),
                     false, true);

    {
        std::vector<Series> eff;
        for (auto& s : sizes) {
            Series ser;
            ser.name = s + " (" + shape[s] + ")";
            for (int p : procs) {
                auto it = speed.find(Key{s, p});
                if (it == speed.end()) continue;
                ser.x.push_back(p);
                ser.y.push_back(it->second / p);
            }
            if (!ser.x.empty()) eff.push_back(ser);
        }
        harness::svgPlot("benchmark/plots/efficiency.svg",
                         "Q1 - Efficiency vs processes",
                         "Number of MPI processes (P)", "Efficiency  E(P)",
                         eff, false, false);
    }

    harness::svgPlot(
        "benchmark/plots/comm_fraction.svg",
        "Q1 - Communication share of algorithm time",
        "Number of MPI processes (P)", "t_comm / t_algo",
        build([](const Agg& a, double) {
            return a.algo > 0 ? a.comm / a.algo : 0.0;
        }, false),
        false, false);

    std::printf("\n### Plots written\n\n");
    std::printf("- `benchmark/plots/time.svg`\n");
    std::printf("- `benchmark/plots/speedup.svg`\n");
    std::printf("- `benchmark/plots/efficiency.svg`\n");
    std::printf("- `benchmark/plots/comm_fraction.svg`\n");
    return 0;
}
