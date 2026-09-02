// harness.hpp
// ---------------------------------------------------------------------------
// Shared utilities for the C++ analysis tool.
//
// Provides what would otherwise need a Python/plotting stack:
//   * median / statistics
//   * reading CSV and emitting Markdown tables
//   * drawing line charts as SVG, with no external plotting library
//
// Header-only, so analyze.cpp stays a single translation unit.
//
// (Test and benchmark ORCHESTRATION stays in shell -- tests/run_tests.sh and
// benchmark/run_bench.sh -- since launching processes and comparing files is
// what a shell does well. This header covers the parts that are not.)
// ---------------------------------------------------------------------------

#ifndef HARNESS_HPP
#define HARNESS_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace harness {

// ===========================================================================
// Files
// ===========================================================================
inline std::string slurpFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline bool writeFile(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << data;
    return true;
}

inline bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// ===========================================================================
// Statistics
// ===========================================================================
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ===========================================================================
// Strings / CSV
// ===========================================================================
inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, sep)) out.push_back(cur);
    return out;
}

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) b--;
    return s.substr(a, b - a);
}

inline std::string fmt(double v, int prec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// Reads a CSV with a header row into a vector of column->value maps.
inline std::vector<std::map<std::string, std::string>>
readCsv(const std::string& path) {
    std::vector<std::map<std::string, std::string>> rows;
    std::ifstream f(path);
    if (!f) return rows;
    std::string line;
    if (!std::getline(f, line)) return rows;
    const std::vector<std::string> header = split(trim(line), ',');
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const std::vector<std::string> cells = split(line, ',');
        if (cells.size() != header.size()) continue;
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < header.size(); i++) row[header[i]] = cells[i];
        rows.push_back(row);
    }
    return rows;
}

// ===========================================================================
// Markdown table emission
// ===========================================================================
inline void printTable(const std::string& title,
                       const std::vector<std::string>& header,
                       const std::vector<std::vector<std::string>>& body,
                       const std::string& note = "") {
    std::printf("\n### %s\n\n", title.c_str());
    if (!note.empty()) std::printf("%s\n\n", note.c_str());
    std::printf("|");
    for (const auto& h : header) std::printf(" %s |", h.c_str());
    std::printf("\n|:--|");
    for (size_t i = 1; i < header.size(); i++) std::printf(":-:|");
    std::printf("\n");
    for (const auto& row : body) {
        std::printf("|");
        for (const auto& c : row) std::printf(" %s |", c.c_str());
        std::printf("\n");
    }
}

// ===========================================================================
// SVG line chart
//
// Written by hand so the tools need no plotting library. SVG is a text format,
// renders in any browser, embeds directly in a Markdown/HTML report, and
// converts to PNG with any standard converter if a raster image is required.
// ===========================================================================
struct Series {
    std::string         name;
    std::vector<double> x, y;
};

inline void svgPlot(const std::string& path, const std::string& title,
                    const std::string& xLabel, const std::string& yLabel,
                    const std::vector<Series>& series, bool logY = false,
                    bool idealLine = false) {
    const double W = 760, H = 460;
    const double L = 78, R = 210, T = 52, Bm = 62;   // margins
    const double pw = W - L - R, ph = H - T - Bm;

    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (const auto& s : series)
        for (size_t i = 0; i < s.x.size(); i++) {
            xmin = std::min(xmin, s.x[i]);
            xmax = std::max(xmax, s.x[i]);
            if (logY && s.y[i] <= 0) continue;
            ymin = std::min(ymin, s.y[i]);
            ymax = std::max(ymax, s.y[i]);
        }
    if (idealLine) { ymax = std::max(ymax, xmax); ymin = std::min(ymin, xmin); }
    if (series.empty() || xmin > xmax) { xmin = 0; xmax = 1; ymin = 0; ymax = 1; }
    if (logY) {
        if (ymin <= 0) ymin = 1e-6;
        ymin = std::log10(ymin); ymax = std::log10(ymax);
    } else if (ymin > 0) {
        ymin = 0;                       // linear charts start at zero
    }
    if (std::fabs(ymax - ymin) < 1e-12) ymax = ymin + 1;
    const double pad = 0.06 * (ymax - ymin);
    ymax += pad;
    if (!logY && ymin < 0) ymin -= pad;

    auto sx = [&](double v) {
        return L + (xmax > xmin ? (v - xmin) / (xmax - xmin) : 0.5) * pw;
    };
    auto sy = [&](double v) {
        const double t = logY ? std::log10(std::max(v, 1e-300)) : v;
        return T + ph - (t - ymin) / (ymax - ymin) * ph;
    };

    const char* palette[] = {"#2f6fb5", "#c1492e", "#3f8f56", "#8a5fb0",
                             "#c98a1e", "#3a9ba8", "#a3486f", "#6b7a3a"};
    const int nPal = 8;

    std::ostringstream o;
    o << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W
      << "\" height=\"" << H << "\" viewBox=\"0 0 " << W << " " << H
      << "\" font-family=\"Helvetica,Arial,sans-serif\">\n";
    o << "<rect width=\"" << W << "\" height=\"" << H << "\" fill=\"#ffffff\"/>\n";
    o << "<text x=\"" << W / 2 << "\" y=\"28\" text-anchor=\"middle\" "
         "font-size=\"17\" font-weight=\"600\">" << title << "</text>\n";

    // horizontal grid + y ticks
    const int nTicks = 5;
    for (int i = 0; i <= nTicks; i++) {
        const double frac = static_cast<double>(i) / nTicks;
        const double yv = ymin + frac * (ymax - ymin);
        const double py = T + ph - frac * ph;
        o << "<line x1=\"" << L << "\" y1=\"" << py << "\" x2=\"" << (L + pw)
          << "\" y2=\"" << py << "\" stroke=\"#e2e2e2\" stroke-width=\"1\"/>\n";
        const double shown = logY ? std::pow(10.0, yv) : yv;
        std::ostringstream lab;
        if (logY || (std::fabs(shown) < 10 && shown != 0))
            lab << std::scientific << std::setprecision(1) << shown;
        else
            lab << std::fixed << std::setprecision(2) << shown;
        o << "<text x=\"" << (L - 9) << "\" y=\"" << (py + 4)
          << "\" text-anchor=\"end\" font-size=\"11\" fill=\"#444\">"
          << lab.str() << "</text>\n";
    }

    // x ticks at the actual sampled x values
    std::vector<double> xs;
    for (const auto& s : series)
        for (double v : s.x)
            if (std::find(xs.begin(), xs.end(), v) == xs.end()) xs.push_back(v);
    std::sort(xs.begin(), xs.end());
    for (double v : xs) {
        const double px = sx(v);
        o << "<line x1=\"" << px << "\" y1=\"" << T << "\" x2=\"" << px
          << "\" y2=\"" << (T + ph)
          << "\" stroke=\"#f0f0f0\" stroke-width=\"1\"/>\n";
        o << "<text x=\"" << px << "\" y=\"" << (T + ph + 20)
          << "\" text-anchor=\"middle\" font-size=\"12\" fill=\"#444\">"
          << static_cast<long long>(v) << "</text>\n";
    }

    // axes
    o << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\""
      << (T + ph) << "\" stroke=\"#333\" stroke-width=\"1.4\"/>\n";
    o << "<line x1=\"" << L << "\" y1=\"" << (T + ph) << "\" x2=\"" << (L + pw)
      << "\" y2=\"" << (T + ph) << "\" stroke=\"#333\" stroke-width=\"1.4\"/>\n";
    o << "<text x=\"" << (L + pw / 2) << "\" y=\"" << (H - 16)
      << "\" text-anchor=\"middle\" font-size=\"13\">" << xLabel << "</text>\n";
    o << "<text x=\"18\" y=\"" << (T + ph / 2)
      << "\" text-anchor=\"middle\" font-size=\"13\" transform=\"rotate(-90 18 "
      << (T + ph / 2) << ")\">" << yLabel << "</text>\n";

    // ideal (linear) reference
    if (idealLine && !xs.empty()) {
        o << "<polyline fill=\"none\" stroke=\"#999\" stroke-width=\"1.5\" "
             "stroke-dasharray=\"6,4\" points=\"";
        for (double v : xs) o << sx(v) << "," << sy(v) << " ";
        o << "\"/>\n";
    }

    // data
    for (size_t si = 0; si < series.size(); si++) {
        const auto& s = series[si];
        const char* col = palette[si % nPal];
        o << "<polyline fill=\"none\" stroke=\"" << col
          << "\" stroke-width=\"2.2\" points=\"";
        for (size_t i = 0; i < s.x.size(); i++)
            o << sx(s.x[i]) << "," << sy(s.y[i]) << " ";
        o << "\"/>\n";
        for (size_t i = 0; i < s.x.size(); i++)
            o << "<circle cx=\"" << sx(s.x[i]) << "\" cy=\"" << sy(s.y[i])
              << "\" r=\"3.4\" fill=\"" << col << "\"/>\n";
    }

    // legend
    double ly = T + 6;
    for (size_t si = 0; si < series.size(); si++) {
        const char* col = palette[si % nPal];
        o << "<line x1=\"" << (L + pw + 16) << "\" y1=\"" << ly << "\" x2=\""
          << (L + pw + 40) << "\" y2=\"" << ly << "\" stroke=\"" << col
          << "\" stroke-width=\"2.6\"/>\n";
        o << "<text x=\"" << (L + pw + 46) << "\" y=\"" << (ly + 4)
          << "\" font-size=\"11\" fill=\"#222\">" << series[si].name
          << "</text>\n";
        ly += 19;
    }
    if (idealLine) {
        o << "<line x1=\"" << (L + pw + 16) << "\" y1=\"" << ly << "\" x2=\""
          << (L + pw + 40) << "\" y2=\"" << ly
          << "\" stroke=\"#999\" stroke-width=\"2\" stroke-dasharray=\"6,4\"/>\n";
        o << "<text x=\"" << (L + pw + 46) << "\" y=\"" << (ly + 4)
          << "\" font-size=\"11\" fill=\"#222\">ideal (linear)</text>\n";
    }

    o << "</svg>\n";
    writeFile(path, o.str());
}

}  // namespace harness

#endif  // HARNESS_HPP
