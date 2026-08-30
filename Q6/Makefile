# Q6 -- Connected Components of a Large Graph
#
#   make            build everything (programs + tools)
#   make test       build, then run the correctness suite
#   make bench      build, then run the benchmark
#   make analyze    regenerate tables and SVG plots from results.csv
#   make clean      remove all binaries
#
# On the RCE cluster, load the MPI module first:
#   module load hpcx-2.7.0/hpcx-ompi
#
# The programs and the tools (generator, analyzer) are C++. Orchestration --
# running the suites and collecting timings -- stays in shell, which is what
# shell is good at:
#   tests/run_tests.sh        correctness suite
#   benchmark/run_bench.sh    timing runs
#   slurm/job_*.sh            cluster submission

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

PROGRAMS := src/sequential src/q6_mpi
TOOLS    := tools/gen_graph tools/analyze

.PHONY: all test bench analyze clean

all: $(PROGRAMS) $(TOOLS)

src/sequential: src/sequential.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

src/q6_mpi: src/q6_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

tools/gen_graph: tools/gen_graph.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

tools/analyze: tools/analyze.cpp tools/harness.hpp
	$(CXX) $(CXXFLAGS) -o $@ tools/analyze.cpp

test: all
	bash tests/run_tests.sh

bench: all
	bash benchmark/run_bench.sh

analyze: tools/analyze
	./tools/analyze

clean:
	rm -f $(PROGRAMS) $(TOOLS)
