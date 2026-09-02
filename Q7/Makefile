# Q7 -- Large-Scale Server Log Analytics
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
# running the suites and collecting timings -- stays in shell:
#   tests/run_tests.sh        correctness suite
#   benchmark/run_bench.sh    timing runs
#   slurm/job_q7.sh           cluster submission

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

PROGRAMS := src/sequential src/q7_mpi
TOOLS    := tools/gen_log tools/analyze

.PHONY: all test bench analyze clean

all: $(PROGRAMS) $(TOOLS)

src/sequential: src/sequential.cpp src/log_io.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/sequential.cpp

src/q7_mpi: src/q7_mpi.cpp src/log_io.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ src/q7_mpi.cpp

tools/gen_log: tools/gen_log.cpp src/log_io.hpp
	$(CXX) $(CXXFLAGS) -Isrc -o $@ tools/gen_log.cpp

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
