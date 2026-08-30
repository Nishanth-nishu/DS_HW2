# Q1 -- Row-Row Distributed Matrix Multiplication
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

PROGRAMS := src/sequential src/q1_mpi
TOOLS    := tools/analyze tools/gen_matrix

.PHONY: all test bench analyze clean

all: $(PROGRAMS) $(TOOLS)

src/sequential: src/sequential.cpp src/matrix_io.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/sequential.cpp

src/q1_mpi: src/q1_mpi.cpp src/matrix_io.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ src/q1_mpi.cpp

tools/gen_matrix: tools/gen_matrix.cpp src/matrix_io.hpp
	$(CXX) $(CXXFLAGS) -Isrc -o $@ tools/gen_matrix.cpp

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
