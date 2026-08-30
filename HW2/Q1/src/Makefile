# Q1 -- Row-Row Distributed Matrix Multiplication
#
#   make            build both binaries
#   make test       build, then run the correctness suite
#   make clean      remove binaries
#
# On the RCE cluster, load the MPI module first:
#   module load hpcx-2.7.0/hpcx-ompi

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

BIN := sequential q1_mpi

.PHONY: all test clean

all: $(BIN)

sequential: sequential.cpp matrix_io.hpp
	$(CXX) $(CXXFLAGS) -o $@ sequential.cpp

q1_mpi: q1_mpi.cpp matrix_io.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ q1_mpi.cpp

test: all
	bash ../tests/run_tests.sh

clean:
	rm -f $(BIN)
