# Q6 -- Connected Components of a Large Graph
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

SRC := $(dir $(lastword $(MAKEFILE_LIST)))
BIN := sequential q6_mpi

.PHONY: all test clean

all: $(BIN)

sequential: sequential.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q6_mpi: q6_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

test: all
	bash ../tests/run_tests.sh

clean:
	rm -f $(BIN)
