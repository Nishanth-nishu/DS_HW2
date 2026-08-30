#!/usr/bin/env python3
"""
gen_graph.py -- reproducible graph generator for Q6 (connected components).

Emits the Q6 input format on stdout:
    V
    k v1 v2 ... vk      (one line per vertex, i-th line = neighbours of vertex i)

Every topology is driven by a fixed --seed, so any dataset can be regenerated
byte-for-byte from the command line recorded in the README.

Topologies
----------
random        Erdos-Renyi style: E random pairs. With E >> V this is a single
              giant component; with E < V/2 it yields many small components.
path          0-1-2-...-(V-1). One component, maximum graph diameter. This is
              the adversarial case for iterative label propagation: it forces
              the largest number of convergence rounds.
star          One hub connected to all others. One component, diameter 2, but
              severely load-imbalanced (one vertex holds V-1 edges).
disconnected  V/2 independent 2-vertex components. Maximum component count.
isolated      No edges at all (E = 0).
cluster       C equal-sized cliques-ish blobs, no inter-blob edges. Gives a
              known, checkable component count with non-trivial internal work.
chain-blocks  Dense blobs joined in a line by single bridge edges. Designed so
              components straddle MPI block boundaries: stresses cross-process
              propagation specifically.

Symmetry
--------
By default each edge is written in BOTH adjacency lists (symmetric, like the
PDF sample). --asymmetric writes each edge in only one list, to test that both
implementations still treat it as undirected.
"""

import argparse
import random
import sys


def emit(V, adj, out):
    w = out.write
    w(f"{V}\n")
    for i in range(V):
        a = adj[i]
        if a:
            w(f"{len(a)} " + " ".join(map(str, a)) + "\n")
        else:
            w("0\n")


def add(adj, u, v, symmetric):
    adj[u].append(v)
    if symmetric:
        adj[v].append(u)


def build(args):
    V = args.vertices
    sym = not args.asymmetric
    rng = random.Random(args.seed)
    adj = [[] for _ in range(V)]
    t = args.topology

    if t == "random":
        E = args.edges
        for _ in range(E):
            u = rng.randrange(V)
            v = rng.randrange(V)
            if u != v:                       # skip self-loops
                add(adj, u, v, sym)

    elif t == "path":
        for i in range(V - 1):
            add(adj, i, i + 1, sym)

    elif t == "star":
        hub = args.hub if args.hub is not None else 0
        for i in range(V):
            if i != hub:
                add(adj, hub, i, sym)

    elif t == "disconnected":
        for i in range(0, V - 1, 2):
            add(adj, i, i + 1, sym)

    elif t == "isolated":
        pass

    elif t == "cluster":
        C = max(1, args.clusters)
        size = V // C
        per = max(1, args.edges // C) if args.edges else size * 2
        for c in range(C):
            lo = c * size
            hi = V if c == C - 1 else (c + 1) * size
            if hi - lo < 2:
                continue
            # spanning path guarantees the blob is one component
            for i in range(lo, hi - 1):
                add(adj, i, i + 1, sym)
            # extra random chords inside the blob only
            for _ in range(per):
                u = rng.randrange(lo, hi)
                v = rng.randrange(lo, hi)
                if u != v:
                    add(adj, u, v, sym)

    elif t == "chain-blocks":
        # Dense blocks connected in a line by a single bridge each. The bridges
        # are what must travel across MPI process boundaries.
        B = max(1, args.clusters)
        size = V // B
        for b in range(B):
            lo = b * size
            hi = V if b == B - 1 else (b + 1) * size
            for i in range(lo, hi - 1):
                add(adj, i, i + 1, sym)
            for _ in range(max(0, (hi - lo) // 2)):
                u = rng.randrange(lo, hi)
                v = rng.randrange(lo, hi)
                if u != v:
                    add(adj, u, v, sym)
            if hi < V:                       # bridge to the next block
                add(adj, hi - 1, hi, sym)

    else:
        sys.exit(f"unknown topology: {t}")

    return V, adj


def main():
    p = argparse.ArgumentParser(description="Reproducible Q6 graph generator")
    p.add_argument("-v", "--vertices", type=int, required=True)
    p.add_argument("-e", "--edges", type=int, default=0,
                   help="edge count for 'random'/'cluster' topologies")
    p.add_argument("-t", "--topology", default="random",
                   choices=["random", "path", "star", "disconnected",
                            "isolated", "cluster", "chain-blocks"])
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-c", "--clusters", type=int, default=8)
    p.add_argument("--hub", type=int, default=None)
    p.add_argument("--asymmetric", action="store_true",
                   help="write each edge in only one adjacency list")
    p.add_argument("-o", "--output", default="-")
    args = p.parse_args()

    if args.vertices < 0:
        sys.exit("vertices must be >= 0")

    V, adj = build(args)
    if args.output == "-":
        emit(V, adj, sys.stdout)
    else:
        with open(args.output, "w") as f:
            emit(V, adj, f)


if __name__ == "__main__":
    main()
