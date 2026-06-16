#!/usr/bin/env python3
"""
Probabilistic graph: compute probability that node 0 reaches node 9.
Lobster with minmaxprob provenance vs Python exact enumeration (2^E worlds).
"""
import random
import time
import itertools
import math

# Small graph: 10 nodes, 20 edges with random probabilities
N = 10
NUM_EDGES = 20
random.seed(42)

nodes = list(range(N))
edges = []
for _ in range(NUM_EDGES):
    u = random.randrange(N)
    v = random.randrange(N)
    while v == u:
        v = random.randrange(N)
    p = random.random()
    edges.append((u, v, p))

print(f"Graph: {N} nodes, {len(edges)} probabilistic edges")

# ---------- Python exact enumeration (2^E worlds) ----------
# For each world, compute reachability from 0 to 9, sum probabilities.
t0 = time.perf_counter()
total_prob = 0.0
# There are 2^20 = 1,048,576 worlds - feasible but slow in Python
num_worlds = 0
for bits in range(1 << len(edges)):
    # Compute probability of this world
    prob = 1.0
    adj = [[] for _ in range(N)]
    for i, (u,v,p) in enumerate(edges):
        if (bits >> i) & 1:
            prob *= p
            adj[u].append(v)
        else:
            prob *= (1 - p)
    if prob == 0:
        continue
    # Check reachability from 0 to 9
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    if 9 in visited:
        total_prob += prob
    num_worlds += 1
t1 = time.perf_counter()
print(f"Python exact enumeration: {t1-t0:.6f} sec, worlds: {num_worlds}")
print(f"Probability 0→9: {total_prob:.6f}")

# ---------- Lobster with minmaxprob ----------
# Write Datalog program with probabilistic facts
dl_facts = "\n".join(f"rel {p}::edge({u},{v})" for u,v,p in edges)
dl_program = f"""type edge(i32, i32)
type path(i32, i32)

{dl_facts}

rel path(A,B) :- edge(A,B)
rel path(A,C) :- path(A,B), edge(B,C)

query path(0,9)
"""
with open("/home/vulkano/work/ds4/prob.scl", "w") as f:
    f.write(dl_program)

import subprocess
t0 = time.perf_counter()
proc = subprocess.run(["./scli", "prob.scl", "-p", "minmaxprob"], capture_output=True, timeout=30)
t1 = time.perf_counter()
print(f"Lobster (scli minmaxprob): {t1-t0:.6f} sec")
out = proc.stdout.decode()
print(f"Output: {out.strip()}")
