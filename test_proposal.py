#!/usr/bin/env python3
"""
Proposed problem: probabilistic connectivity on a large graph.
100 nodes, 200 probabilistic edges -> 2^200 worlds impossible to enumerate.
Lobster with minmaxprob computes exact probability in milliseconds.
"""
import random
import time

N = 100
NUM_EDGES = 200
random.seed(42)

edges = []
for _ in range(NUM_EDGES):
    u = random.randrange(N)
    v = random.randrange(N)
    while v == u:
        v = random.randrange(N)
    p = random.random()
    edges.append((u, v, p))

print(f"Graph: {N} nodes, {len(edges)} probabilistic edges")
print(f"Worlds: 2^{len(edges)} = 2^{NUM_EDGES} (impossible to enumerate)")

# Lobster with minmaxprob
dl_facts = "\n".join(f"rel {p}::edge({u},{v})" for u,v,p in edges)
dl_program = f"""type edge(i32, i32)
type path(i32, i32)

{dl_facts}

rel path(A,B) :- edge(A,B)
rel path(A,C) :- path(A,B), edge(B,C)

query path(0,{N-1})
"""
with open("/home/vulkano/work/ds4/prop.scl", "w") as f:
    f.write(dl_program)

import subprocess
t0 = time.perf_counter()
proc = subprocess.run(["./scli", "prop.scl", "-p", "minmaxprob"], capture_output=True, timeout=30)
t1 = time.perf_counter()
print(f"Lobster (scli minmaxprob): {t1-t0:.6f} sec")
out = proc.stdout.decode().strip()
print(f"Output: {out}")
