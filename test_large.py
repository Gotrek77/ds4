#!/usr/bin/env python3
"""
Large graph transitive closure to compare Python BFS vs Lobster.
Goal: find a case where Lobster is faster.
"""
import random
import time
from collections import deque

# Parameters - large dense graph
N = 2000
EDGE_PROB = 0.02  # ~80k edges

random.seed(42)
edges = []
for i in range(N):
    for j in range(N):
        if i != j and random.random() < EDGE_PROB:
            edges.append((i, j))

print(f"Graph: {N} nodes, {len(edges)} edges")

# ---------- Python BFS (transitive closure) ----------
t0 = time.perf_counter()
adj = [[] for _ in range(N)]
for u,v in edges:
    adj[u].append(v)

total_pairs = 0
for src in range(N):
    visited = set()
    q = deque()
    q.append(src)
    visited.add(src)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    total_pairs += len(visited)
t1 = time.perf_counter()
print(f"Python BFS: {t1-t0:.6f} sec, reachable pairs: {total_pairs}")

# ---------- Lobster via scli ----------
dl_edges = "\n".join(f"rel edge({u},{v})" for u,v in edges)
dl_program = f"""type edge(i32, i32)
type path(i32, i32)

{dl_edges}

rel path(A,B) :- edge(A,B)
rel path(A,C) :- path(A,B), edge(B,C)

query path
"""
with open("/home/vulkano/work/ds4/large.scl", "w") as f:
    f.write(dl_program)

import subprocess
t0 = time.perf_counter()
proc = subprocess.run(["./scli", "large.scl"], capture_output=True, timeout=120)
t1 = time.perf_counter()
print(f"Lobster (scli): {t1-t0:.6f} sec")
out = proc.stdout.decode()
path_count = out.count("(")
print(f"Output pairs: ~{path_count}")
