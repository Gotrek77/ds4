#!/usr/bin/env python3
"""
Water jug problem: 3 bottles of capacities 12, 7, 5.
Initial: (12, 0, 0). Goal: get 1 liter in any bottle.
Solve with BFS, measure time.
"""
import time
from collections import deque

CAPS = (12, 7, 5)
START = (12, 0, 0)
# Goal: any bottle contains exactly 1 liter
def goal(state):
    return 1 in state

def possible_moves(state):
    moves = []
    for i in range(3):
        for j in range(3):
            if i == j: continue
            src = state[i]
            dst = state[j]
            if src == 0: continue
            cap_dst = CAPS[j]
            space = cap_dst - dst
            if space == 0: continue
            pour = min(src, space)
            new_state = list(state)
            new_state[i] = src - pour
            new_state[j] = dst + pour
            moves.append(tuple(new_state))
    return moves

def bfs():
    visited = set()
    q = deque()
    q.append((START, [START]))
    visited.add(START)
    while q:
        state, path = q.popleft()
        if goal(state):
            return path
        for ns in possible_moves(state):
            if ns not in visited:
                visited.add(ns)
                q.append((ns, path + [ns]))
    return None

t0 = time.perf_counter()
path = bfs()
t1 = time.perf_counter()
print(f"Time: {t1-t0:.6f} seconds")
if path:
    print(f"Path length: {len(path)}")
    for s in path:
        print(s)
else:
    print("No solution found")
