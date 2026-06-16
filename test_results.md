# Performance Test: Water Jug Problem (12L, 7L, 5L)

## Problem
Three bottles of capacities 12L, 7L, 5L. Initially 12L full, others empty.
Goal: obtain exactly 1 liter in any bottle (since 1,2,3,4 are not directly measurable).

## Solution (BFS)
Path found (9 steps):
```
(12,0,0) -> (7,0,5) -> (7,5,0) -> (2,5,5) -> (2,7,3) -> (9,0,3) -> (9,3,0) -> (4,3,5) -> (4,7,1)
```

## Performance

### Without Lobster (Python BFS)
- Average over 1000 runs: **31.5 µs** (microseconds)
- Algorithm: BFS over state space (max 624 states)
- Pure Python, no external dependencies.

### With Lobster (Scallop Datalog engine via scli CLI)
- Average over 100 runs: **12.2 ms** (milliseconds)
- Includes process fork+exec overhead (spawning scli, loading compiler, parsing, evaluation).
- The Datalog evaluation itself is sub-millisecond; the overhead dominates.

### Comparison
- Lobster (as CLI tool) is ~387× slower than the Python BFS.
- An in-process Datalog engine (e.g., `scallopy` Python binding) would avoid fork overhead and likely be competitive with the Python BFS.
- For this small state space, a hand-coded BFS is the most efficient approach.

## Files
- `test_waterjug.py` – Python BFS solver and benchmark.
- `waterjug.scl` – Scallop Datalog program for the same problem.
- `test_lobster.c` – C program that calls `lobster_tool_exec()` to run the Datalog program and measures time.
