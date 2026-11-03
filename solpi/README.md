# SoL-Pi mechanisms on ds4: harness and benchmark

This directory holds an agent harness that runs against `ds4-server`. It
implements two of SoL-Pi's mechanisms
([NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi), arXiv 2609.20519) and
measures what they change on local inference. It uses only the Python
standard library and needs no change to the engine.

| File | Role |
|---|---|
| `solpi.py` | Server lifecycle, memory sampler, tools, Action Fusion, ObservationPack, the agent loop, and the task fixture |
| `bench.py` | Runs baseline and SoL-Pi episodes, prints the comparison, writes `results.json` |
| `test_solpi.py` | Offline test with a scripted fake model and simulated prefix caching |

## Architecture

```
bench.py ──► Agent (solpi.py) ──HTTP /v1/chat/completions──► ds4-server (one fresh process per mode)
               │  tools: read_file, write_file, run                      ▲
               │  SoL-Pi: write_and_run  (Action Fusion)                 │ /proc RSS, nvidia-smi
               │          pack >10 KB → handle + 1 KB head/tail,         │
               │          recall(handle, offset, length)                 MemSampler
               └─ per turn: prompt / cached / completion tokens, latency
```

- **Tasks:** three small Python projects, `inventory`, `schedule` and
  `textutil`, each with three bugs. Each suite prints 30-40 KB per run, with
  failure examples and a per-function summary at the end. An episode ends
  when the agent itself sees the suite pass, or after `--turns` model turns.
  A harness-side oracle would let the baseline skip the validation turn that
  Action Fusion saves. Commands run without a Python bytecode cache: a fix
  written in the same second as the previous run, with the same size, would
  otherwise run stale code.
- **Modes:** `baseline`, `fusion` (Action Fusion only), `pack`
  (ObservationPack only) and `solpi` (both). They separate what each
  mechanism contributes.
- **Start:** with `--start log` (the default), every episode opens with the
  failing test output in the task message, full or packed. Every mode must
  read the failures, and no result hinges on a lucky first fix. With
  `--start cold`, the agent explores on its own.
- **Action Fusion:** `write_and_run` writes a file and runs a validation
  command in one tool call.
- **ObservationPack:** any tool result over 10 KB is archived. The model gets
  a handle, the first and last KB, and `recall` for exact pages. Packing
  happens when the result is created; earlier messages are never rewritten,
  so the server's KV prefix cache stays valid. A harness that rewrites
  history would force ds4 to prefill again from the first changed byte.
- **Profiling:**
  - *Tokens:* each response's usage block reports prompt, completion and
    `cached_tokens`. Cached tokens come from the KV prefix cache and are
    almost free locally; the rest is real prefill work.
  - *KV:* the allocated KV comes from the server's startup memory report.
    The live KV at the peak context is estimated from the split between the
    raw window and the compressed rows.
  - *Process:* RSS peak and growth come from `/proc`, and GPU memory from
    `nvidia-smi` when the GPU reports it. Unified-memory GB10 usually reports
    N/A.

## Run

DGX Spark through the Docker wrapper. The server runs in the same container,
so no port needs publishing:

```bash
M=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
sh docker/ds4-docker.sh make
sh docker/ds4-docker.sh run python3 solpi/bench.py -m "$M" -- --cuda
```

Native build with a 24 GB GPU and SSD streaming:

```bash
python3 solpi/bench.py -m "$M" -- --cuda --ssd-streaming --ssd-streaming-cache-experts 1000
```

Offline self-test, with no model: `python3 solpi/test_solpi.py`.

The defaults run 12 episodes: four modes on three tasks. Each starts a fresh
server, so on a Spark the full run takes about 15-20 minutes. ds4 at
temperature 0 is deterministic, so repeats of the same episode are identical.
The three tasks provide the variety, and `--repeat N` only reports medians.
Narrow the run with `--modes` and `--tasks`, e.g.
`--modes baseline,solpi --tasks schedule`.
