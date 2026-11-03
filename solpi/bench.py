#!/usr/bin/env python3
"""Benchmark SoL-Pi mechanisms against a baseline agent on ds4-server.

    python3 solpi/bench.py -m gguf/MODEL.gguf [--server-args "--cuda ..."]
    python3 solpi/bench.py --url http://127.0.0.1:8000        # existing server

Each mode runs the same multi-step debugging task (a 3-bug project with a
~40 KB test log) for at most --turns model turns.  By default every mode gets
its own fresh ds4-server, so prefix caches and memory peaks cannot leak from
one mode into the other.  Results: a per-turn trace, a comparison table and
solpi-results/<timestamp>.json.
"""

import argparse
import json
import os
import shlex
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solpi  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def summarize(agent, server_mem, sampler):
    t = agent.turns
    prompt = sum(x["prompt_tokens"] for x in t)
    completion = sum(x["completion_tokens"] for x in t)
    cached = sum(x["cached_tokens"] for x in t)
    peak_ctx = max((x["prompt_tokens"] + x["completion_tokens"] for x in t), default=0)
    return {
        "mode": agent.mode,
        "solved": agent.solved_ok,
        "turns": len(t),
        "tool_calls": agent.tool_calls,
        "fused_calls": agent.fused_calls,
        "packed_observations": agent.packed,
        "obs_kb_raw": agent.obs_bytes_raw / 1024,
        "obs_kb_sent": agent.obs_bytes_sent / 1024,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tokens_exchanged": prompt + completion,
        "cached_tokens": cached,
        "prefill_tokens": prompt - cached,
        "peak_context_tokens": peak_ctx,
        "kv_allocated_mb": server_mem.get("kv_mb"),
        "kv_peak_live_mb": solpi.kv_live_mb(server_mem, peak_ctx),
        "rss_peak_mb": sampler.rss_peak_mb if sampler else None,
        "rss_growth_mb": (sampler.rss_peak_mb - sampler.rss_start_mb)
                         if sampler and sampler.rss_start_mb is not None else None,
        "gpu_peak_mb": sampler.gpu_peak_mb if sampler else None,
        "wall_s": agent.wall_s,
        "model_s": sum(x["latency_s"] for x in t),
        "turn_trace": t,
    }


def run_mode(mode, args, outdir):
    sampler = None
    server = None
    mem = {}
    url = args.url
    try:
        if not url:
            port = free_port()
            cmd = ([args.server, "-m", args.model, "--ctx", str(args.ctx),
                    "--host", "127.0.0.1", "--port", str(port)] + shlex.split(args.server_args))
            server = solpi.Ds4Server(cmd, port, os.path.join(outdir, f"server-{mode}.log"))
            print(f"[{mode}] starting: {' '.join(cmd)}", flush=True)
            load_s = server.start()
            mem = server.mem
            url = server.url
            print(f"[{mode}] server ready in {load_s:.0f}s; KV allocated "
                  f"{mem.get('kv_mb', float('nan')):.0f} MB for ctx {mem.get('ctx', '?')}", flush=True)
            sampler = solpi.MemSampler(server.pid)
            sampler.start()
        elif args.server_pid:
            sampler = solpi.MemSampler(args.server_pid)
            sampler.start()
        ws = solpi.new_workspace(outdir)
        agent = solpi.Agent(url, mode, ws, max_turns=args.turns, max_tokens=args.max_tokens)
        agent.run()
        for i, t in enumerate(agent.turns, 1):
            print(f"[{mode}] turn {i:2d}: prompt {t['prompt_tokens']:6d} (cached {t['cached_tokens']:6d})"
                  f"  out {t['completion_tokens']:5d}  {t['latency_s']:6.1f}s  "
                  f"{','.join(t['tool_calls']) or '-'}", flush=True)
        if sampler:
            sampler.stop()
        res = summarize(agent, mem, sampler)
        res["workspace"] = ws
        return res
    finally:
        if sampler:
            sampler.stop()
        if server:
            server.stop()


ROWS = [
    ("solved", "task solved", "{}", None),
    ("turns", "model turns", "{:.0f}", True),
    ("tool_calls", "tool calls", "{:.0f}", True),
    ("fused_calls", "  fused (Action Fusion)", "{:.0f}", None),
    ("packed_observations", "  packed (ObservationPack)", "{:.0f}", None),
    ("obs_kb_sent", "observation KB sent to model", "{:.1f}", True),
    ("tokens_exchanged", "tokens exchanged (prompt+out)", "{:.0f}", True),
    ("prompt_tokens", "  prompt tokens", "{:.0f}", True),
    ("cached_tokens", "  of which KV prefix reuse", "{:.0f}", None),
    ("prefill_tokens", "  of which prefilled (real cost)", "{:.0f}", True),
    ("completion_tokens", "  completion tokens", "{:.0f}", True),
    ("peak_context_tokens", "peak live context (tokens)", "{:.0f}", True),
    ("kv_peak_live_mb", "KV cache peak live (MB, est.)", "{:.1f}", True),
    ("kv_allocated_mb", "KV cache allocated (MB)", "{:.1f}", None),
    ("rss_peak_mb", "server RSS peak (MB)", "{:.0f}", True),
    ("rss_growth_mb", "server RSS growth (MB)", "{:.0f}", True),
    ("gpu_peak_mb", "GPU memory peak (MB)", "{:.0f}", True),
    ("model_s", "time in model (s)", "{:.1f}", True),
    ("wall_s", "wall-clock (s)", "{:.1f}", True),
]


def median_runs(runs):
    """Median of numeric fields across repeats (first run for the rest)."""
    out = dict(runs[0])
    for k, v in runs[0].items():
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if len(vals) == len(runs) and vals:
            out[k] = statistics.median(vals)
    out["solved"] = f"{sum(bool(r['solved']) for r in runs)}/{len(runs)}"
    return out


def fmt(fmtspec, v):
    if v is None:
        return "n/a"
    try:
        return fmtspec.format(v)
    except (ValueError, TypeError):
        return str(v)


def report(results):
    base, sol = results.get("baseline"), results.get("solpi")
    cols = [m for m in ("baseline", "solpi") if m in results]
    print()
    print(f"{'metric':<36}" + "".join(f"{c:>14}" for c in cols) + ("    saving" if base and sol else ""))
    print("-" * (36 + 14 * len(cols) + (10 if base and sol else 0)))
    for key, label, f, lower_is_better in ROWS:
        line = f"{label:<36}" + "".join(f"{fmt(f, results[c].get(key)):>14}" for c in cols)
        if base and sol and lower_is_better:
            b, s = base.get(key), sol.get(key)
            if isinstance(b, (int, float)) and isinstance(s, (int, float)) and b:
                line += f"   {100.0 * (b - s) / b:+6.1f}%"
        print(line)
    print()
    print("saving = (baseline - solpi) / baseline; positive is better for SoL-Pi.")
    print("Prefilled tokens are what ds4 actually computes: KV prefix reuse is nearly free locally.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="GGUF model (starts its own ds4-server per mode)")
    ap.add_argument("--server", default="./ds4-server", help="ds4-server binary")
    ap.add_argument("--server-args", default="", help="extra ds4-server flags, e.g. \"--cuda\"")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--url", help="use an already running server instead of starting one")
    ap.add_argument("--server-pid", type=int, help="with --url: PID to sample for memory")
    ap.add_argument("--modes", default="baseline,solpi")
    ap.add_argument("--turns", type=int, default=10, help="max model turns per episode")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=1, help="episodes per mode (median reported)")
    ap.add_argument("--out", default="solpi-results")
    args = ap.parse_args()
    if not args.url and not args.model:
        ap.error("pass -m MODEL or --url URL")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.abspath(os.path.join(args.out, stamp))
    os.makedirs(outdir, exist_ok=True)
    all_runs = {}
    for mode in args.modes.split(","):
        all_runs[mode] = [run_mode(mode, args, outdir) for _ in range(args.repeat)]
    results = {m: median_runs(r) for m, r in all_runs.items()}
    report(results)
    path = os.path.join(outdir, "results.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": all_runs}, f, indent=2, default=str)
    print(f"\nfull results: {path}")


if __name__ == "__main__":
    main()
