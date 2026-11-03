#!/usr/bin/env python3
"""Benchmark SoL-Pi mechanisms against a baseline agent on ds4-server.

    python3 solpi/bench.py -m gguf/MODEL.gguf [options] -- --cuda [more ds4-server flags]
    python3 solpi/bench.py --url http://127.0.0.1:8000        # existing server

Everything after `--` is passed to ds4-server.

Modes ablate the mechanisms: baseline, fusion (Action Fusion only), pack
(ObservationPack only), solpi (both).  Every (task, mode) episode runs on a
fresh workspace and, unless --url is given, a fresh ds4-server, so prefix
caches and memory peaks cannot leak between episodes.  With --start log (the
default) each episode opens with the failing test output, so all modes must
read the failures and none depends on a lucky first fix.

Output: a per-turn trace, a per-task table, the totals over tasks with savings
against the baseline, and solpi-results/<timestamp>/results.json.
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
        "recalls": agent.recalls,
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


def run_episode(task, mode, args, outdir):
    sampler = server = None
    mem = {}
    url = args.url
    fusion, pack = solpi.MODES[mode]
    tag = f"{task}/{mode}"
    try:
        if not url:
            port = free_port()
            cmd = ([args.server, "-m", args.model, "--ctx", str(args.ctx),
                    "--host", "127.0.0.1", "--port", str(port)] + shlex.split(args.server_args))
            server = solpi.Ds4Server(cmd, port, os.path.join(outdir, f"server-{task}-{mode}.log"))
            print(f"[{tag}] starting: {' '.join(cmd)}", flush=True)
            load_s = server.start()
            mem, url = server.mem, server.url
            print(f"[{tag}] server ready in {load_s:.0f}s; KV allocated "
                  f"{mem.get('kv_mb', float('nan')):.0f} MB for ctx {mem.get('ctx', '?')}", flush=True)
            sampler = solpi.MemSampler(server.pid)
            sampler.start()
        elif args.server_pid:
            sampler = solpi.MemSampler(args.server_pid)
            sampler.start()
        module = solpi.TASKS[task][0]
        ws = solpi.new_workspace(outdir, task)
        agent = solpi.Agent(url, ws, module, fusion=fusion, pack=pack, start=args.start,
                            max_turns=args.turns, max_tokens=args.max_tokens).run()
        for i, t in enumerate(agent.turns, 1):
            print(f"[{tag}] turn {i:2d}: prompt {t['prompt_tokens']:6d} (cached {t['cached_tokens']:6d})"
                  f"  out {t['completion_tokens']:5d}  {t['latency_s']:6.1f}s  "
                  f"{','.join(t['tool_calls']) or '-'}", flush=True)
        if sampler:
            sampler.stop()
        res = summarize(agent, mem, sampler)
        res.update(task=task, workspace=ws)
        print(f"[{tag}] {'solved' if res['solved'] else 'NOT solved'} in {res['turns']} turns, "
              f"{res['prefill_tokens']} prefilled tokens, {res['wall_s']:.1f}s", flush=True)
        return res
    finally:
        if sampler:
            sampler.stop()
        if server:
            server.stop()


ROWS = [
    ("solved", "tasks solved", "{}", None),
    ("turns", "model turns", "{:.0f}", True),
    ("tool_calls", "tool calls", "{:.0f}", True),
    ("fused_calls", "  fused (Action Fusion)", "{:.0f}", None),
    ("packed_observations", "  packed (ObservationPack)", "{:.0f}", None),
    ("recalls", "  recall() calls", "{:.0f}", None),
    ("obs_kb_sent", "observation KB sent to model", "{:.1f}", True),
    ("tokens_exchanged", "tokens exchanged (prompt+out)", "{:.0f}", True),
    ("prompt_tokens", "  prompt tokens", "{:.0f}", True),
    ("cached_tokens", "  of which KV prefix reuse", "{:.0f}", None),
    ("prefill_tokens", "  of which prefilled (real cost)", "{:.0f}", True),
    ("completion_tokens", "  completion tokens", "{:.0f}", True),
    ("peak_context_tokens", "peak live context (tokens, max)", "{:.0f}", True),
    ("kv_peak_live_mb", "KV cache peak live (MB, est., max)", "{:.1f}", True),
    ("kv_allocated_mb", "KV cache allocated (MB)", "{:.1f}", None),
    ("rss_peak_mb", "server RSS peak (MB, max)", "{:.0f}", True),
    ("gpu_peak_mb", "GPU memory peak (MB, max)", "{:.0f}", True),
    ("model_s", "time in model (s)", "{:.1f}", True),
    ("wall_s", "wall-clock (s)", "{:.1f}", True),
]
# Peaks combine across tasks by max, everything else by sum.
PEAK_KEYS = {"peak_context_tokens", "kv_peak_live_mb", "kv_allocated_mb", "rss_peak_mb", "gpu_peak_mb"}


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def median_runs(runs):
    """Median of numeric fields across repeats of one (task, mode)."""
    out = dict(runs[0])
    for k in runs[0]:
        vals = [r[k] for r in runs if is_num(r.get(k))]
        if len(vals) == len(runs) and vals:
            out[k] = statistics.median(vals)
    out["solved_count"] = sum(bool(r["solved"]) for r in runs)
    out["runs"] = len(runs)
    return out


def totals(per_task):
    """Combine one mode's per-task medians into totals over tasks."""
    tot = {}
    for k, v in per_task[0].items():
        vals = [r.get(k) for r in per_task]
        if all(is_num(x) for x in vals):
            tot[k] = max(vals) if k in PEAK_KEYS else sum(vals)
    tot["solved"] = f"{sum(r['solved_count'] for r in per_task)}/{sum(r['runs'] for r in per_task)}"
    return tot


def fmt(spec, v):
    if v is None:
        return "n/a"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def report(results, tasks, modes):
    print("\nper task (median over repeats)")
    print(f"{'task':<11}{'mode':<10}{'solved':>7}{'turns':>7}{'prefill':>9}"
          f"{'obs KB':>8}{'wall s':>8}")
    for task in tasks:
        for mode in modes:
            r = results[task][mode]
            print(f"{task:<11}{mode:<10}{r['solved_count']:>4}/{r['runs']:<2}{r['turns']:>7.0f}"
                  f"{r['prefill_tokens']:>9.0f}{r['obs_kb_sent']:>8.1f}{r['wall_s']:>8.1f}")

    tot = {m: totals([results[t][m] for t in tasks]) for m in modes}
    print(f"\ntotals over {len(tasks)} task(s)")
    print(f"{'metric':<36}" + "".join(f"{m:>11}" for m in modes))
    print("-" * (36 + 11 * len(modes)))
    for key, label, spec, _ in ROWS:
        print(f"{label:<36}" + "".join(f"{fmt(spec, tot[m].get(key)):>11}" for m in modes))

    if "baseline" in tot and len(modes) > 1:
        others = [m for m in modes if m != "baseline"]
        print("\nsaving vs baseline: (baseline - mode) / baseline, positive is better")
        print(f"{'metric':<36}" + "".join(f"{m:>11}" for m in others))
        print("-" * (36 + 11 * len(others)))
        base = tot["baseline"]
        for key, label, _, lower_is_better in ROWS:
            if not lower_is_better or not is_num(base.get(key)) or not base[key]:
                continue
            cells = ""
            for m in others:
                v = tot[m].get(key)
                cells += f"{100.0 * (base[key] - v) / base[key]:>+10.1f}%" if is_num(v) else f"{'n/a':>11}"
            print(f"{label:<36}" + cells)
    print("\nPrefilled tokens are what ds4 actually computes; KV prefix reuse is nearly free locally.")
    print("ds4 allocates the KV cache for the whole --ctx at startup: live KV is an estimate of how")
    print("much of it a context fills, not memory the process gives back.")


def parse_cli(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="GGUF model (starts a fresh ds4-server per episode)")
    ap.add_argument("--server", default="./ds4-server", help="ds4-server binary")
    ap.add_argument("--server-args", default="",
                    help="extra ds4-server flags; use --server-args=\"--cuda\" or put them after --")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--url", help="use an already running server instead of starting one")
    ap.add_argument("--server-pid", type=int, help="with --url: PID to sample for memory")
    ap.add_argument("--modes", default=",".join(solpi.MODES), help="comma list of " + ",".join(solpi.MODES))
    ap.add_argument("--tasks", default=",".join(solpi.TASKS), help="comma list of " + ",".join(solpi.TASKS))
    ap.add_argument("--start", choices=("log", "cold"), default="log",
                    help="log: episodes open with the failing test output (default); cold: agent explores")
    ap.add_argument("--turns", type=int, default=10, help="max model turns per episode")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=1,
                    help="episodes per (task, mode); ds4 at temperature 0 is deterministic, so 1 is usually enough")
    ap.add_argument("--out", default="solpi-results")
    server_tail = []
    if "--" in argv:
        cut = argv.index("--")
        argv, server_tail = argv[:cut], argv[cut + 1:]
    args = ap.parse_args(argv)
    args.server_args = " ".join([args.server_args] + [shlex.quote(a) for a in server_tail]).strip()
    if not args.url and not args.model:
        ap.error("pass -m MODEL or --url URL")
    for name, valid in (("modes", solpi.MODES), ("tasks", solpi.TASKS)):
        bad = [x for x in getattr(args, name).split(",") if x not in valid]
        if bad:
            ap.error(f"unknown {name}: {','.join(bad)}")
    return args


def main():
    args = parse_cli(sys.argv[1:])
    modes, tasks = args.modes.split(","), args.tasks.split(",")
    outdir = os.path.abspath(os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(outdir, exist_ok=True)
    runs = {t: {m: [run_episode(t, m, args, outdir) for _ in range(args.repeat)] for m in modes}
            for t in tasks}
    results = {t: {m: median_runs(runs[t][m]) for m in modes} for t in tasks}
    report(results, tasks, modes)
    path = os.path.join(outdir, "results.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f, indent=2, default=str)
    print(f"\nfull results: {path}")


if __name__ == "__main__":
    main()
