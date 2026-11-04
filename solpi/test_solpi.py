#!/usr/bin/env python3
"""Offline test of the harness: a scripted fake model behind a fake OpenAI
server, which reports cached_tokens like ds4 does (the longest prompt prefix
shared with the previous request, in ~4-byte tokens)."""

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import solpi  # noqa: E402

FIXES = {
    "inventory": [("int(packs) + int(size)", "int(packs) * int(size)"),
                  ("int(euros) * 100 + int(cents or 0)",
                   "int(euros) * 100 + (int((cents + \"00\")[:2]) if cents else 0)"),
                  ("line_total_cents(*l))[:n]", "line_total_cents(*l), reverse=True)[:n]")],
    "schedule": [("return year % 4 == 0", "return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"),
                 ("month in (4, 6, 9)", "month in (4, 6, 9, 11)"),
                 ("            month = 1\n", "            month = 1\n            year += 1\n")],
    "textutil": [(".lstrip(\"-\")", ".strip(\"-\")"), ("text.split(\" \")", "text.split()"),
                 ("text[:width] + \"...\"", "text[:width - 3] + \"...\"")],
}
TEST = "python3 run_tests.py"


def fixed(task):
    src = solpi.TASKS[task][1]
    for a, b in FIXES[task]:
        assert a in src, (task, a)
        src = src.replace(a, b)
    return src


def call(name, **args):
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def policy(messages, tools):
    """The same plan in every mode, using the fused tool when it is offered.
    A cold start runs the suite first; a log start already has the output."""
    names = {t["function"]["name"] for t in tools}
    module = re.search(r"bugs in (\w+\.py)", messages[1]["content"]).group(1)
    task = module[:-3]
    done = sum(m["role"] == "assistant" for m in messages)
    plan = [] if "Current test output" in messages[1]["content"] else [call("run", command=TEST)]
    plan.append(call("read_file", path=module))
    if "recall" in names:
        plan.append(call("recall", handle="obs-1", offset=0, length=200))
    if "write_and_run" in names:
        plan.append(call("write_and_run", path=module, content=fixed(task), command=TEST))
    else:
        plan += [call("write_file", path=module, content=fixed(task)), call("run", command=TEST)]
    if done < len(plan):
        return {"role": "assistant", "content": "", "tool_calls": [plan[done]]}
    return {"role": "assistant", "content": "done"}


class Fake(BaseHTTPRequestHandler):
    last = ""

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = json.dumps(req["messages"])
        common = len(os.path.commonprefix([prompt, Fake.last]))
        Fake.last = prompt
        msg = policy(req["messages"], req["tools"])
        body = json.dumps({"choices": [{"message": msg}], "usage": {
            "prompt_tokens": len(prompt) // 4, "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": common // 4}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    tmp = tempfile.mkdtemp(prefix="solpi-test-")

    mem = solpi.parse_memory_report(
        "ds4: memory: KV 0.78 GiB (raw 0.36 + compressed 0.42) + buffers\n"
        "ds4: memory detail: ctx=32768 prefill_cap=4096 raw_kv_rows=4352 compressed_kv_rows=8194\n")
    assert abs(mem["kv_mb"] - 798.72) < 0.01 and mem["ctx"] == 32768 and mem["raw_rows"] == 4352

    # The sampler must start, sample a real process and stop cleanly, twice.
    sampler = solpi.MemSampler(os.getpid(), period=0.01)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    sampler.stop()
    assert sampler.rss_peak_mb > 0 and not sampler.is_alive()

    # Every fixture fails as shipped and passes with its three fixes.
    for task, (module, _, _) in solpi.TASKS.items():
        ws = solpi.Workspace(solpi.new_workspace(tmp, task))
        out = ws.run(TEST)
        assert out.rstrip().endswith("[exit 1]") and len(out) > solpi.PACK_THRESHOLD, task
        ws.write(module, fixed(task))
        assert ws.run(TEST).rstrip().endswith("[exit 0]"), task

    # A piped run exits 0 although the suite fails: it must not end the episode.
    a = solpi.Agent("http://unused", solpi.new_workspace(tmp, "inventory"), "inventory.py")
    a.observe("run", {"command": "python3 run_tests.py | tail -3"})
    assert not a.passed_seen
    a.ws.write("inventory.py", fixed("inventory"))
    a.observe("run", {"command": "python3 run_tests.py | tail -3"})
    assert a.passed_seen

    # A server that rejects the request ends the episode with an error, not a crash.
    class Reject(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"error":{"message":"prompt exceeds context"}}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    rej = HTTPServer(("127.0.0.1", 0), Reject)
    threading.Thread(target=rej.serve_forever, daemon=True).start()
    a = solpi.Agent(f"http://127.0.0.1:{rej.server_address[1]}",
                    solpi.new_workspace(tmp, "schedule"), "schedule.py").run()
    assert a.error.startswith("HTTP 400") and not a.solved_ok and a.turns == []
    rej.shutdown()

    for start in ("cold", "log"):
        results = {}
        for task, (module, _, _) in solpi.TASKS.items():
            results[task] = {}
            for mode, (fusion, pack) in solpi.MODES.items():
                Fake.last = ""
                a = solpi.Agent(url, solpi.new_workspace(tmp, task), module,
                                fusion=fusion, pack=pack, start=start).run()
                r = bench.summarize(a, mem, None)
                assert r["solved"], (start, task, mode)
                assert r["fused_calls"] == int(fusion)
                assert (r["packed_observations"] > 0) == pack
                assert r["recalls"] == int(pack)
                results[task][mode] = bench.median_runs([r])
        for task in solpi.TASKS:
            b, f, p, s = (results[task][m] for m in solpi.MODES)
            assert f["turns"] == b["turns"] - 1               # fusion saves the validation turn
            assert p["obs_kb_sent"] < b["obs_kb_sent"] / 4    # pack shrinks what the model reads
            assert s["prefill_tokens"] < b["prefill_tokens"]
        print(f"\n===== start={start}")
        bench.report(results, list(solpi.TASKS), list(solpi.MODES))

    a = bench.parse_cli(["-m", "x.gguf", "--repeat", "3", "--", "--cuda", "--ssd-streaming"])
    assert a.repeat == 3 and a.server_args == "--cuda --ssd-streaming"
    assert bench.parse_cli(["-m", "x", "--server-args=--cuda"]).server_args == "--cuda"

    srv.shutdown()
    shutil.rmtree(tmp)
    print("\ntest_solpi: ok")


if __name__ == "__main__":
    main()
