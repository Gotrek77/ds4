#!/usr/bin/env python3
"""Offline test of the harness: a scripted fake model behind a fake OpenAI
server, which reports cached_tokens like ds4 does (the longest prompt prefix
shared with the previous request, in ~4-byte tokens)."""

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import solpi  # noqa: E402

FIXED = (solpi.INVENTORY
         .replace("int(packs) + int(size)", "int(packs) * int(size)")
         .replace("int(euros) * 100 + int(cents or 0)",
                  "int(euros) * 100 + (int((cents + \"00\")[:2]) if cents else 0)")
         .replace("line_total_cents(*l))[:n]", "line_total_cents(*l), reverse=True)[:n]"))
TEST = "python3 run_tests.py"


def call(name, **args):
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def policy(messages, tools):
    """The same plan in both modes, using the fused tool when offered."""
    fused = any(t["function"]["name"] == "write_and_run" for t in tools)
    done = sum(m["role"] == "assistant" for m in messages)
    plan = [call("run", command=TEST), call("read_file", path="inventory.py")]
    if fused:
        plan += [call("recall", handle="obs-1", offset=0, length=200),
                 call("write_and_run", path="inventory.py", content=FIXED, command=TEST)]
    else:
        plan += [call("write_file", path="inventory.py", content=FIXED), call("run", command=TEST)]
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
    assert abs(solpi.kv_live_mb(mem, 4352) - (mem["raw_mb"] + mem["comp_mb"] * 4352 / 32768)) < 1e-6

    results = {}
    for mode in ("baseline", "solpi"):
        Fake.last = ""
        a = solpi.Agent(url, mode, solpi.new_workspace(tmp), max_turns=10).run()
        results[mode] = bench.summarize(a, mem, None)
    b, s = results["baseline"], results["solpi"]
    assert b["solved"] and s["solved"]
    assert b["turns"] == 4 and s["turns"] == 4          # run, read, (recall|write), (fused|run)
    assert s["fused_calls"] == 1 and b["fused_calls"] == 0
    assert s["packed_observations"] == 2 and b["packed_observations"] == 0
    assert s["obs_kb_sent"] < b["obs_kb_sent"] / 5
    assert s["prompt_tokens"] < b["prompt_tokens"] and s["peak_context_tokens"] < b["peak_context_tokens"]
    assert all(t["cached_tokens"] > 0 for t in s["turn_trace"][1:])
    bench.report(results)
    srv.shutdown()
    shutil.rmtree(tmp)
    print("\ntest_solpi: ok")


if __name__ == "__main__":
    main()
