"""SoL-Pi mechanisms and a profiling agent harness around ds4-server.

The harness is the agent: it drives ds4-server through the OpenAI-compatible
chat API with tool calling, executes the tools locally, and records what every
turn costs.  The two SoL-Pi mechanisms live entirely on the harness side,
exactly where SoL-Pi puts them in Pi:

  Action Fusion   write_and_run(path, content, command): an edit and its
                  validation command in one tool call, saving a model turn.
  ObservationPack tool results over PACK_THRESHOLD bytes are archived and
                  replaced by a handle plus the first and last PACK_VIEW
                  bytes; recall(handle, offset, length) pages exact bytes.

Profiling comes from three places: the usage block of every response (prompt,
completion and cached tokens: cached tokens are KV prefix reuse, the rest
had to be prefilled), the server's own memory report at startup (allocated KV
and its split between the raw window and compressed rows), and sampling of the
server process (RSS from /proc, GPU memory from nvidia-smi when it reports it).

Standard library only.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

PACK_THRESHOLD = 10 * 1024
PACK_VIEW = 1024
RECALL_MAX = 4096
RUN_TIMEOUT = 120


# ---------------------------------------------------------------- server

class Ds4Server:
    """A ds4-server process, its startup memory report and its PID."""

    def __init__(self, cmd, port, log_path, url=None):
        self.cmd = cmd
        self.port = port
        self.log_path = log_path
        self.url = url or f"http://127.0.0.1:{port}"
        self.proc = None
        self.mem = {}

    def start(self, timeout=1800):
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ds4-server exited early, see {self.log_path}")
            text = self._log_text()
            if "listening on" in text:
                self.mem = parse_memory_report(text)
                return time.time() - t0
            time.sleep(0.5)
        raise RuntimeError(f"ds4-server did not start in {timeout}s, see {self.log_path}")

    def _log_text(self):
        with open(self.log_path, "rb") as f:
            return f.read().decode("utf-8", "replace")

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc:
            self.log.close()


def parse_memory_report(text):
    """ds4 prints the KV allocation once per context:
         ds4: memory: KV 0.78 GiB (raw 0.36 + compressed 0.42) + ...
         ds4: memory detail: ctx=32768 ... raw_kv_rows=4352 compressed_kv_rows=8194
    """
    mem = {}
    m = re.search(r"memory: KV ([\d.]+) GiB \(raw ([\d.]+) \+ compressed ([\d.]+)\)", text)
    if m:
        mem["kv_mb"], mem["raw_mb"], mem["comp_mb"] = (float(x) * 1024 for x in m.groups())
    m = re.search(r"memory detail: ctx=(\d+).*?raw_kv_rows=(\d+)", text)
    if m:
        mem["ctx"], mem["raw_rows"] = int(m.group(1)), int(m.group(2))
    return mem


def kv_live_mb(mem, tokens):
    """Estimate the KV bytes a context of `tokens` actually fills.

    DeepSeek V4 keeps a raw sliding window (filled up to raw_rows) plus
    compressed rows that grow linearly with the context."""
    if not mem.get("ctx"):
        return None
    raw = mem["raw_mb"] * min(1.0, tokens / max(1, mem["raw_rows"]))
    return raw + mem["comp_mb"] * min(1.0, tokens / mem["ctx"])


class MemSampler(threading.Thread):
    """Sample the server's RSS (and GPU memory when available) until stopped."""

    def __init__(self, pid, period=0.25):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period
        self.rss_peak_mb = 0.0
        self.rss_start_mb = None
        self.gpu_peak_mb = None
        self._halt = threading.Event()
        self._gpu_ok = shutil.which("nvidia-smi") is not None

    def _rss_mb(self):
        try:
            with open(f"/proc/{self.pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return None

    def _gpu_mb(self):
        if not self._gpu_ok:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            self._gpu_ok = False
            return None
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid) and parts[1].isdigit():
                return float(parts[1])
        return None  # unified-memory GPUs (GB10) often report N/A

    def run(self):
        tick = 0
        while not self._halt.is_set():
            rss = self._rss_mb()
            if rss is not None:
                if self.rss_start_mb is None:
                    self.rss_start_mb = rss
                self.rss_peak_mb = max(self.rss_peak_mb, rss)
            if tick % 4 == 0:
                gpu = self._gpu_mb()
                if gpu is not None:
                    self.gpu_peak_mb = max(self.gpu_peak_mb or 0.0, gpu)
            tick += 1
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


# ---------------------------------------------------------------- client

def chat(url, payload, timeout=3600):
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dsv4-local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- tools

def tool(name, desc, **props):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {
            k: {"type": t, "description": d} for k, (t, d) in props.items()},
            "required": list(props)}}}


BASE_TOOLS = [
    tool("read_file", "Read a file of the project.", path=("string", "Relative path.")),
    tool("write_file", "Replace a file with new content.",
         path=("string", "Relative path."), content=("string", "Full new file content.")),
    tool("run", "Run a shell command in the project directory and return its output.",
         command=("string", "Shell command.")),
]

FUSION_TOOLS = [
    tool("write_and_run",
         "Replace a file and immediately run a validation command, in one call. "
         "Prefer this over write_file followed by run.",
         path=("string", "Relative path."), content=("string", "Full new file content."),
         command=("string", "Validation command to run after writing.")),
]

PACK_TOOLS = [
    tool("recall", "Read exact bytes of a packed observation by handle.",
         handle=("string", "Observation handle, e.g. obs-1."),
         offset=("integer", "Byte offset."), length=("integer", "Bytes to read (max 4096).")),
]

# Ablation modes: each SoL-Pi mechanism alone, then both.
MODES = {
    "baseline": (False, False),
    "fusion": (True, False),
    "pack": (False, True),
    "solpi": (True, True),
}


class ObservationPack:
    """Large observations become a stable handle plus a head/tail view; the
    full text stays archived for exact paged recall."""

    def __init__(self, threshold=PACK_THRESHOLD, view=PACK_VIEW):
        self.threshold, self.view = threshold, view
        self.archive = {}

    def pack(self, text):
        data = text.encode("utf-8", "replace")
        if len(data) <= self.threshold:
            return text, False
        handle = f"obs-{len(self.archive) + 1}"
        self.archive[handle] = data
        head = data[:self.view].decode("utf-8", "replace")
        tail = data[-self.view:].decode("utf-8", "replace")
        elided = len(data) - 2 * self.view
        return (f"[{handle}: {len(data)} bytes, showing first and last {self.view}; "
                f"recall(handle, offset, length) returns exact bytes]\n{head}\n"
                f"... [{elided} bytes elided] ...\n{tail}"), True

    def recall(self, handle, offset, length):
        data = self.archive.get(handle)
        if data is None:
            return f"error: unknown handle {handle}"
        offset = max(0, int(offset))
        length = max(0, min(int(length), RECALL_MAX))
        return data[offset:offset + length].decode("utf-8", "replace")


class Workspace:
    def __init__(self, root):
        self.root = os.path.realpath(root)

    def _path(self, rel):
        p = os.path.realpath(os.path.join(self.root, rel))
        if os.path.commonpath([p, self.root]) != self.root:
            raise ValueError(f"path escapes the project: {rel}")
        return p

    def read(self, path):
        with open(self._path(path), encoding="utf-8", errors="replace") as f:
            return f.read()

    def write(self, path, content):
        p = self._path(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content.encode())} bytes to {path}"

    def run(self, command):
        try:
            # No bytecode cache: a fix written within the same second as the
            # last run and with the same size would otherwise run stale code.
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run(["bash", "-c", command], cwd=self.root, capture_output=True,
                               text=True, timeout=RUN_TIMEOUT, env=env)
            return f"$ {command}\n{r.stdout}{r.stderr}[exit {r.returncode}]"
        except subprocess.TimeoutExpired:
            return f"$ {command}\n[timeout after {RUN_TIMEOUT}s]"


# ---------------------------------------------------------------- agent

SYSTEM = ("You are a careful software engineer working in a small Python project. "
          "Use the tools to find and fix the bugs until `python3 run_tests.py` passes. "
          "Be brief: act with tools, do not explain at length.")
FUSION_HINT = (" When you change a file, use write_and_run to write it and run the tests in "
               "one step.")
PACK_HINT = (" Long outputs are packed: the first and last KB are shown and recall() pages "
             "the rest only if you need it.")
TEST_CMD = "python3 run_tests.py"


class Agent:
    """One agent episode on a fresh workspace.

    fusion and pack select the SoL-Pi mechanisms.  start="log" opens the
    episode with the failing test output in the task message (full, or packed
    when pack is on), so every mode must read the failures and the comparison
    does not hinge on a lucky first fix; start="cold" lets the agent explore."""

    def __init__(self, url, workspace, module, fusion=False, pack=False, start="log",
                 max_turns=10, max_tokens=4096):
        assert start in ("log", "cold")
        self.url, self.ws, self.module = url, Workspace(workspace), module
        self.fusion, self.start = fusion, start
        self.max_turns, self.max_tokens = max_turns, max_tokens
        self.pack = ObservationPack() if pack else None
        self.tools = BASE_TOOLS + (FUSION_TOOLS if fusion else []) + (PACK_TOOLS if pack else [])
        self.mode = {v: k for k, v in MODES.items()}[(fusion, pack)]
        self.turns = []        # one record per model request
        self.tool_calls = 0
        self.fused_calls = 0
        self.recalls = 0
        self.packed = 0
        self.obs_bytes_raw = 0
        self.obs_bytes_sent = 0
        self.passed_seen = False

    def _execute(self, name, args):
        if name == "read_file":
            return self.ws.read(args["path"])
        if name == "write_file":
            return self.ws.write(args["path"], args["content"])
        if name == "run":
            return self.ws.run(args["command"])
        if self.fusion and name == "write_and_run":
            self.fused_calls += 1
            return self.ws.write(args["path"], args["content"]) + "\n" + self.ws.run(args["command"])
        if self.pack and name == "recall":
            self.recalls += 1
            return self.pack.recall(args["handle"], args.get("offset", 0), args.get("length", RECALL_MAX))
        return f"error: unknown tool {name}"

    def _deliver(self, name, out):
        """Account an observation and pack it when ObservationPack is on."""
        self.obs_bytes_raw += len(out.encode("utf-8", "replace"))
        if self.pack and name != "recall":
            out, packed = self.pack.pack(out)
            self.packed += packed
        self.obs_bytes_sent += len(out.encode("utf-8", "replace"))
        return out

    def observe(self, name, args):
        try:
            out = self._execute(name, args)
        except Exception as e:  # tool errors are observations, not crashes
            out = f"error: {e}"
        # The episode ends when the agent itself has seen the suite pass: a
        # harness-side oracle would let the baseline skip the validation turn
        # that Action Fusion exists to save.
        # The suite's own summary decides, not the exit code: `run_tests.py |
        # tail` exits 0 even when the suite fails.
        if name in ("run", "write_and_run") and "run_tests" in str(args.get("command", "")) \
                and re.search(r"SUMMARY: \d+ cases, 0 failures", out):
            self.passed_seen = True
        return self._deliver(name, out)

    def solved(self):
        cmd = TEST_CMD + " > /dev/null 2>&1"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(["bash", "-c", cmd], cwd=self.ws.root, env=env).returncode == 0

    def task_message(self):
        task = (f"The test suite fails. Fix the bugs in {self.module} so that "
                f"`{TEST_CMD}` passes.")
        if self.start == "cold":
            return task
        return task + "\n\nCurrent test output:\n" + self._deliver("run", self.ws.run(TEST_CMD))

    def run(self):
        system = SYSTEM + (FUSION_HINT if self.fusion else "") + (PACK_HINT if self.pack else "")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": self.task_message()}]
        t0 = time.time()
        for _ in range(self.max_turns):
            payload = {"model": "deepseek-v4-flash", "messages": messages, "tools": self.tools,
                       "temperature": 0, "max_tokens": self.max_tokens, "think": False,
                       "stream": False}
            t_req = time.time()
            resp = chat(self.url, payload)
            latency = time.time() - t_req
            usage = resp.get("usage", {})
            details = usage.get("prompt_tokens_details") or {}
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            self.turns.append({
                "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cached_tokens": details.get("cached_tokens", 0),
                "tool_calls": [c["function"]["name"] for c in calls],
            })
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            if msg.get("reasoning_content"):
                assistant["reasoning_content"] = msg["reasoning_content"]
            messages.append(assistant)
            if not calls:
                break
            for c in calls:
                self.tool_calls += 1
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": self.observe(c["function"]["name"], args)})
            if self.passed_seen:
                break
        self.wall_s = time.time() - t0
        self.solved_ok = self.solved()
        return self


# ---------------------------------------------------------------- fixtures

# Every task is a small module with three bugs and a verbose suite (one log
# line per case, ~25-40 KB) whose last lines hold failure examples and a
# per-function summary, so a packed view's last KB carries what matters.

INVENTORY = '''\
"""Order arithmetic for a small shop."""


def parse_qty(text):
    """Quantity like "12" or "3x4" (3 packs of 4 pieces)."""
    text = text.strip()
    if "x" in text:
        packs, size = text.split("x")
        return int(packs) + int(size)
    return int(text)


def unit_price_cents(price):
    """Price string in euros like "12.5" or "0.99" -> integer cents."""
    euros, _, cents = price.partition(".")
    return int(euros) * 100 + int(cents or 0)


def line_total_cents(qty, price):
    return parse_qty(qty) * unit_price_cents(price)


def top_lines(lines, n):
    """The n most expensive (qty, price) lines, most expensive first."""
    return sorted(lines, key=lambda l: line_total_cents(*l))[:n]
'''

SCHEDULE = '''\
"""Calendar helpers for a booking system."""


def is_leap(year):
    return year % 4 == 0


def days_in_month(year, month):
    if month == 2:
        return 29 if is_leap(year) else 28
    return 30 if month in (4, 6, 9) else 31


def add_days(year, month, day, n):
    """The date n >= 0 days after (year, month, day), as a tuple."""
    day += n
    while day > days_in_month(year, month):
        day -= days_in_month(year, month)
        month += 1
        if month > 12:
            month = 1
    return year, month, day
'''

TEXTUTIL = '''\
"""Text helpers for a static site generator."""
import re


def slugify(title):
    """Lowercase; every run of non-alphanumerics becomes one '-'; no '-' at the ends."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).lstrip("-")


def word_count(text):
    """Number of whitespace-separated words."""
    return len(text.split(" "))


def truncate(text, width):
    """At most `width` characters; a cut text ends with '...' counted in the width."""
    if len(text) <= width:
        return text
    return text[:width] + "..."
'''

RUNNER_HEAD = r'''"""Verbose regression suite: one log line per case, summary at the end."""
import random
import sys
import time

import MODULE as m

rng = random.Random(1234)
fails = {}
total = 0
start = time.time()


def check(fn, arg, got, want):
    global total
    total += 1
    status = "ok  " if got == want else "FAIL"
    print(f"[{total:04d}] {status} {fn}({arg!r}) -> {got!r} (expected {want!r}) "
          f"t+{(time.time() - start) * 1000:.3f}ms")
    if got != want:
        fails.setdefault(fn, []).append((arg, got, want))

'''

RUNNER_TAIL = r'''
print()
for fn, items in fails.items():
    for arg, got, want in items[:2]:
        print(f"example: {fn}({arg!r}) -> {got!r}, expected {want!r}")
print(f"SUMMARY: {total} cases, {sum(len(v) for v in fails.values())} failures")
for fn, items in fails.items():
    print(f"  {fn}: {len(items)} failures")
sys.exit(1 if fails else 0)
'''

INVENTORY_CASES = r'''
def ref_qty(t):
    t = t.strip()
    if "x" in t:
        a, b = t.split("x")
        return int(a) * int(b)
    return int(t)


def ref_cents(p):
    e, _, c = p.partition(".")
    return int(e) * 100 + int((c + "00")[:2]) if c else int(e) * 100


for _ in range(160):
    q = rng.choice([str(rng.randint(1, 50)), f"{rng.randint(1, 9)}x{rng.randint(2, 12)}"])
    check("parse_qty", q, m.parse_qty(q), ref_qty(q))
for _ in range(160):
    p = rng.choice([f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}",
                    f"{rng.randint(0, 99)}.{rng.randint(0, 9)}", str(rng.randint(1, 99))])
    check("unit_price_cents", p, m.unit_price_cents(p), ref_cents(p))
for _ in range(80):
    lines = [(str(rng.randint(1, 9)), f"{rng.randint(1, 30)}.{rng.randint(0, 99):02d}")
             for _ in range(6)]
    want = sorted(lines, key=lambda l: ref_qty(l[0]) * ref_cents(l[1]), reverse=True)[:3]
    check("top_lines", lines, m.top_lines(lines, 3), want)
'''

SCHEDULE_CASES = r'''
import calendar
import datetime

for _ in range(140):
    y = rng.choice([rng.randint(1600, 2400), rng.choice([1700, 1800, 1900, 2000, 2100, 2400])])
    check("is_leap", y, m.is_leap(y), calendar.isleap(y))
for _ in range(140):
    y, mo = rng.randint(1890, 2110), rng.randint(1, 12)
    check("days_in_month", (y, mo), m.days_in_month(y, mo), calendar.monthrange(y, mo)[1])
for _ in range(120):
    d = datetime.date(rng.randint(1890, 2110), rng.randint(1, 12), rng.randint(1, 28))
    n = rng.randint(0, 400)
    e = d + datetime.timedelta(days=n)
    check("add_days", (d.year, d.month, d.day, n), m.add_days(d.year, d.month, d.day, n),
          (e.year, e.month, e.day))
'''

TEXTUTIL_CASES = r'''
import re

WORDS = ["Hello", "world", "DS4", "fast", "C", "inference", "on", "a", "Spark", "2026"]
SEPS = [" ", "  ", ", ", "! ", " - ", "\n", "\t", "?"]


def phrase():
    out = rng.choice(["", " ", "-", "  "])
    for _ in range(rng.randint(1, 6)):
        out += rng.choice(WORDS) + rng.choice(SEPS)
    return out


def ref_slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def ref_trunc(t, w):
    return t if len(t) <= w else t[:w - 3] + "..."


for _ in range(140):
    t = phrase()
    check("slugify", t, m.slugify(t), ref_slug(t))
for _ in range(140):
    t = phrase()
    check("word_count", t, m.word_count(t), len(t.split()))
for _ in range(120):
    t, w = phrase(), rng.randint(5, 30)
    check("truncate", (t, w), m.truncate(t, w), ref_trunc(t, w))
'''

TASKS = {
    "inventory": ("inventory.py", INVENTORY, INVENTORY_CASES),
    "schedule": ("schedule.py", SCHEDULE, SCHEDULE_CASES),
    "textutil": ("textutil.py", TEXTUTIL, TEXTUTIL_CASES),
}


def make_fixture(root, task="inventory"):
    """Write task's buggy module and its verbose suite into root."""
    module, source, cases = TASKS[task]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, module), "w") as f:
        f.write(source)
    with open(os.path.join(root, "run_tests.py"), "w") as f:
        f.write(RUNNER_HEAD.replace("MODULE", module[:-3]) + cases + RUNNER_TAIL)
    return root


def new_workspace(base=None, task="inventory"):
    return make_fixture(tempfile.mkdtemp(prefix=f"solpi-{task}-", dir=base), task)
