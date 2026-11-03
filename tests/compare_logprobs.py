#!/usr/bin/env python3
"""Compare two `ds4 --dump-logprobs` files step by step.

    python3 tests/compare_logprobs.py A.json B.json
    python3 tests/compare_logprobs.py --text A.json    # print the continuation
    python3 tests/compare_logprobs.py --stats A.json   # eos step and distinct-2

Reports how many greedy steps agree on the selected token, and the largest
logprob difference over the tokens both runs list in their top-k.  Exit code
0 when the files are identical, 1 otherwise (useful for bit-exact checks).
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)["steps"]


def tid(tok):
    """Tokens are {"id", "text", "bytes"} objects in ds4 dumps."""
    return tok["id"] if isinstance(tok, dict) else tok


def text(path):
    """Greedy continuation of a dump, decoded from the token bytes."""
    raw = b"".join(bytes(s["selected"]["bytes"]) for s in load(path))
    return raw.decode("utf-8", errors="replace")


def stats(path):
    """Free-generation health: the step that ends the answer (-1: never) and
    the fraction of distinct token bigrams (low means looping text)."""
    steps = load(path)
    ids = [tid(s["selected"]) for s in steps]
    eos = -1
    for i, s in enumerate(steps):
        if "end" in s["selected"].get("text", "") and "sentence" in s["selected"].get("text", ""):
            eos = i
            break
    body = ids[:eos] if eos >= 0 else ids
    pairs = list(zip(body, body[1:]))
    distinct2 = len(set(pairs)) / len(pairs) if pairs else 1.0
    return eos, distinct2


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--text":
        print(text(sys.argv[2]))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--stats":
        eos, d2 = stats(sys.argv[2])
        print(f"{eos} {d2:.3f}")
        return 0
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    same_prefix = 0
    max_d = 0.0
    sum_d = 0.0
    n_d = 0
    for sa, sb in zip(a, b):
        if tid(sa["selected"]) != tid(sb["selected"]):
            break
        same_prefix += 1
        lb = {tid(t["token"]): t["logprob"] for t in sb["top_logprobs"]}
        for t in sa["top_logprobs"][:5]:
            if tid(t["token"]) in lb:
                d = abs(t["logprob"] - lb[tid(t["token"])])
                max_d = max(max_d, d)
                sum_d += d
                n_d += 1
    identical = open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
    print(f"{sys.argv[1]} vs {sys.argv[2]}: "
          f"{'IDENTICAL' if identical else 'different'}, "
          f"same greedy tokens for {same_prefix}/{min(len(a), len(b))} steps, "
          f"top-5 |dlogprob| max {max_d:.4g} mean {sum_d / max(n_d, 1):.4g}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
