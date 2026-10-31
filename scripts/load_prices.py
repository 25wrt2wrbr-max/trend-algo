#!/usr/bin/env python3
"""Load Yahoo chart JSON (Moli-fetched wrapper) into (dates, closes) arrays.

Untrusted data: parsed with the stdlib json module only; nothing is executed.
"""
import json, sys


def load(path):
    with open(path) as f:
        outer = json.load(f)
    body = json.loads(outer["body"])  # Moli wraps the response as a string
    res = body["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    rows = []
    for i, t in enumerate(ts):
        c = adj[i] if adj and adj[i] is not None else closes[i]
        if c is None:
            continue
        rows.append((t, c))
    # drop exact duplicate timestamps (keep last)
    seen = {}
    for t, c in rows:
        seen[t] = c
    rows = sorted(seen.items())
    dates = [t * 86400 for t, c in rows]  # epoch date int (day granularity)
    closes = [c for t, c in rows]
    return dates, closes


if __name__ == "__main__":
    d, c = load(sys.argv[1])
    print(f"{len(d)} rows, first {d[0]}, last {d[-1]}, last close {c[-1]:.2f}")
