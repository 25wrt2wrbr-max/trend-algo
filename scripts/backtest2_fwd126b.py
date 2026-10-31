#!/usr/bin/env python3
"""Stricter evaluation: Spearman score~fwd, label accuracy, per-asset,
weight variants, reversed split."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_prices import load

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WIN, FWD = 63, 63
OOS = 0.20


def ret_sign(c):
    r = c[-1] / c[0] - 1.0
    return max(-1.0, min(1.0, r / 0.10))


def regression(c):
    n = len(c)
    mx, my = (n - 1) / 2, sum(c) / n
    sxx = sxy = syy = 0.0
    for x, y in zip(range(n), c):
        sxx += (x - mx) ** 2
        sxy += (x - mx) * (y - my)
        syy += (y - my) ** 2
    if sxx <= 0 or syy <= 0:
        return float("nan")
    r2 = sxy * sxy / (sxx * syy)
    z = (sxy / sxx) * (n - 1) / my / 0.30
    return max(-1.0, min(1.0, z)) * r2


def efficiency(c):
    net = c[-1] - c[0]
    path = sum(abs(c[i] - c[i - 1]) for i in range(1, len(c)))
    return net / path if path > 0 else float("nan")


def sma_pos(c):
    n = len(c)
    sma = sum(c) / n
    half = n // 2
    sma_old = sum(c[:half]) / half
    d = max(-1.0, min(1.0, (c[-1] / sma - 1.0) / 0.05))
    s = max(-1.0, min(1.0, (sma / sma_old - 1.0) / 0.05))
    return d * s


def range_pos(c):
    """Close position within window [low, high], signed to +-1."""
    lo, hi, cl = min(c), max(c), c[-1]
    if hi <= lo:
        return 0.0
    p = (cl - lo) / (hi - lo)          # 0..1
    return (p - 0.5) * 2.0             # -1..1


IND = {"tsmom": ret_sign, "regr2": regression, "effr": efficiency,
       "sma": sma_pos, "rng": range_pos}
WEIGHTS = {
    "A (final candidate: tsmom .40 regr2 .25 effr .20 rng .15)":
        {"tsmom": .40, "regr2": .25, "effr": .20, "rng": .15},
    "B (equal 5)": {"tsmom": .2, "regr2": .2, "effr": .2, "sma": .2, "rng": .2},
    "C (tsmom only)": {"tsmom": 1.0},
    "D (tsmom .5 effr .5)": {"tsmom": .5, "effr": .5},
    "E (no rng, prev 4)": {"tsmom": .35, "regr2": .30, "effr": .15, "sma": .20},
}


def score_with(w, c):
    s = t = 0.0
    for k, wt in w.items():
        v = IND[k](c)
        if not math.isnan(v):
            s += wt * v
            t += wt
    return s / t if t else float("nan")


def spearman(xs, ys):
    n = len(xs)
    if n < 10:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def build(assets, w, th):
    """yield (asset, scope, score, fwd, label)"""
    for name, (dates, closes) in assets.items():
        n = len(closes)
        cut = int(n * (1 - OOS))
        for i in range(WIN - 1, n - FWD):
            sc = score_with(w, closes[i - WIN + 1:i + 1])
            if math.isnan(sc):
                continue
            fwd = closes[i + FWD] / closes[i] - 1.0
            lab = "BULL" if sc >= th else ("BEAR" if sc <= -th else "NEUT")
            scope = "fit" if i < cut else "test"
            yield name, scope, sc, fwd, lab


def summarize(rows):
    xs = [r[2] for r in rows]
    ys = [r[3] for r in rows]
    rho = spearman(xs, ys)
    acc = {"BULL": [0, 0], "BEAR": [0, 0], "NEUT": [0, 0]}
    means = {"BULL": [], "BEAR": [], "NEUT": []}
    for _, _, sc, fwd, lab in rows:
        acc[lab][1] += 1
        means[lab].append(fwd)
        if lab == "BULL" and fwd > 0: acc["BULL"][0] += 1
        if lab == "BEAR" and fwd < 0: acc["BEAR"][0] += 1
        if lab == "NEUT" and abs(fwd) < 0.03: acc["NEUT"][0] += 1
    out = [f"rho={rho:+.3f}"]
    for lab in ("BULL", "BEAR", "NEUT"):
        hit, tot = acc[lab]
        m = sum(means[lab]) / len(means[lab]) if means[lab] else float("nan")
        out.append(f"{lab}: n={tot:5d} acc={hit/max(tot,1):.2f} fwd={m:+.4f}")
    return "  ".join(out)


def main():
    assets = {}
    for f in os.listdir(DATA):
        if f.endswith(".json"):
            assets[f[:-5].upper()] = load(os.path.join(DATA, f))

    for wname, w in WEIGHTS.items():
        for th in (0.15, 0.25):
            rows_fit, rows_test, rows_all = [], [], []
            per_asset = {}
            for name, scope, sc, fwd, lab in build(assets, w, th):
                rows_all.append((name, scope, sc, fwd, lab))
                (rows_fit if scope == "fit" else rows_test).append(
                    (name, scope, sc, fwd, lab))
                per_asset.setdefault(name, []).append((name, scope, sc, fwd, lab))
            print(f"\n== {wname}  th={th}")
            print(f"  ALL : {summarize(rows_all)}")
            print(f"  fit : {summarize(rows_fit)}")
            print(f"  test: {summarize(rows_test)}")
            if th == 0.25:
                for name in sorted(per_asset):
                    print(f"    {name:5s}: {summarize(per_asset[name])}")


if __name__ == "__main__":
    main()
