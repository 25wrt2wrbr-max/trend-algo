#!/usr/bin/env python3
"""Round 3: (1) vol-normalized TSMOM, (2) efficiency-conditioned BEAR,
(3) final candidate vs baseline."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_prices import load

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WIN, FWD = 63, 21
OOS = 0.20


def daily_vol(c):
    n = len(c)
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, n) if c[i - 1] > 0]
    if len(rets) < 10:
        return float("nan")
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)


def volscore(c):
    """TSMOM normalized by realized vol: ret / (daily_vol * sqrt(63)).
    Saturate at +-3 window-sigmas. (Barroso & Santa-Clara style)"""
    if c[0] <= 0:
        return float("nan")
    r = c[-1] / c[0] - 1.0
    v = daily_vol(c)
    if math.isnan(v) or v <= 0:
        return float("nan")
    z = r / (v * math.sqrt(len(c) - 1))
    return max(-1.0, min(1.0, z / 3.0))


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


def range_pos(c):
    lo, hi = min(c), max(c)
    if hi <= lo:
        return 0.0
    return ((c[-1] - lo) / (hi - lo) - 0.5) * 2.0


def rawscore(c):
    r = c[-1] / c[0] - 1.0
    return max(-1.0, min(1.0, r / 0.10))


VOLS = {"vtsmom": volscore, "regr2": regression, "effr": efficiency, "rng": range_pos}
WV = {"vtsmom": .40, "regr2": .25, "effr": .20, "rng": .15}


def score(c, inds, w):
    s = t = 0.0
    for k in inds:
        v = inds[k](c)
        if not math.isnan(v):
            s += w[k] * v
            t += w[k]
    return s / t if t else float("nan")


def load_all():
    out = {}
    for f in os.listdir(DATA):
        if f.endswith(".json"):
            out[f[:-5].upper()] = load(os.path.join(DATA, f))
    return out


def rows_for(assets, inds, w, th):
    rows = []
    for name, (dates, closes) in assets.items():
        n = len(closes)
        cut = int(n * (1 - OOS))
        for i in range(WIN - 1, n - FWD):
            c = closes[i - WIN + 1:i + 1]
            sc = score(c, inds, w)
            if math.isnan(sc):
                continue
            fwd = closes[i + FWD] / closes[i] - 1.0
            er = efficiency(c)
            lab = "BULL" if sc >= th else ("BEAR" if sc <= -th else "NEUT")
            rows.append((name, "fit" if i < cut else "test", sc, fwd, lab, er))
    return rows


def summ(rows, tag):
    if not rows:
        return f"{tag}: n=0"
    m = sum(r[3] for r in rows) / len(rows)
    def hit(r):
        return sum(1 for x in r if (x[4] == "BEAR" and x[3] < 0) or x[3] > 0) / len(r)
    return f"{tag}: n={len(rows):5d} acc={hit(rows):.2f} fwd={m:+.4f}"


def main():
    assets = load_all()

    print("== final vol-normalized ensemble (vtsmom .40 regr2 .25 effr .20 rng .15)")
    for th in (0.20, 0.25, 0.30):
        rows = rows_for(assets, VOLS, WV, th)
        for scope in ("fit", "test", None):
            sub = rows if scope is None else [r for r in rows if r[1] == scope]
            print(f" th={th} {scope or 'ALL ':4s} {summ(sub, '')}")

    print("\n== efficiency-conditioned BEAR (th=0.25)")
    rows = rows_for(assets, VOLS, WV, 0.25)
    for scope in ("fit", "test"):
        sub = [r for r in rows if r[1] == scope]
        bear = [r for r in sub if r[4] == "BEAR"]
        orderly = [r for r in bear if abs(r[5]) >= 0.35]
        choppy = [r for r in bear if abs(r[5]) < 0.35]
        print(f" {scope}: all-bear {summ(bear,'')}")
        print(f"        orderly(|er|>=.35) {summ(orderly,'')}")
        print(f"        choppy(|er|<.35)  {summ(choppy,'')}")

    print("\n== per-asset BEAR forward returns, split by |ER| (all periods, th=0.25)")
    rows_all = rows_for(assets, VOLS, WV, 0.25)
    for name in sorted(assets):
        bear = [r for r in rows_all if r[0] == name and r[4] == "BEAR"]
        o = [r for r in bear if abs(r[5]) >= 0.35]
        cp = [r for r in bear if abs(r[5]) < 0.35]
        print(f" {name:5s} all {summ(bear,'')}")
        print(f"        orderly {summ(o,'')}  choppy {summ(cp,'')}")

    print("\n== BULL check with same split (th=0.25)")
    for scope in ("fit", "test"):
        sub = [r for r in rows_all if r[1] == scope]
        bull = [r for r in sub if r[4] == "BULL"]
        o = [r for r in bull if abs(r[5]) >= 0.35]
        cp = [r for r in bull if abs(r[5]) < 0.35]
        print(f" {scope}: all-bull {summ(bull,'')}")
        print(f"        orderly {summ(o,'')}  choppy {summ(cp,'')}")


if __name__ == "__main__":
    main()
