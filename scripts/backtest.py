#!/usr/bin/env python3
"""Backtest trend-classification signals on 3-month (63 trading day) windows.

Question tested: if you classify a 63-day window as BULLISH / BEARISH / NEUTRAL
today, what is the *forward* return over the next 21 trading days (1 month)?
A good classifier should show mean fwd ret(BULL) > mean fwd ret(NEUTRAL) >
mean fwd ret(BEAR), and up-day hit rates to match.

Split: first 80% of each asset's timeline = "fit" (threshold selection),
last 20% = "test" (reported separately). Pooled across assets.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_prices import load

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WIN = 63          # ~3 months of trading days
FWD = 21          # forward horizon to judge the label by: 1 month
OOS_FRAC = 0.20   # last 20% of each asset held out


# ---------- indicators, each returns score in [-1, +1] (NaN if not computable) ----------

def ret_sign(closes):
    """TSMOM: sign of the window return (Moskowitz/Ooi/Pedersen)."""
    c0, c1 = closes[0], closes[-1]
    if c0 <= 0:
        return float("nan")
    r = c1 / c0 - 1.0
    return max(-1.0, min(1.0, r / 0.10))  # saturate at +-10%


def regression(closes):
    """Slope*z * R^2: direction weighted by linear fit quality."""
    n = len(closes)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(closes) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, closes))
    syy = sum((y - my) ** 2 for y in closes)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy)
    # normalize slope by mean price and window length: +-1 at +-30% per window
    z = (slope * (n - 1)) / my / 0.30
    z = max(-1.0, min(1.0, z))
    return z * r2


def efficiency(closes):
    """Kaufman Efficiency Ratio: net change / sum |daily changes|, signed."""
    net = closes[-1] - closes[0]
    path = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if path <= 0:
        return float("nan")
    return max(-1.0, min(1.0, net / path))


def sma_pos(closes):
    """Price vs SMA(distance) * SMA slope, each normalized."""
    n = len(closes)
    sma = sum(closes) / n
    if sma <= 0:
        return float("nan")
    d = (closes[-1] / sma - 1.0) / 0.05          # +-1 at +-5% vs SMA
    half = n // 2
    sma_old = sum(closes[:half]) / half
    s = (sma / sma_old - 1.0) / 0.05             # SMA slope over window half
    s = max(-1.0, min(1.0, s))
    d = max(-1.0, min(1.0, d))
    return d * s


IND = {
    "tsmom": ret_sign,
    "regr2": regression,
    "effr": efficiency,
    "sma": sma_pos,
}
# ensemble weights (fit phase will sanity-check these against alternatives)
W = {"tsmom": 0.35, "regr2": 0.30, "effr": 0.15, "sma": 0.20}


def ensemble(closes):
    s = 0.0
    tot = 0.0
    for k, fn in IND.items():
        v = fn(closes)
        if not math.isnan(v):
            s += W[k] * v
            tot += W[k]
    return s / tot if tot > 0 else float("nan")


def classify(score, th):
    if math.isnan(score):
        return None
    if score >= th:
        return "BULL"
    if score <= -th:
        return "BEAR"
    return "NEUT"


def load_all():
    out = {}
    for f in os.listdir(DATA):
        if f.endswith(".json"):
            name = f[:-5].upper()
            out[name] = load(os.path.join(DATA, f))
    return out


def evaluate(labels_by_asset, scope):
    """scope: 'fit' or 'test' -> (mean_fwd_bull, mean_fwd_bear, mean_fwd_neut,
    hit_bull, hit_bear, n_bull, n_bear, n_neut, fwd_neutral_mean)"""
    acc = {"BULL": [], "BEAR": [], "NEUT": []}
    for name, (dates, closes, labels) in labels_by_asset.items():
        n = len(closes)
        cut = int(n * (1 - OOS_FRAC))
        lo, hi = (0, cut) if scope == "fit" else (cut, n)
        for i in range(lo, hi):
            lab = labels[i]
            if lab is None or i + FWD >= n:
                continue
            fwd = closes[i + FWD] / closes[i] - 1.0
            acc[lab].append(fwd)
    def stats(xs):
        if not xs:
            return (float("nan"),) * 3 + 0,
        m = sum(xs) / len(xs)
        hit = sum(1 for x in xs if x > 0) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        t = m / math.sqrt(var / len(xs)) if var > 0 and len(xs) > 2 else float("nan")
        return m, hit, t, len(xs)
    b = stats(acc["BULL"]); r = stats(acc["BEAR"]); n = stats(acc["NEUT"])
    return b, r, n


def run(score_fn, th, assets):
    labels_by = {}
    for name, (dates, closes) in assets.items():
        labels = [None] * len(closes)
        for i in range(WIN - 1, len(closes)):
            labels[i] = classify(score_fn(closes[i - WIN + 1:i + 1]), th)
        labels_by[name] = (dates, closes, labels)
    fit = evaluate(labels_by, "fit")
    test = evaluate(labels_by, "test")
    return fit, test


def show(tag, th, fit, test):
    def line(scope, st):
        b, r, n = st
        print(f"  {scope:4s} BULL m={b[0]:+.4f} hit={b[1]:.2f} t={b[2]:+.1f} n={b[3]:4d} | "
              f"BEAR m={r[0]:+.4f} hit={r[1]:.2f} t={r[2]:+.1f} n={r[3]:4d} | "
              f"NEUT m={n[0]:+.4f} hit={n[1]:.2f} n={n[3]:4d}")
    print(f"[{tag}] th={th}")
    line("fit", fit)
    line("test", test)


def main():
    assets = load_all()
    print(f"assets: {sorted(assets)}, window={WIN}d fwd={FWD}d oos={OOS_FRAC:.0%}")
    for name, (d, c) in sorted(assets.items()):
        print(f"  {name:8s} {len(c):5d} rows")

    # 1) individual indicators at a fixed threshold
    for key, fn in IND.items():
        fit, test = run(fn, 0.15, assets)
        show(f"ind:{key}", 0.15, fit, test)
    fit, test = run(ensemble, 0.15, assets)
    show("ensemble", 0.15, fit, test)

    # 2) threshold sweep for the ensemble
    print("\nensemble threshold sweep:")
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        fit, test = run(ensemble, th, assets)
        show(f"ens", th, fit, test)


if __name__ == "__main__":
    main()
