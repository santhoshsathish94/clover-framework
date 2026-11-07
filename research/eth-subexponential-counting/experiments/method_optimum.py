"""Best case for the paper's engine with tuned parameters: at what n can it beat exhaustive search at all?

The explicit parameters in the paper are chosen for proofs (factors such as 128). Here we keep only the
construction's own requirements (Sections 2-4, Lemma incidence-parameters / incidence-labels, L1220-1400,
L2747-2845) and choose r, Q, R freely to minimise the exponent per variable, as the most favourable reading:

  interval      I = R consecutive integers centred at Q p inside {0..Q},  p = (r - 1) / (2r),  1 <= R <= Q + 1
  measurements  tail = Pr[Bin(Q, p) not in I];  tau = 3 r tail (smallest allowed, L1355-1360);  need tau <= 1/2
  dimension     D = r^(2Q);  balance 2(b + w) D^2 <= M <= 3n  (b = w = 3)  and  2 Q r^2 <= n
  costs / n     taylor  x + H2(x) with x = 3(R - 1)/Q (log2 3 once x >= 2/3)
                meas    2 tau log2(2D + 1)
                group   2 / r
Everything polynomial is ignored (favours the engine). Exhaustive search is 1 per variable.

    python method_optimum.py [--out results/method_optimum.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def best_for(r, Q):
    """Minimum exponent per variable over R for fixed r, Q; returns (total, R, parts) or None."""
    p = (r - 1) / (2 * r)
    k = np.arange(Q + 1)
    lf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, Q + 1)))))
    pmf = np.exp(lf[Q] - lf - lf[::-1] + k * math.log(p) + (Q - k) * math.log1p(-p))
    pref = np.concatenate(([0.0], np.cumsum(pmf)))
    log2D1 = 2 * Q * math.log2(r) + 1  # log2(2D + 1) <= log2 D + 1
    R = np.arange(1, Q + 2)
    a0 = np.clip(np.floor(Q * p - (R - 1) / 2).astype(np.int64), 0, Q + 1 - R)
    tail = np.maximum(0.0, 1.0 - (pref[a0 + R] - pref[a0]))
    tau = 3 * r * tail
    x = 3 * (R - 1) / Q
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where((x > 0) & (x < 1), -x * np.log2(x) - (1 - x) * np.log2(1 - x), 0.0)
    tay = np.where(x >= 2 / 3, math.log2(3), x + h)
    tot = np.where(tau <= 0.5, tay + 2 * tau * log2D1 + 2 / r, np.inf)
    i = int(np.argmin(tot))
    if not np.isfinite(tot[i]):
        return None
    return float(tot[i]), int(R[i]), dict(taylor=float(tay[i]), measurements=float(2 * tau[i] * log2D1), group=2 / r)


def best_at(log2_n, dim="balance"):
    """Minimum over r, Q allowed at n = 2^log2_n. dim="free": no balance requirement, only D^2 <= 2^(n/2)."""
    best = None
    for r in range(2, 65):
        if math.log2(2 * r * r) > log2_n:
            break
        if dim == "free":
            qmax = int(min(2.0 ** log2_n, 1.6e7) / (8 * math.log2(r)))  # 4Q log2 r <= n/2, capped for run time
        else:
            qmax = int((log2_n - 2) / (4 * math.log2(r)))  # 12 D^2 <= M = 3n at most, D = r^(2Q)
        qmax = min(qmax, int(2 ** (log2_n - 1) / (r * r)) if log2_n < 60 else qmax)  # 2 Q r^2 <= n
        if qmax < 1:
            continue
        qs = sorted({qmax, max(1, qmax // 2), max(1, qmax * 3 // 4), max(1, qmax - 1)})
        for Q in qs:
            b = best_for(r, Q)
            if b and (best is None or b[0] < best[0]):
                best = (b[0], r, Q, b[1], b[2])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "method_optimum.json"))
    ap.add_argument("--dim", choices=("balance", "free"), default="balance")
    args = ap.parse_args()
    if args.dim == "free":
        return main_free(args)
    rows, crossing = [], None
    grid = [10, 20, 30, 40, 64, 100, 200, 400, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
    for L in grid:
        b = best_at(L)
        if b is None:
            rows.append(dict(log2_n=L, feasible=False))
            print(f"log2 n = {L:>7,}: no feasible design")
            continue
        tot, r, Q, R, parts = b
        rows.append(dict(log2_n=L, total=round(tot, 4), r=r, Q=Q, R=R, **{k: round(v, 4) for k, v in parts.items()}))
        print(f"log2 n = {L:>7,}: best exponent/n = {tot:.4f}  (r={r}, Q={Q}, R={R}; taylor {parts['taylor']:.3f}, "
              f"meas {parts['measurements']:.3f}, group {parts['group']:.3f})")
        if crossing is None and tot < 1:
            crossing = L
    lo_hi = None
    if crossing:
        i = grid.index(crossing)
        lo, hi = (grid[i - 1] if i else 1), crossing
        while hi - lo > 1:
            mid = (lo + hi) // 2
            b = best_at(mid)
            if b and b[0] < 1:
                hi = mid
            else:
                lo = mid
        lo_hi = hi
        print(f"smallest log2 n where some design beats exhaustive search: {hi:,} (n ~ 10^{hi * math.log10(2):,.0f})")
    else:
        print("no design beats exhaustive search on this grid")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(model=__doc__.strip().splitlines()[0], rows=rows,
                                              smallest_log2_n_below_brute_force=lo_hi), indent=2), encoding="utf-8")


def main_free(args):
    """Sensitivity: no balance requirement; the exponent then depends on n only through Q <= n / (8 log2 r)."""
    rows, lo, hi = [], 16, None
    for n in (64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216):
        b = best_at(math.log2(n), dim="free")
        tot, r, Q, R, parts = b
        rows.append(dict(n=n, total=round(tot, 4), r=r, Q=Q, R=R, **{k: round(v, 4) for k, v in parts.items()}))
        print(f"n = {n:>9,}: best exponent/n = {tot:.4f}  (r={r}, Q={Q}, R={R}, log2 D = {2 * Q * math.log2(r):,.0f}; "
              f"taylor {parts['taylor']:.3f}, meas {parts['measurements']:.3f}, group {parts['group']:.3f})")
        if hi is None and tot < 1:
            hi = n
        elif hi is None:
            lo = n
    if hi:
        while hi - lo > max(1, lo // 200):
            mid = (lo + hi) // 2
            if best_at(math.log2(mid), dim="free")[0] < 1:
                hi = mid
            else:
                lo = mid
        print(f"smallest n where some design beats exhaustive search without the balance requirement: about {hi:,}")
    out = Path(args.out).with_name("method_optimum_free.json")
    out.write_text(json.dumps(dict(model="no balance requirement; D^2 <= 2^(n/2)", rows=rows,
                                   smallest_n_below_brute_force=hi), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
