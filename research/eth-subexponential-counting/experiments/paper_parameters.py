"""Where does the paper's subexponential engine actually start?

Runs the explicit parameter construction and numerical checks of OpenAI, "The Exponential Time Hypothesis is False"
(openai/math @ 4eaee87, main_full_version.tex): eq. (overview-explicit-parameters) L1363-1373, Lemma
incidence-parameters L2747-2845, and the guards at L3152 and L3185-3197. When any check fails the paper "checks the
original equations directly on all 2^n assignments" (exhaustive search). The checks depend on n only through
lambda = ceil(log2 n), so we scan lambda, not n.

For the bounded-occurrence 3-CNF engine (Sections 2-4): width w = 3, at most b = 3 occurrences per variable,
beta = 1, R = (beta + 3) * ceil(sqrt(lambda)). Costs (eq. overview-three-costs, L973-978), up to polynomial factors:
    Taylor data  C_q = sum_{j <= min(q, n)} 2^j C(n, j),   q = floor(3 n R / Q)
    measurements (2D + 1)^(2 tau n)
    one group    2^(2n / r)
For astronomically large n only the exponent per variable matters:
    log2 C_q / n -> x + H2(x) for x = q/n < 2/3 (else log2 3);  2 tau log2(2D + 1);  2 / r.
Brute force is 1 per variable.

    python paper_parameters.py [--out results/paper_parameters.json]
"""
import argparse
import json
import math
from pathlib import Path

B_OCC, WIDTH = 3, 3


def ceil_log2(x):
    return (x - 1).bit_length()  # least l with x <= 2**l, for x >= 1


def isqrt_ceil(x):
    s = math.isqrt(x)
    return s if s * s == x else s + 1


def params(lam):
    nu = ceil_log2(lam)
    beta = 1
    while B_OCC > lam ** beta or WIDTH > lam ** beta:
        beta += 1
    r = math.isqrt(lam // nu ** 4)  # largest r with r^2 nu^4 <= lambda
    Q = lam // (128 * nu)
    R = (beta + 3) * isqrt_ceil(lam)
    return nu, beta, r, Q, R


def checks(lam):
    """The paper's guards and checks for the n with ceil(log2 n) = lam; returns (passed, failed check names, params)."""
    nu, beta, r, Q, R = params(lam)
    p = dict(lam=lam, nu=nu, beta=beta, r=r, Q=Q, R=R)
    if r < 2 or Q <= R:
        return False, ["r>=2" if r < 2 else "Q>R"], p
    failed = []
    if R < 32:
        failed.append("R>=32")
    if 6 * R > Q:
        failed.append("6R<=Q")
    mu = Q * (r - 1) / (2 * r)
    a0 = math.floor(mu - (R - 1) / 2)
    if a0 < 0 or a0 + R - 1 > Q:
        failed.append("interval")
    log2_D = 2 * Q * math.log2(r)
    log2_n = lam - 1  # least favourable n with this lambda (and M = n)
    if math.log2(2 * (B_OCC + WIDTH)) + 2 * log2_D > log2_n:  # 2(b+w)D^2 <= M
        failed.append("2(b+w)D^2<=M")
    if math.log2(2 * Q * r * r) > log2_n:
        failed.append("2Qr^2<=n")
    # tail Pr[W not in I] <= 1/(b r^2 lambda), via the paper's Hoeffding bound 2exp(-R^2/(8Q))
    if math.log(2) - R * R / (8 * Q) > -math.log(B_OCC * r * r * lam):
        failed.append("tail")
    p.update(log2_D=log2_D, tau=1 / (r * lam))
    return not failed, failed, p


def exponent_per_variable(p):
    """Exponent of the three costs divided by n, for n -> infinity at these parameters (brute force = 1)."""
    x = 3 * p["R"] / p["Q"]
    if x >= 2 / 3:
        taylor = math.log2(3)
    else:
        taylor = x - x * math.log2(x) - (1 - x) * math.log2(1 - x)
    meas = 2 * p["tau"] * (p["log2_D"] + 1)  # log2(2D+1) <= log2 D + 1
    group = 2 / p["r"]
    return dict(taylor=taylor, measurements=meas, one_group=group, total=taylor + meas + group, x_q_over_n=x)


def runs_and_below(target):
    def pred(lam):
        ok, _, p = checks(lam)
        return ok and (target is None or exponent_per_variable(p)["total"] < target)
    return pred


def first_lambda(pred, nu_max=64):
    """Smallest lambda with pred: per nu-block bisection (the parameters move monotonically inside a block),
    confirmed on the 20,000 lambdas just below the hit and on a sample across the block."""
    for nu in range(1, nu_max + 1):
        lo, hi = 2 ** (nu - 1) + 1, 2 ** nu
        if not pred(hi):
            continue
        a, b = lo, hi
        while a < b:
            m = (a + b) // 2
            if pred(m):
                b = m
            else:
                a = m + 1
        for lam in range(max(lo, a - 20_000), a):
            assert not pred(lam), f"non-monotone at lambda {lam} < {a}"
        for i in range(1, 1000):
            lam = lo + (a - lo) * i // 1000
            assert lam >= a or not pred(lam), f"non-monotone at lambda {lam} < {a}"
        return a
    return None


def row(lam):
    ok, _, p = checks(lam)
    e = exponent_per_variable(p)
    return dict(lam=lam, log10_n=round((lam - 1) * math.log10(2), 1), r=p["r"], Q=p["Q"], R=p["R"],
                q_over_n=round(e["x_q_over_n"], 4), taylor=round(e["taylor"], 4), measurements=round(e["measurements"], 6),
                one_group=round(e["one_group"], 4), total=round(e["total"], 4), checks_pass=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "paper_parameters.json"))
    args = ap.parse_args()
    first_fail = {}
    for lam in range(5, 1_000_001):  # every n up to 2^1,000,000: which check fails first
        ok, failed, _ = checks(lam)
        assert not ok, f"engine runs already at lambda={lam}"
        first_fail[failed[0]] = first_fail.get(failed[0], 0) + 1
    lam_run = first_lambda(runs_and_below(None))
    lam_faster = first_lambda(runs_and_below(1.0))
    lam_half = first_lambda(runs_and_below(0.5))
    lam_tenth = first_lambda(runs_and_below(0.1))
    marks = [x for x in (lam_run, lam_faster, lam_half, lam_tenth) if x]
    rows = [row(lam) for lam in sorted(set(marks + [2 ** k for k in range(34, 61, 2)]))]
    out = dict(source="openai/math@4eaee87 main_full_version.tex L1363-1373, L2747-2845, L3152, L3185-3197",
               engine="bounded-occurrence 3-CNF (b=3, w=3, beta=1); exponents are per variable, brute force = 1",
               lambdas_scanned_exhaustively=[5, 1_000_000], first_failing_check_counts=first_fail,
               first_lambda_engine_runs=lam_run, first_lambda_below_brute_force=lam_faster,
               first_lambda_below_half=lam_half, first_lambda_below_tenth=lam_tenth, table=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"lambda 5..1,000,000 (n up to 2^1,000,000): every one fails a check -> exhaustive search; first failure: {first_fail}")
    for lab, lam in (("engine runs", lam_run), ("exponent < brute force", lam_faster),
                     ("exponent < 0.5 n", lam_half), ("exponent < 0.1 n", lam_tenth)):
        print(f"first lambda where {lab}: " + (f"{lam:,}  (n ~ 10^{(lam - 1) * math.log10(2):,.0f})" if lam else "none up to 2^64"))
    print("lambda | log10 n | r | Q | R | q/n | taylor | measurements | one group | total | checks")
    for x in rows:
        print(f"{x['lam']:,} | {x['log10_n']:,} | {x['r']} | {x['Q']:,} | {x['R']:,} | {x['q_over_n']} | {x['taylor']} | "
              f"{x['measurements']} | {x['one_group']} | {x['total']} | {x['checks_pass']}")


if __name__ == "__main__":
    main()
