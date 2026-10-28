"""
Driver for E2, E3, and E4 on Newell & Simon (1956) Logic Theory Machine:
  - E2: Parses & verifies the full 67-theorem Principia Mathematica *2 campaign in `dmoews/logic-theorist`
        (via `run_logic.test` / `analyze_output.py`) and compares with our `LogicTheorist1956` implementation.
  - E3.1: Similarity filter `CSm` ON vs OFF across all 67 theorems (measuring `csm_passed / csm_calls`
          vs Newell & Simon's "about one theorem in five" on p. 31 Footnote 12, `LMc` calls, and theorems proved).
  - E3.2: Contraction `VCt` ON vs OFF on `*2.06` (`(p -> q) -> ((q -> r) -> (p -> r))`) and across all 67 theorems.
  - E4.1: Section II "Learning Processes" (p. 48): All-prior theorems vs Proved-only theorems vs Axioms-only (`*1.2..*1.6`).
  - E4.2: Subproblem Priority Ordering (`Ex(CK)`, p. 46): Minimal-K (`min_k`) vs FIFO (`fifo`) vs Maximal-K (`max_k`).
"""
import subprocess
from typing import List, Tuple
from lt1956_engine import Expr, parse_pm_expr, view_variables_vv, view_contracted_vct, describe_d
from lt1956_methods import LogicTheorist1956


def load_pm_axioms_and_theorems(repo_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    def read_pairs(path: str) -> List[Tuple[str, str]]:
        pairs = []
        curr_id = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "*" in line:
                        curr_id = line.lstrip("# ").split(",")[0].strip()
                else:
                    assert curr_id is not None
                    pairs.append((curr_id, line))
                    curr_id = None
        return pairs

    axioms = read_pairs(f"{repo_dir}/pm-axioms.txt")
    theorems = read_pairs(f"{repo_dir}/pm-theorems.txt")
    return axioms, theorems


def run_e2_dmoews_campaign(repo_dir: str):
    print("=== E2: dmoews/logic-theorist Full Principia Mathematica *2 Campaign (67 theorems) ===")
    import os
    raw_path = f"{repo_dir}/pm_run_raw.out"
    ana_path = f"{repo_dir}/pm_analyzed.out"
    if not os.path.exists(ana_path):
        with open(raw_path, "w", encoding="utf-8") as rf:
            subprocess.run(["python3", "run_logic.py"], cwd=repo_dir, stdout=rf, check=True)
        with open(raw_path, "r", encoding="utf-8") as rf, open(ana_path, "w", encoding="utf-8") as af:
            subprocess.run(["python3", "analyze_output.py"], cwd=repo_dir, stdin=rf, stdout=af, check=True)

    analyzed_stdout = open(ana_path, "r", encoding="utf-8").read()
    lines = analyzed_stdout.splitlines()

    proved_total = 0
    proved_first52 = 0
    by_msb_only = 0
    by_mdt = 0
    by_mch = 0
    multi_subprob = 0

    # Parse problems in order from analyzed_stdout: each problem starts with 'Proven: ' or 'Not proven: '
    prob_idx = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Proven: ") or line.startswith("Not proven: "):
            prob_idx += 1
            is_proven = line.startswith("Proven: ")
            proof_steps = []
            j = i + 1
            while j < len(lines) and not (lines[j].startswith("Proven: ") or lines[j].startswith("Not proven: ") or lines[j].startswith("Result: ")):
                if lines[j].startswith("     "):
                    proof_steps.append(lines[j].strip())
                j += 1
            if prob_idx in (6, 17, 29, 31, 56):
                print(f"  Exact 1956 IPL-I result on #{prob_idx}: {line}")
                for s in proof_steps:
                    print(f"    step: {s}")
            if is_proven:
                proved_total += 1
                if prob_idx <= 52:
                    proved_first52 += 1
                if len(proof_steps) == 1 and proof_steps[0].startswith("substitution"):
                    by_msb_only += 1
                elif len(proof_steps) == 2 and proof_steps[0].startswith("detachment"):
                    by_mdt += 1
                elif len(proof_steps) in (1, 2) and ("forward-chain" in proof_steps[0] or "backward-chain" in proof_steps[0]):
                    by_mch += 1
                else:
                    multi_subprob += 1
            i = j
        else:
            i += 1

    print(f"  Summary line at end of analyze_output.py: {lines[-1]}")
    print(f"  Proved by 1956 IPL-I Logic Theorist (all 67):     {proved_total} / 67 ({proved_total/67*100:.1f}%)")
    print(f"  Proved in first 52 theorems (1957 WJCC subset):   {proved_first52} / 52 ({proved_first52/52*100:.1f}%)")
    print(f"  Breakdown of the {proved_total} proofs verified by analyze_output.py:")
    print(f"    - Substitution + Replacement only (MSb):        {by_msb_only} (Newell & Simon p. 37 states 'some 21')")
    print(f"    - 1-step Detachment + MSb (MDt):                {by_mdt}")
    print(f"    - 1-step Chaining + MSb (MCh):                  {by_mch}")
    print(f"    - Multi-step Subproblem Chaining/Detachment:    {multi_subprob}")


def run_e3_similarity_and_contraction(axioms_raw: List[Tuple[str, str]], theorems_raw: List[Tuple[str, str]]):
    print("\n=== E3.1: Similarity Filter (CSm: D=(K,J,H)) ON vs OFF across all 67 *2 Theorems ===")
    tr_free = str.maketrans({"p": "A", "q": "B", "r": "C", "s": "D"})

    for use_csm in (True, False):
        known: List[Tuple[str, Expr]] = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
        proved = 0
        proved_msb = 0
        total_csm_calls = 0
        total_csm_passed = 0
        total_lmc_calls = 0
        total_work_w = 0

        for tid, tstr in theorems_raw:
            prob = parse_pm_expr(tstr, free_vars=False)
            lt = LogicTheorist1956(known, use_csm=use_csm, use_vct=True, priority_mode="min_k", work_limit=50)
            ok, trace, _ = lt.prove_executive(prob)
            if ok:
                proved += 1
                if trace and trace.startswith("MSb("):
                    proved_msb += 1
            total_csm_calls += lt.stats["csm_calls"]
            total_csm_passed += lt.stats["csm_passed"]
            total_lmc_calls += lt.stats["lmc_top_calls"]
            total_work_w += lt.work_w
            known.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))

        pass_pct = (100.0 * total_csm_passed / total_csm_calls) if total_csm_calls else 0.0
        mode_str = "CSm ON  (1956 heuristic)" if use_csm else "CSm OFF (unfiltered LMc)"
        print(f"  {mode_str}: proved={proved}/67 (MSb={proved_msb}), CSm={total_csm_passed}/{total_csm_calls} ({pass_pct:.1f}%), LMc_calls={total_lmc_calls}, total_W={total_work_w}")

    print("\n=== E3.2: Contraction (VCt, pp. 39-42) ON vs OFF on *2.06 and across all 67 *2 Theorems ===")
    # Inspect *2.06 descriptions under VV and VCt and compare against *2.04 right side:
    t206_str = dict(theorems_raw)["*2.06"]
    t204_str = dict(theorems_raw)["*2.04"].translate(tr_free)
    e206 = parse_pm_expr(t206_str, free_vars=False)
    e204 = parse_pm_expr(t204_str, free_vars=True)

    view_variables_vv(e206)
    dl_206_vv, dr_206_vv = describe_d(e206.left), describe_d(e206.right)
    view_contracted_vct(e206)
    dl_206_ct, dr_206_ct = describe_d(e206.left), describe_d(e206.right)
    view_variables_vv(e204)
    dl_204r, dr_204r = describe_d(e204.right.left), describe_d(e204.right.right)

    print(f"  *2.06 under VV  (variables as units):   DL={dl_206_vv}, DR={dr_206_vv} (matches P-868 p. 39: (2,2,2) and (3,3,4))")
    print(f"  *2.06 under VCt (contracted binary):    DL={dl_206_ct}, DR={dr_206_ct} (matches P-868 p. 40: (1,1,1) and (2,2,2))")
    print(f"  *2.04 right side under VV:              DL={dl_204r}, DR={dr_204r} (matches P-868 p. 41: (1,1,1) and (2,2,2))")
    assert dl_206_vv == (2, 2, 2) and dr_206_vv == (3, 3, 4)
    assert dl_206_ct == (1, 1, 1) and dr_206_ct == (2, 2, 2)
    assert (dl_206_ct, dr_206_ct) == (dl_204r, dr_204r)

    # Test proving *2.06 from {*1.2..*1.6, *2.01..*2.05} with VCt ON vs VCt OFF (and MSb+MDt only, as on pp. 41-42):
    known_pre206 = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
    for tid, tstr in theorems_raw[:5]:  # *2.01 .. *2.05
        known_pre206.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))

    lt_vct_on = LogicTheorist1956(known_pre206, use_csm=True, use_vct=True, work_limit=30)
    lt_vct_off = LogicTheorist1956(known_pre206, use_csm=True, use_vct=False, work_limit=30)
    mdt_on = lt_vct_on.method_mdt(parse_pm_expr(t206_str, free_vars=False))
    mdt_off = lt_vct_off.method_mdt(parse_pm_expr(t206_str, free_vars=False))
    print(f"  Proving *2.06 by MDt with VCt ON : {mdt_on}")
    print(f"  Proving *2.06 by MDt with VCt OFF: {mdt_off} (fails MDt on p. 41 without contraction!)")
    assert mdt_on is not None and mdt_off is None

    # Full 67-theorem campaign with VCt ON vs VCt OFF (using MSb + MDt only vs full MSb + MDt + MCh):
    for label, allow_mch in [("MSb + MDt only", False), ("Full MSb + MDt + MCh", True)]:
        counts = {}
        for use_vct in (True, False):
            known = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
            cnt = 0
            for tid, tstr in theorems_raw:
                prob = parse_pm_expr(tstr, free_vars=False)
                lt = LogicTheorist1956(known, use_csm=True, use_vct=use_vct, work_limit=50)
                if not allow_mch:
                    ok = (lt.method_msb(prob) is not None) or (lt.method_mdt(prob) is not None)
                else:
                    ok, _, _ = lt.prove_executive(prob)
                if ok:
                    cnt += 1
                known.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))
            counts[use_vct] = cnt
        print(f"  67-theorem campaign ({label:<20}): VCt ON = {counts[True]}/67 vs VCt OFF = {counts[False]}/67")


def run_e4_learning_and_priority(axioms_raw: List[Tuple[str, str]], theorems_raw: List[Tuple[str, str]]):
    print("\n=== E4.1: Section II 'Learning Processes' (Theorem-Memory Accumulation, p. 48) ===")
    tr_free = str.maketrans({"p": "A", "q": "B", "r": "C", "s": "D"})

    for mem_mode, desc in [
        ("all_prior", "All prior theorems stored in T (P-868 / 1957 baseline)"),
        ("proved_only", "Only previously PROVED theorems stored in T (self-bootstrapped)"),
        ("axioms_only", "Axioms *1.2..*1.6 ONLY stored in T (no theorem learning)"),
    ]:
        known = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
        proved_all67 = 0
        proved_first52 = 0
        for idx, (tid, tstr) in enumerate(theorems_raw, start=1):
            prob = parse_pm_expr(tstr, free_vars=False)
            lt = LogicTheorist1956(known, use_csm=True, use_vct=True, priority_mode="min_k", work_limit=50)
            ok, _, _ = lt.prove_executive(prob)
            if ok:
                proved_all67 += 1
                if idx <= 52:
                    proved_first52 += 1
            if mem_mode == "all_prior" or (mem_mode == "proved_only" and ok):
                known.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))
        print(f"  {desc:<64}: {proved_first52:2d}/52 (first 52) | {proved_all67:2d}/67 (all 67)")

    print("\n=== E4.2: Subproblem Priority Rule Ex(CK) (p. 46: Minimal-K vs FIFO vs Maximal-K) ===")
    for pmode, desc in [
        ("min_k", "Minimal-K first (Ex(CK) simplest subproblem heuristic, p. 46)"),
        ("fifo", "FIFO (breadth-first queue order)"),
        ("max_k", "Maximal-K first (anti-heuristic negative control: deepest first)"),
    ]:
        known = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
        proved = 0
        multi_proved = 0
        multi_work = 0
        for tid, tstr in theorems_raw:
            prob = parse_pm_expr(tstr, free_vars=False)
            lt = LogicTheorist1956(known, use_csm=True, use_vct=True, priority_mode=pmode, work_limit=50)
            ok, _, depth = lt.prove_executive(prob)
            if ok:
                proved += 1
                if depth >= 1:
                    multi_proved += 1
                    multi_work += lt.work_w
            known.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))
        print(f"  {desc:<66}: total proved={proved}/67, depth>=1 subproblem proofs={multi_proved} (total W={multi_work})")


def main():
    repo_dir = "/root/dm-study/foundations/newell-simon-1956/work/dmoews_ipli_run"
    axioms_raw, theorems_raw = load_pm_axioms_and_theorems(repo_dir)
    run_e2_dmoews_campaign(repo_dir)
    run_e3_similarity_and_contraction(axioms_raw, theorems_raw)
    run_e4_learning_and_priority(axioms_raw, theorems_raw)

    # Compare per-theorem results between dmoews IPL-I and LogicTheorist1956:
    print("\n=== Disagreement Check: dmoews IPL-I (48/67) vs LogicTheorist1956 (46/67) ===")
    ana_lines = open(f"{repo_dir}/pm_analyzed.out", "r", encoding="utf-8").read().splitlines()
    dmoews_status = [l.startswith("Proven: ") for l in ana_lines if l.startswith("Proven: ") or l.startswith("Not proven: ")]
    tr_free = str.maketrans({"p": "A", "q": "B", "r": "C", "s": "D"})
    known = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
    for idx, (tid, tstr) in enumerate(theorems_raw):
        prob = parse_pm_expr(tstr, free_vars=False)
        lt = LogicTheorist1956(known, use_csm=True, use_vct=True, priority_mode="min_k", work_limit=50)
        ok, trace, _ = lt.prove_executive(prob)
        if ok != dmoews_status[idx]:
            print(f"  Diff on #{idx+1} {tid} {tstr}: dmoews={dmoews_status[idx]}, ours={ok} (our W={lt.work_w}, trace={trace})")
        known.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))

    # Also check which theorem is proved by MSb with CSm OFF (20) vs CSm ON (19):
    known_on = [(aid, parse_pm_expr(astr.translate(tr_free), free_vars=True)) for aid, astr in axioms_raw]
    for tid, tstr in theorems_raw:
        prob = parse_pm_expr(tstr, free_vars=False)
        r_on = LogicTheorist1956(known_on, use_csm=True).method_msb(prob)
        r_off = LogicTheorist1956(known_on, use_csm=False).method_msb(prob)
        if (r_on is not None) != (r_off is not None):
            print(f"  MSb CSm ON vs OFF diff on {tid} {tstr}: CSm_ON={r_on}, CSm_OFF={r_off}")
        known_on.append((tid, parse_pm_expr(tstr.translate(tr_free), free_vars=True)))


if __name__ == "__main__":
    main()
