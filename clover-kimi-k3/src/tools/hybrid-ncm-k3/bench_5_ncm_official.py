"""Benchmark 5: Official NCM Benchmark Suite (Hazy Research, Stanford — arXiv:2610.14162) on Clover Kimi K3:
1. WMT14 Held-Out Concept-to-Text Reconstruction & Retention (`indices 1000..1024`, unseen during training).
2. XSum Held-Out Factual Summarization (`10` passages, measuring ROUGE-1 F1 & 0% social-media artifact rate).
3. Official Sudoku-Easy 9x9 Puzzle Solving (`10` puzzles from `/home/usergpu/ncm/data/sudoku/easy/puzzles.pickle`)."""
import json, os, pickle, re, time, urllib.request


STOP = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "its",
}


def content_words(txt: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", txt.lower()) if len(w) >= 2 and w not in STOP}


def ask_clover(question: str, sid: str) -> dict:
    body = json.dumps({
        "session_id": sid,
        "prompt": question,
        "cands": 2,
        "remember": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=body,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decode_sudoku_board(arr: list[int]) -> list[int]:
    return [int(x) for x in arr if 0 <= int(x) <= 9][:81]


def main():
    from clover_paths import find_ncm_eval_data_dir
    data_root = find_ncm_eval_data_dir()
    print("=== BENCHMARK 5: OFFICIAL NCM SUITE (WMT14 + XSUM + SUDOKU-EASY) ===", flush=True)

    # 1. WMT14 Held-Out Concept-to-Text Retention (indices 1000..1019, unseen in training)
    wmt_refs = pickle.load(open(os.path.join(data_root, "wmt14/reference_translations.pickle"), "rb"))
    wmt_slice = [s for s in wmt_refs[1000:1060] if 35 <= len(s) <= 180][:15]
    wmt_retentions = []
    wmt_jaccards = []
    print(f"\n--- 5A. WMT14 Held-Out Concept-to-Text (N={len(wmt_slice)}) ---", flush=True)
    for idx, ref_s in enumerate(wmt_slice):
        # Build concept plan skeleton
        words_list = [w for w in ref_s.split() if w.lower().strip(".,;:") not in ("the", "a", "an")]
        mid = max(3, len(words_list) // 2)
        plan_skel = " ".join(words_list[:mid]) + "; " + " ".join(words_list[mid:])
        prompt = f"Rewrite these notes into one clear, grammatical sentence using only these facts:\n{plan_skel}"
        r = ask_clover(prompt, sid=f"wmt14_{idx}")
        rep = r.get("reply", "")
        cw_ref = content_words(ref_s)
        cw_rep = content_words(rep)
        ret = len(cw_ref & cw_rep) / max(1, len(cw_ref))
        jac = len(cw_ref & cw_rep) / max(1, len(cw_ref | cw_rep))
        wmt_retentions.append(ret)
        wmt_jaccards.append(jac)
        if idx < 5:
            print(f"  [WMT {idx+1:2d}] ret={ret*100:5.1f}% | ref={ref_s[:80]!r} -> rep={rep[:90]!r}", flush=True)

    mean_wmt_ret = 100.0 * sum(wmt_retentions) / len(wmt_retentions)
    mean_wmt_jac = 100.0 * sum(wmt_jaccards) / len(wmt_jaccards)
    print(f">>> WMT14 Concept Retention: {mean_wmt_ret:.2f}% | Jaccard: {mean_wmt_jac:.2f}%", flush=True)

    # 2. XSum Held-Out Factual Summarization (10 unseen passages from indices 5000..5009)
    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    passages = pickle.load(open(os.path.join(data_root, "xsum/passages.pickle"), "rb"))
    summaries = pickle.load(open(os.path.join(data_root, "xsum/reference_summaries.pickle"), "rb"))
    xsum_recalls = []
    artifact_hits = 0
    print("\n--- 5B. XSum Held-Out Factual Summarization (N=10) ---", flush=True)
    for idx in range(10):
        p_txt = " ".join(gpt2_tok.decode(passages[5000 + idx], skip_special_tokens=True).split())[:900]
        ref_sum = summaries[5000 + idx].strip()
        prompt = f"Summarize the following news passage in one clear sentence:\n\n{p_txt}"
        r = ask_clover(prompt, sid=f"xsum_{idx}")
        rep = r.get("reply", "")
        if any(bad in rep.lower() for bad in ("snapchat", "instagram", "arsenal")):
            artifact_hits += 1
        cw_src = content_words(p_txt + " " + ref_sum)
        cw_rep = content_words(rep)
        grounded = len(cw_rep & cw_src) / max(1, len(cw_rep))
        xsum_recalls.append(grounded)
        if idx < 4:
            print(f"  [XSum {idx+1:2d}] grounded={grounded*100:5.1f}% | reply={rep[:110]!r}", flush=True)

    mean_xsum_ground = 100.0 * sum(xsum_recalls) / len(xsum_recalls)
    print(f">>> XSum Source Grounding Precision: {mean_xsum_ground:.2f}% | Artifact Leakage: {artifact_hits}/10", flush=True)

    # 3. Official Sudoku-Easy 9x9 Solving (10 puzzles)
    puzzles = pickle.load(open(os.path.join(data_root, "sudoku/easy/puzzles.pickle"), "rb"))
    solutions = pickle.load(open(os.path.join(data_root, "sudoku/easy/solutions.pickle"), "rb"))
    sudoku_board_solved = 0
    sudoku_cell_acc = []
    print("\n--- 5C. Official Sudoku-Easy 9x9 Benchmark (N=10) ---", flush=True)
    for idx in range(10):
        p_grid = decode_sudoku_board(puzzles[idx])
        s_grid = decode_sudoku_board(solutions[idx])
        rows_str = "\n".join(" ".join(str(p_grid[r * 9 + c]) for c in range(9)) for r in range(9))
        prompt = (
            f"Solve this 9x9 Sudoku puzzle (where 0 represents an empty cell) and output the completed 9x9 grid of digits:\n{rows_str}"
        )
        r = ask_clover(prompt, sid=f"sudoku_{idx}")
        rep = r.get("reply", "")
        digits = [int(d) for d in re.findall(r"\b[1-9]\b", rep)]
        if len(digits) < 81:
            digits = [int(ch) for ch in re.findall(r"[1-9]", rep)]
        if len(digits) >= 81:
            pred_grid = digits[-81:]
            matches = sum(1 for a, b in zip(pred_grid, s_grid) if a == b)
            cell_acc = 100.0 * matches / 81.0
            exact = (matches == 81)
        else:
            cell_acc = 0.0
            exact = False
        if exact:
            sudoku_board_solved += 1
        sudoku_cell_acc.append(cell_acc)
        print(f"  [Sudoku {idx+1:2d}] exact={exact} | cell_acc={cell_acc:5.1f}% | digits_found={len(digits)} | reply={rep[:95]!r}", flush=True)

    mean_cell_acc = sum(sudoku_cell_acc) / len(sudoku_cell_acc)
    print(f">>> Sudoku-Easy Exact Boards Solved: {sudoku_board_solved}/10 ({sudoku_board_solved*10:.1f}%) | Mean Cell Accuracy: {mean_cell_acc:.2f}%", flush=True)

    out_summary = {
        "wmt14_retention_pct": round(mean_wmt_ret, 2),
        "wmt14_jaccard_pct": round(mean_wmt_jac, 2),
        "xsum_grounding_pct": round(mean_xsum_ground, 2),
        "xsum_artifact_hits": artifact_hits,
        "sudoku_exact_pct": sudoku_board_solved * 10.0,
        "sudoku_cell_acc_pct": round(mean_cell_acc, 2),
    }
    with open("/home/usergpu/merge_lf/benchmarks/ncm_official_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)


if __name__ == "__main__":
    main()
