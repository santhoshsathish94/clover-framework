"""70-task benchmark on the laptop with the frozen Clover scorer.

  --arm agent : POST each task to the local server's /api/chat (fresh session, cands=2, remember=False), as the
                standardized suite does (run-standardized-70-task-benchmark.py).
  --arm raw   : LFM3 alone, exactly as benchmark_raw_models_without_clover_70_tasks.py (MMLU: C engine + tied head over
                A-D; other 60: greedy HF generate, enable_thinking=False, batch 6, max_length 512, max_new 96..128).
Rust tasks need rustc (not on the laptop): replies are saved and marked "rust_scored_on_k3"; rescore_rust.py scores them on k3.
Results append to <out>.jsonl (resumable); a summary goes to <out>.summary.json.
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import clover_scorer as S  # noqa: E402


def score(task, reply):
    if task["benchmark"] == "Rust_Generative_Compilation":
        return None, None, "rust_scored_on_k3"
    ok, sc, det = S.score_single_task(task, reply)
    return bool(ok), float(sc), str(det)[:200]


def post_chat(url, sid, prompt):
    body = json.dumps({"session_id": sid, "prompt": prompt, "cands": 2, "remember": False}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode("utf-8"))


def run_agent(tasks, done, out_f, url, label):
    for i, t in enumerate(tasks, 1):
        if t["task_id"] in done:
            continue
        t0 = time.perf_counter()
        err, res = None, {}
        try:
            res = post_chat(url, f"bench_{label}_{t['task_id']}", t["prompt"])
            reply = str(res.get("reply") or "")
        except Exception as exc:
            reply, err = "", f"{type(exc).__name__}: {exc}"
        ok, sc, det = score(t, reply) if err is None else (False, 0.0, err)
        rec = {"index": i, "task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"],
               "passed": ok, "score": sc, "score_detail": det, "wall_s": round(time.perf_counter() - t0, 1),
               "plan_ms": res.get("plan_lat_ms"), "render_ms": res.get("k3_moe_plus_ncm_ms"), "latent_steps": res.get("latent_steps"),
               "exit_reason": res.get("exit_reason"), "tools": [c.get("tool") for c in res.get("tool_calls") or []],
               "plans": res.get("plans"), "planner_text": res.get("raw_decoded_gen"), "raw_output": reply, "error": err}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[{i:02d}/70] {'PASS' if ok else ('----' if ok is None else 'FAIL')} {t['task_id']:36s} {rec['wall_s']:6.1f}s "
              f"tools={rec['tools']} {det}", flush=True)


def run_plan(tasks, done, out_f, base, planner, search_cache, search_mode):
    import lfm3_agent
    eng = lfm3_agent.CloverLFM3Engine(base, planner, search_cache=search_cache, search_mode=search_mode, load_renderer=False)
    print("PLANNER", json.dumps(eng.describe()), flush=True)
    for i, t in enumerate(tasks, 1):
        if t["task_id"] in done:
            continue
        err, res = None, {}
        try:
            res = eng.plan_only(t["prompt"])
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        rec = {"index": i, "task_id": t["task_id"], "prompt": t["prompt"], **res, "error": err}
        out_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out_f.flush()
        print(f"[plan {i:02d}/70] {t['task_id']:36s} {res.get('plan_s', 0):6.1f}s latent={res.get('latent_steps')} "
              f"tools={[c.get('tool') for c in res.get('tool_calls') or []]} {err or ''}", flush=True)
    if hasattr(eng.search_tool, "stats"):
        print("SEARCH", json.dumps(eng.search_tool.stats), flush=True)


def run_render(tasks, done, out_f, base, planner, plans_path):
    import lfm3_agent
    planned = {}
    for ln in Path(plans_path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            planned[r["task_id"]] = r
    eng = lfm3_agent.CloverLFM3Engine(base, planner, load_planner=False)
    eng.plan_replay = {planned[t["task_id"]]["prompt"]: (planned[t["task_id"]]["plans"], planned[t["task_id"]]["latent_steps"],
                                                       planned[t["task_id"]]["exit_reason"], planned[t["task_id"]]["tool_calls"])
                       for t in tasks if t["task_id"] in planned and planned[t["task_id"]].get("error") is None}
    for i, t in enumerate(tasks, 1):
        if t["task_id"] in done:
            continue
        p = planned.get(t["task_id"]) or {}
        t0 = time.perf_counter()
        err, res = p.get("error") if p else "not planned", {}
        if err is None:
            try:
                res = eng.agent_reply(t["prompt"], cands=2, remember=False)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        reply = str(res.get("reply") or "")
        ok, sc, det = score(t, reply) if err is None else (False, 0.0, err)
        rec = {"index": i, "task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"],
               "passed": ok, "score": sc, "score_detail": det, "plan_s": p.get("plan_s"),
               "render_s": round(time.perf_counter() - t0, 1), "wall_s": round((p.get("plan_s") or 0) + time.perf_counter() - t0, 1),
               "latent_steps": p.get("latent_steps"), "exit_reason": p.get("exit_reason"),
               "tools": [c.get("tool") for c in p.get("tool_calls") or []], "plans": p.get("plans"),
               "planner_text": p.get("planner_text"), "raw_output": reply, "error": err, "pipeline": "split"}
        out_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out_f.flush()
        print(f"[render {i:02d}/70] {'PASS' if ok else ('----' if ok is None else 'FAIL')} {t['task_id']:36s} "
              f"plan={p.get('plan_s', 0):6.1f}s render={rec['render_s']:6.1f}s {det}", flush=True)


def run_raw(tasks, done, out_f, base):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lfm3_clover as LC
    model_dir = base / "data" / "lfm3" / "hf"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = 124893
    todo = [t for t in tasks if t["task_id"] not in done]
    mmlu = [t for t in todo if t["benchmark"] == "MMLU"]
    if mmlu:
        eng = LC.Lfm3NativeCPUEngine(base / "data" / "lfm3" / "pruned", base / "engine")
        emb = torch.from_numpy(__import__("numpy").memmap(base / "data" / "lfm3" / "pruned" / "global-0" / "embed_tokens.bf16.bin",
                                                          dtype="uint16", mode="r").copy()).view(torch.bfloat16).view(128000, 2048)
        flat, segs, last = [], [], []
        for t in mmlu:
            base_q = re.sub(r"\n*State the correct option letter.*$", "", t["prompt"].strip(), flags=re.IGNORECASE).strip()
            ids = tok.encode(f"{base_q}\nAnswer: (", add_special_tokens=False)[-256:]
            s0 = len(flat)
            flat.extend(ids)
            segs.extend([s0] * len(ids))
            last.append(len(flat) - 1)
        t0 = time.perf_counter()
        h, _ = eng.forward_ids(flat, seg_starts=segs)
        each = (time.perf_counter() - t0) / len(mmlu)
        l_ids = [tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"]
        for j, t in enumerate(mmlu):
            lg = F.linear(h[last[j]].to(torch.bfloat16), emb).float()
            pred = "ABCD"[int(torch.tensor([float(lg[i]) for i in l_ids]).argmax())]
            reply = f"The correct answer is ({pred})."
            ok, sc, det = score(t, reply)
            out_f.write(json.dumps({"task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"], "passed": ok,
                                    "score": sc, "score_detail": det, "wall_s": round(each, 2), "raw_output": reply}, ensure_ascii=False) + "\n")
        out_f.flush()
        del eng
        print(f"raw MMLU done ({len(mmlu)})", flush=True)
    gen = [t for t in todo if t["benchmark"] != "MMLU"]
    if not gen:
        return
    import lfm3_agent
    # The repo's arm 0 used bf16 on k3 (AVX512-BF16); bf16 is emulated on this laptop, so the planner's precision is used.
    model, _ = lfm3_agent.load_planner_backbone(model_dir, int8_ffn=True)
    print("RAW_PRECISION fp32 + int8 feed_forward (laptop planner precision)", flush=True)
    for b in range(0, len(gen), 6):
        batch, max_new, prompts = gen[b:b + 6], 96, []
        for t in batch:
            u = t["prompt"]
            if t["benchmark"] == "GSM8K":
                u += "\n Solve step by step and end your response with #### <number>."
                max_new = max(max_new, 112)
            elif t["benchmark"] in ("HumanEval_Python", "Rust_Generative_Compilation"):
                max_new = max(max_new, 128)
            elif t["benchmark"] == "Sudoku_Easy_9x9":
                max_new = max(max_new, 112)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False))
        t0 = time.perf_counter()
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            seqs = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=124900)
        each = (time.perf_counter() - t0) / len(batch)
        for row, t in enumerate(batch):
            reply = tok.decode(seqs[row][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ok, sc, det = score(t, reply)
            out_f.write(json.dumps({"task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"], "passed": ok,
                                    "score": sc, "score_detail": det, "wall_s": round(each, 2), "raw_output": reply}, ensure_ascii=False) + "\n")
            out_f.flush()
        print(f"raw batch {b // 6 + 1}/{(len(gen) + 5) // 6} {each * len(batch):.0f}s", flush=True)


def summarize(path, tasks):
    recs = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            recs[r["task_id"]] = r
    cats = collections.defaultdict(lambda: {"passed": 0, "scored": 0, "total": 0, "score_sum": 0.0, "pending_rust": 0})
    for t in tasks:
        r = recs.get(t["task_id"])
        c = cats[t["category"]]
        c["total"] += 1
        if r is None:
            continue
        if r["passed"] is None:
            c["pending_rust"] += 1
            continue
        c["scored"] += 1
        c["passed"] += int(bool(r["passed"]))
        c["score_sum"] += float(r["score"] or 0.0)
    tot = {k: sum(c[k] for c in cats.values()) for k in ("passed", "scored", "total", "pending_rust")}
    tot["mean_score_pct_scored"] = round(100.0 * sum(c["score_sum"] for c in cats.values()) / max(1, tot["scored"]), 2)
    return {"by_category": {k: dict(v) for k, v in sorted(cats.items())}, "overall": tot,
            "wall_s_sum": round(sum(float(r.get("wall_s") or 0) for r in recs.values()), 1)}


def keep_awake():
    """Process-scoped request that Windows not idle-sleep during a run; Windows drops it when this process exits."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("agent", "raw", "plan", "render"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--base", default=str(HERE.parent))
    ap.add_argument("--planner", default=None, help="planner dir (default: base LFM3)")
    ap.add_argument("--plans", default=None, help="render arm: the plan arm's output file")
    ap.add_argument("--search-cache", default=None)
    ap.add_argument("--search-mode", choices=("record", "replay", "live"), default="record")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated task_ids (trial runs)")
    a = ap.parse_args()
    planner = a.planner or str(Path(a.base) / "data" / "lfm3" / "hf")
    tasks = json.loads((HERE / "tasks70.json").read_text(encoding="utf-8"))
    if a.ids:
        keep = set(a.ids.split(","))
        tasks = [t for t in tasks if t["task_id"] in keep]
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(ln)["task_id"] for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if a.arm == "agent":
        with urllib.request.urlopen(a.url.rstrip("/") + "/api/health", timeout=30) as r:
            health = json.loads(r.read())
        (out.with_suffix(".engine.json")).write_text(json.dumps(health, indent=2), encoding="utf-8")
    with open(out, "a", encoding="utf-8") as f:
        if a.arm == "agent":
            run_agent(tasks, done, f, a.url, out.stem)
        elif a.arm == "plan":
            run_plan(tasks, done, f, Path(a.base), planner, a.search_cache, a.search_mode)
        elif a.arm == "render":
            run_render(tasks, done, f, Path(a.base), planner, a.plans)
        else:
            run_raw(tasks, done, f, Path(a.base))
    if a.arm == "plan":
        print("BENCH70_PLAN_DONE", flush=True)
        return
    summ = summarize(out, tasks)
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["overall"]), flush=True)
    print("BENCH70_DONE", flush=True)


if __name__ == "__main__":
    main()
