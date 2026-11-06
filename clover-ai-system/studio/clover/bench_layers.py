"""Layer-by-layer ablation on the 70-task matrix (laptop), starting from LFM3 alone.

  --step 0 : LFM3 alone as its model card documents it (chat template with reasoning, greedy, up to 2,048 new
             tokens per turn), no tools.
  --step 1 : step 0 + LFM3's own tool calling: run_python, web_search / fetch_url (record/replay cache, so every
             later step sees the same web results), read-only files, time. Shell and file writes stay off.
Later steps add one Clover layer at a time on top of step 1 (--layers memory|ontology|compiler|ncm, see lfm3_layers).
With --base-run (step 1's jsonl) a task is regenerated only when a layer changes LFM3's input: memory/ontology context
found, or the compiler rejects the base reply's code. Otherwise the base reply is reused (the pipeline is deterministic;
--check-det re-runs a few base tasks to confirm) and only the post-hoc layers (NCM render, grounding filter) apply.
--final-on-limit / --regen-overflow measure two harness fixes the same way, on the base tasks that ran out of tool
rounds or of thinking room.
Same tasks (tasks70.json), same frozen scorer (clover_scorer via bench70.score), same engine build, fixed system
prompt (no date). Rust replies are scored on k3 (rescore_rust.py). Resumable: appends to <out>.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import winperf  # noqa: E402

THROTTLE_OFF = winperf.disable_power_throttling()
import bench70  # noqa: E402
import lfm3_native as N  # noqa: E402
import paths  # noqa: E402
import permissions as P  # noqa: E402

SYSTEM = "You are a helpful assistant."
# bench70.run_raw (the repo's arm 0) adds this to GSM8K prompts; the frozen scorer reads '#### n' first and otherwise can
# take a price like '$2' from a correct '\boxed{18}' answer.
GSM8K_SUFFIX = "\n Solve step by step and end your response with #### <number>."


class ReplayWeb:
    """The Studio's web search (web_search.WebSearch, provider from config/search.json under `base`) with a
    call -> result cache: record once, replay in every later step. Keys name the provider."""

    def __init__(self, path, mode, base=None):
        self.path, self.mode, self.base, self.cache, self.inner = Path(path), mode, base, {}, None
        self.stats = {"hits": 0, "live": 0, "missing": 0}
        if self.path.exists():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    self.cache[r["key"]] = r["result"]

    def _inner(self):
        if self.inner is None:
            import web_search
            cfg = web_search.SearchConfig.shared(paths.config_file(self.base, "search.json"),
                                                 paths.vault_file(self.base, "search-keys.json")) if self.base else None
            self.inner = web_search.WebSearch(cfg)
        return self.inner

    def _get(self, key, fn):
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        if self.mode == "replay":
            self.stats["missing"] += 1
            return "No live external results (not in the replay cache)."
        res = fn(self._inner())
        self.stats["live"] += 1
        self.cache[key] = res
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
        return res

    def execute(self, query, typed=()):
        return self._get(f"search:{self._inner().provider()[0]}\t" + query, lambda w: w.execute(query, typed=typed))

    def fetch_url_content(self, url, max_chars=5000, typed=()):
        return self._get(f"fetch\t{max_chars}\t{url}", lambda w: w.fetch_url_content(url, max_chars=max_chars, typed=typed))

    def research(self, query):
        return self._get(f"research:{self._inner().provider()[0]}\t" + query, lambda w: w.research(query))


def recall_memory(base, src):
    """The Studio's Layer 2 over a copy of the chat files in `src`, in a temporary repository; every task is asked as
    the first message of a new chat. -> (repo, index)"""
    import shutil
    import clover_repo
    import memory_index
    tmp = Path(tempfile.mkdtemp())
    (tmp / "AGENTS.md").write_text("# AGENTS.md\n", encoding="utf-8")
    repo = clover_repo.Repo(tmp, tmp / "AGENTS.md")
    shutil.copytree(Path(src) / "sessions", repo.dir / "sessions", dirs_exist_ok=True)
    repo.reindex()
    idx = memory_index.MemoryIndex(tmp, repo, start=False, embedder=memory_index.Embedder(
        Path(base) / "data" / "embed" / "bge-small-en-v1.5"))
    idx.refresh()
    repo.current = ("bench-task", 1)
    print(f"RECALL_MEMORY {len(idx.units)} messages from {len(repo.session_files())} chats", flush=True)
    return repo, idx


def main():
    bench70.keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--base-run", default=None)
    ap.add_argument("--check-det", type=int, default=2)
    # harness fixes measured like layers, on the base tasks they can change: tool budget used up / thinking budget used up
    ap.add_argument("--final-on-limit", action="store_true")
    ap.add_argument("--regen-overflow", action="store_true")
    ap.add_argument("--force-answer", action="store_true")
    ap.add_argument("--engine", choices=("cpu", "gpu", "omni", "omni-cpu"), default="cpu")
    ap.add_argument("--quant", choices=("q4_0", "q4_k_m"), default="q4_0")  # omni only
    ap.add_argument("--tools-when-asked", action="store_true")  # the Studio default: tools only if the prompt asks
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(HERE.parent))
    ap.add_argument("--web-cache", default=None)
    ap.add_argument("--web-mode", choices=("record", "replay"), default="record")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    # recall (Layer 2) needs a user history: Clover chat files (sessions/<month>/<chat>.jsonl) under this folder
    ap.add_argument("--recall-memory", default=None)
    a = ap.parse_args()
    base = Path(a.base)
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

    perms = P.PermissionStore(Path(tempfile.mkdtemp()) / "perms.json")
    perms.update({"code": "allow", "internet": "allow", "files": "allow", "shell": "deny", "edit_files": "deny"})
    if a.engine.startswith("omni"):
        import omni_native
        eng = omni_native.OmniNative(base, perms, engine="cpu" if a.engine == "omni-cpu" else "gpu", quant=a.quant)
    else:
        eng = N.LFM3Native(base, perms, engine=a.engine)
    eng._web = ReplayWeb(a.web_cache or (out.parent / "web-cache.jsonl"), a.web_mode, base)
    layers = tuple(x for x in a.layers.split(",") if x)
    kit = None
    if layers:
        import lfm3_layers
        assert all(x in lfm3_layers.LAYERS + (("recall",) if a.recall_memory else ()) for x in layers), layers
        kit = lfm3_layers.LayerKit(base)
        if "recall" in layers:
            kit.records, kit.recall = recall_memory(base, a.recall_memory)
    tools_on = a.step >= 1
    base_recs = {}
    if a.base_run:
        for ln in Path(a.base_run).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                base_recs[r["task_id"]] = r
    by_id = {t["task_id"]: t for t in tasks}

    def run(t, lay):
        prompt = t["prompt"] + (GSM8K_SUFFIX if t["benchmark"] == "GSM8K" else "")
        only = N.asked_tools(t["prompt"]) if a.tools_when_asked else None
        return eng.run_turn([], prompt, tools=tools_on and (only is None or bool(only)), sampling="greedy",
                            thinking=True, max_new=a.max_new, max_rounds=6, system=SYSTEM, layers=lay, kit=kit,
                            layer_query=t["prompt"], final_on_limit=a.final_on_limit, force_answer=a.force_answer,
                            only_tools=only)

    det_check = []
    if base_recs and a.check_det:
        cands = sorted((r for r in base_recs.values() if r["task_id"] in by_id and not r.get("error")),
                       key=lambda r: r["wall_s"])[:a.check_det]
        for r in cands:
            same = run(by_id[r["task_id"]], ()).get("reply") == r["raw_output"]
            det_check.append({"task_id": r["task_id"], "identical": same})
            print("DETERMINISM", r["task_id"], same, flush=True)
        if not all(d["identical"] for d in det_check):
            print("DETERMINISM FAILED: regenerating every task", flush=True)
            base_recs = {}
    meta = {"step": a.step, "engine": eng.describe(), "throttle_off": THROTTLE_OFF, "system": SYSTEM,
            "gsm8k_suffix": GSM8K_SUFFIX, "sampling": "greedy", "thinking": True, "max_new": a.max_new,
            "tools": tools_on, "layers": layers, "base_run": a.base_run, "determinism": det_check,
            "final_on_limit": a.final_on_limit, "regen_overflow": a.regen_overflow, "force_answer": a.force_answer,
            "tools_when_asked": a.tools_when_asked}
    out.with_suffix(".engine.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("LAYERS_STEP", json.dumps(meta, default=str), flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for i, t in enumerate(tasks, 1):
            if t["task_id"] in done:
                continue
            t0 = time.perf_counter()
            err, res, derived = None, {}, False
            try:
                b = base_recs.get(t["task_id"])
                if b is not None and not b.get("error"):
                    q = t["prompt"]
                    regen = ("memory" in layers and kit.memory_context(q)[0] is not None) or \
                            ("recall" in layers and kit.recall_context(q)[0] is not None) or \
                            ("ontology" in layers and kit.ontology_context(q)[0] is not None) or \
                            ("compiler" in layers and kit.compiler_check(q, b["raw_output"])[0] is not None) or \
                            (a.final_on_limit and b["raw_output"].startswith(("(Stopped after the maximum number of tool rounds",
                                                                              "(Clover used all "))) or \
                            (a.regen_overflow and b["raw_output"].startswith("(Clover ran out of thinking room"))
                    if not regen:
                        derived, reply, log = True, b["raw_output"], []
                        if "compiler" in layers:
                            log.append(kit.compiler_check(q, reply)[1])
                        if "ncm" in layers:
                            reply, info = kit.ncm_render(reply)
                            log.append(info)
                        if "ontology" in layers:
                            reply, ch = kit.grounding_filter(q, reply)
                            log.append({"layer": "ontology_grounding", "changed": ch})
                        res = {"reply": reply, "tool_calls": b.get("tool_calls"), "rounds": b.get("rounds"),
                               "reasoning": b.get("reasoning"), "layers": log,
                               "generation": [{"new_tokens": b.get("new_tokens", 0), "finish": x} for x in b.get("finish") or []]}
                if not derived:
                    res = run(t, layers)
                reply = str(res.get("reply") or "")
            except Exception as exc:
                reply, err = "", f"{type(exc).__name__}: {exc}"
            ok, sc, det = bench70.score(t, reply) if err is None else (False, 0.0, err)
            gens = res.get("generation") or []
            rec = {"index": i, "task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"],
                   "passed": ok, "score": sc, "score_detail": det, "wall_s": round(time.perf_counter() - t0, 1),
                   "new_tokens": sum(g.get("new_tokens", 0) for g in gens), "rounds": res.get("rounds"),
                   "finish": [g.get("finish") for g in gens],
                   "tools": [(c["tool"], c["ran"]) for c in res.get("tool_calls") or []],
                   "tool_calls": res.get("tool_calls"), "reasoning": res.get("reasoning"), "raw_output": reply, "error": err,
                   "derived_from_base": derived, "layer_log": res.get("layers")}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"[{i:02d}/{len(tasks)}] {'PASS' if ok else ('----' if ok is None else 'FAIL')} {t['task_id']:36s} "
                  f"{rec['wall_s']:6.1f}s tok={rec['new_tokens']} {'base' if derived else 'gen'} "
                  f"layers={[x.get('layer') for x in res.get('layers') or []]} tools={rec['tools']} {det}", flush=True)
    summ = bench70.summarize(out, tasks)
    summ["web"] = eng._web.stats
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["overall"]), flush=True)
    print("LAYERS_STEP_DONE", a.step, flush=True)


if __name__ == "__main__":
    main()
