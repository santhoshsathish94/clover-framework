"""C1 + C2 + C3 together: the phone's fused runtime (clover-framework s23_on_device_clover.py) as a Studio engine.

C1 is our trained LFM3-Omni-2B trunks on Clover's C engine (CPU, 4-bit), C2 the HybridLink latent steps when the
model opens <latent>, C3 the 11-level NCM rendering from C1's own hidden states - all from the precomputed parts in the
clover-ai-studio dataset (setup component c2c3, data/omni/precomputed). It runs in its own process (fused_worker.py).
What it cannot do, from its design: photos, audio and video (the C engine reads text), tools (the probe of the regression
ran it without), and prompts past ~1,280 tokens (run_cycle's limit) - long documents and long chats are cut. Clover's
S layers (memory, ontology, workspace, records, recall, the user's preferred form) and the Clover cycle are added as on
the other engines, without the compiler check and without the cycle's tools (web search, a Python solution).
Measured against the GPU engine in the regression before it can be a default (human, 2026-11-03: "default only if not
worse")."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUSED = HERE / "fused"  # framework.zip (the framework's code at a pinned commit) and the prebuilt engine DLL
NEEDS = ("trunks/global-0/embed_tokens.bf16.bin", "trunks/tokenizer.json", "hybridagent/plan-trunk-0/index.bin",
         "bridges/bridge-trunk-0/index.bin", "ncm/lfm3_ncm_core_bf16.pt", "ncm/vqvae_wte.bf16.bin")


def precomputed(base):
    return Path(base) / "data" / "omni" / "precomputed"


def missing(base):
    p = precomputed(base)
    out = [n for n in NEEDS if not (p / n).is_file()]
    if not (FUSED / "framework.zip").is_file():
        out.append("clover/fused/framework.zip")
    if not (FUSED / "liblfm3_cpu_engine.dll").is_file():
        out.append("clover/fused/liblfm3_cpu_engine.dll")
    return out


def _junction(link, target):
    import _winapi
    if not link.exists():
        _winapi.CreateJunction(str(target), str(link))


def _link(link, target):
    if not link.exists():
        try:
            os.link(target, link)
        except OSError:
            shutil.copyfile(target, link)


def layout(base):
    """The runtime's folders (pruned/, ncm/, ncm/precomputed/) as junctions and hard links into data/omni/precomputed
    and the framework's code from framework.zip, under runtime/fused. -> (code folder, runtime base folder)"""
    p, root = precomputed(base), Path(base) / "runtime" / "fused"
    z = FUSED / "framework.zip"
    stamp = hashlib.sha256(z.read_bytes()).hexdigest()[:16]
    code = root / f"code-{stamp}"
    mobile = code / "clover-ai-system" / "mobile"
    if not (mobile / "s23_on_device_clover.py").is_file():
        tmp = root / f"code-{stamp}.new"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        os.replace(tmp, code)
    so = mobile / "liblfm3_cpu_engine.so"  # the name the runtime loads; newer than the .c so it is not rebuilt
    shutil.copyfile(FUSED / "liblfm3_cpu_engine.dll", so)
    os.utime(so)
    rb = root / "base"
    pre = rb / "ncm" / "precomputed"
    pre.mkdir(parents=True, exist_ok=True)
    _link(rb / "tokenizer.json", p / "trunks" / "tokenizer.json")
    _junction(rb / "pruned", p / "trunks")
    for n in ("lfm3_ncm_core_bf16.pt", "vqvae_wte.bf16.bin", "vqvae_dec.bf16.bin", "config.yaml", "ncm_runtime.py"):
        _link(rb / "ncm" / n, p / "ncm" / n)
    for n in ("models", "utils"):
        _junction(rb / "ncm" / n, p / "ncm" / n)
    _link(pre / "manifest.json", p / "ncm" / "manifest.json")
    _junction(pre / "plan-trunk-0", p / "hybridagent" / "plan-trunk-0")
    _junction(pre / "bridge-trunk-0", p / "bridges" / "bridge-trunk-0")
    for d in (p / "ncm").iterdir():
        if d.is_dir() and d.name.split("-")[0] in ("ncm", "ptr", "vqvae"):
            _junction(pre / d.name, d)
    return mobile, rb


class FusedEngine:
    inputs = ["text", "vision", "audio", "video"]
    engine_kind = "fused"
    gpu = None
    side_gate = None

    def __init__(self, base_dir, perms, progress=None, layer=3):
        import llamacpp_engine as L
        progress = progress or (lambda m: None)
        self.base_dir, self.perms, self.layer = Path(base_dir), perms, int(layer)
        need = missing(base_dir)
        if need:
            raise FileNotFoundError("C1 + C2 + C3 needs " + ", ".join(need) + " (setup component c2c3)")
        progress("C1 + C2 + C3: laying out the precomputed parts")
        code, rb = layout(base_dir)
        py = Path(sys.executable)
        if py.name.lower() == "pythonw.exe" and (py.parent / "python.exe").is_file():
            py = py.parent / "python.exe"
        log = self.base_dir / "logs" / "fused-worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log, "w", encoding="utf-8", errors="replace")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        progress("C1 + C2 + C3: loading our trained LFM3-Omni-2B, HybridLink and NCM on the CPU")
        t0 = time.time()
        self.proc = subprocess.Popen([str(py), str(HERE / "fused_worker.py"), str(code), str(rb)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._log, cwd=str(code), env=env, text=True,
                                     encoding="utf-8", creationflags=0x08000000 if os.name == "nt" else 0)
        self._job = L._kill_with_parent(self.proc)
        L._no_power_throttling(self.proc)
        line = self._answer()
        if not line:
            raise RuntimeError(f"the C1 + C2 + C3 runtime stopped while loading; see {log}")
        self.info = json.loads(line)
        self.setup_s = round(time.time() - t0, 1)
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self.workdir, self.workname, self.after_write = None, "", None
        self.c1_inference_only = False

    def _answer(self):
        """The worker's next JSON line ('' when it stopped); anything else on the channel is skipped."""
        while True:
            line = self.proc.stdout.readline()
            if not line or line.lstrip().startswith("{"):
                return line

    def _ask(self, text, history, max_new):
        """One call of the runtime: text with the last messages of history (user/assistant only) -> its answer dict."""
        history = [{"role": m.get("role"), "content": str(m.get("content") or "")} for m in (history or [])
                   if m.get("role") in ("user", "assistant")][-6:]
        req = {"prompt": text, "history": history, "layer": self.layer, "ncm": self.layer >= 3,
               "max_new": min(512, int(max_new or 256))}
        with self.lock:
            self.stage = {"name": "C1 + C2 + C3 on the CPU", "since": time.time()}
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                line = self._answer()
            finally:
                self.stage = {"name": "idle", "since": time.time()}
        if not line:
            raise RuntimeError("the C1 + C2 + C3 runtime stopped; see logs/fused-worker.log")
        ans = json.loads(line)
        if ans.get("error"):
            raise RuntimeError("C1 + C2 + C3: " + ans["error"])
        return ans

    @staticmethod
    def _generation(ans):
        return {"prompt_tokens": ans.get("input_tokens"), "new_tokens": ans.get("generated_tokens"),
                "decode_s": ans.get("seconds"), "finish": "stop"}

    @staticmethod
    def _fused_layer(answers):
        last = answers[-1] if answers else {}
        return {"layer": "fused", "c1": "trained LFM3-Omni-2B on the C engine", "c1_ms": last.get("c1"),
                "c2": last.get("c2"), "c3": last.get("c3"),
                "seconds": round(sum(a.get("seconds") or 0 for a in answers), 2), "calls": len(answers)}

    def _context(self, layers, kit, query):
        """Clover's observations for the query (memory, ontology, workspace, records) -> (blocks, layer infos)."""
        blocks, infos = [], []
        if kit is not None:
            for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"),
                             ("workspace", "workspace_context"), ("records", "records_context")):
                if name in layers:
                    block, info = getattr(kit, fn)(query)
                    if block:
                        blocks.append(block)
                        infos.append(info)
        return blocks, infos

    def describe(self):
        return {"mode": "fused", "model": "LFM3-Omni-2B (our trained trunks)", "engine": "fused",
                "device": "CPU", "backend": "Clover C engine + HybridLink + NCM (fused runtime)",
                "layers": ["C1 trained LFM3-Omni-2B (C engine, 4-bit)", "C2 HybridLink latent steps", "C3 11-level NCM"],
                "inputs": self.inputs, "outputs": ["text"], "setup_s": self.setup_s, "threads": self.info.get("threads"),
                "tools": [], "max_tokens": 1280, "reasoning": False}

    def run_turn(self, hist, prompt, on_event=None, cancel=None, max_new=2048, media=(), layers=(), kit=None,
                 layer_query=None, meaning=None, tools=True, extra_tools=(), only_tools=None, ask=None, allow=(), **kw):
        import clover_cycle as C
        import lfm3_native as N
        import omni_native as ON
        import semantics
        import workspaces as WS
        emit = on_event or (lambda e: None)
        query = layer_query or prompt
        op_plan = C.OperatingCyclePlan(emit, model_name="LFM3-Omni-2B (C1+C2+C3)", query=query, offered_tools=only_tools or ())
        self.active_op_plan = op_plan
        op_plan.start_stage("context")
        after, layer_log = self._context(layers, kit, query)
        for info in layer_log:
            emit(dict(info, type="layer"))
        ws_active_p = (kit.workspaces.active() or {}).get("path") if (kit is not None and getattr(kit, "workspaces", None) is not None) else None
        l_blk, l_info = WS.local_context_for_paths(semantics.user_paths(prompt), workdir=getattr(self, "workdir", None), active_ws_path=ws_active_p)
        if l_blk:
            after.append(l_blk)
            layer_log.append(l_info)
            emit(dict(l_info, type="layer"))
        before = []
        if kit is not None and "recall" in layers:
            block, info = kit.recall_context(query, sum(1 for m in hist or () if m.get("role") == "user"))
            if block:
                before.append(block)
            if info:
                layer_log.append(info)
                if info.get("refs"):
                    emit(dict(info, type="layer"))
        if meaning is not None and getattr(meaning, "guidance", ""):
            before.append(meaning.guidance)
        b_dir = Path(getattr(self, "base_dir", None) or Path.cwd())
        sensory_msgs, sensory_layers, _, _ = ON.run_s3_sensory_context(
            b_dir,
            b_dir / "media",
            media,
            query,
            kit=kit,
            emit=emit,
            turn_images=getattr(self, "turn_images", None),
        )
        for sm in sensory_msgs:
            if sm.get("content"):
                before.append(str(sm["content"]))
        layer_log.extend(sensory_layers)

        base_schemas = [t for t in N.STUDIO_TOOLS if only_tools is None or t["function"]["name"] in only_tools]
        extra_schemas = [t for t in extra_tools if only_tools is None or t["schema"]["function"]["name"] in only_tools]
        all_schemas = (base_schemas + [t["schema"] for t in extra_schemas]) if tools else []
        by_name = {t["function"]["name"]: t["function"] for t in all_schemas}
        local_runner = object.__new__(ON.OmniNative)
        local_runner.base_dir = b_dir
        local_runner.media_dir = b_dir / "media"
        local_runner.perms = getattr(self, "perms", None)
        local_runner.workdir = getattr(self, "workdir", None)
        local_runner.workname = getattr(self, "workname", "")
        local_runner.after_write = getattr(self, "after_write", None)
        local_runner.turn_images = getattr(self, "turn_images", None)
        local_runner.turn_cancel = cancel
        local_runner.turn_paths = list(meaning.paths) if meaning is not None else N.user_paths(prompt)
        local_runner.turn_hosts = N.typed_hosts(meaning.text if meaning is not None else prompt)
        local_runner.turn_unknown, local_runner.turn_ran, local_runner.turn_web, local_runner.turn_web_cap = set(), set(), 0, 8
        local_runner._py, local_runner._web = None, None
        local_runner._extra = {t["schema"]["function"]["name"]: t for t in extra_schemas} if tools else {}
        local_runner.tool_tasks = getattr(self, "tool_tasks", None)

        s3_pre_call, s3_extra_calls = (
            ON.plan_s3_pre_calls(local_runner, query, only_tools, by_name, meaning=meaning, past_msgs=hist or ())
            if (tools and all_schemas and only_tools and not (cancel and cancel.is_set()))
            else (None, [])
        )
        ctx_sources = [x.get("layer") for x in layer_log if isinstance(x, dict) and x.get("layer")]
        op_plan.complete_stage(
            "context",
            "Context established (" + (", ".join(dict.fromkeys(ctx_sources)) if ctx_sources else "AGENTS.md & conversation state") + ")",
        )
        op_plan.start_stage("direction", "Direction set (C2 HybridLink latent planning -> C3 11-level NCM)")
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution", "Executing C1 -> C2 -> C3 fused runtime…")
        rounds = []
        pre_text = ""
        if s3_pre_call is not None:
            pre_rounds, _, pre_text = ON.execute_s3_pre_calls(
                local_runner, s3_pre_call, s3_extra_calls, ask=ask, allow=allow, emit=emit
            )
            rounds.extend(pre_rounds)
            before.append(pre_text)

        if getattr(self, "c1_inference_only", False) and kit is not None and hasattr(kit, "c1_infer_states"):
            if hasattr(self, "_c1_server"):
                local_runner._c1_server = self._c1_server
            before_c1i = [b for b in before if b != pre_text]
            msgs_f = (
                [{"role": "system", "content": "You are Clover running on LFM3-Omni-2B (C1+C2+C3)."}]
                + list(hist or ())
                + [{"role": "user", "content": "\n\n".join(before_c1i + [prompt] + after)}]
            )
            planned_t = [rd["tool"] for rd in rounds if rd.get("ran")] or (list(only_tools) if only_tools is not None else [])
            return local_runner._c1_inference_turn(
                msgs_f, len(hist or ()), pre_text, query, kit, layers, emit, cancel,
                op_plan, rounds, layer_log, meaning, "LFM3-Omni-2B", "C1 trained LFM3-Omni-2B (C engine, 4-bit)", planned_t,
            )

        max_clover_cycles = 3
        cycle_history, answers = [], []
        cur_prompt = "\n\n".join(before + [prompt] + after)
        cur_hist = list(hist or ())
        reply = ""
        direct_plan = ON._direct_s3_or_concept_plan(query, s3_pre_call, s3_extra_calls, rounds)
        if direct_plan is not None:
            emit({"type": "plan", "round": 1, "stage": "c1_inference", "model": "LFM3-Omni-2B", "cycle": 1})
            emit({"type": "plan", "round": 1, "stage": "fused_c1_c2_c3", "model": "LFM3-Omni-2B", "cycle": 1})
            ans = {"reply": direct_plan, "input_tokens": max(1, len((query or "").split())), "generated_tokens": 0, "seconds": 0.0, "c1": "inference only"}
            answers.append(ans)
            emit({"type": "plan", "round": 1, "stage": "fused_done", "model": "LFM3-Omni-2B", "ms": 0, "cycle": 1})
            reply = direct_plan
            if "ontology" in layers and kit is not None:
                reply, changed = kit.grounding_filter(query, reply)
                layer_log.append({"layer": "ontology_grounding", "changed": changed, "cycle": 1})
            op_plan.complete_stage("execution")
            op_plan.start_stage("outcome", "Cycle 1 Outcome: verifying output against reality…")
            findings, c_meta = C.verify_turn_outcome(query, reply, rounds, kit=kit, layers=layers, meaning=meaning)
            c_meta["cycle"] = 1
            c_meta["findings"] = list(findings)
            c_meta["ok"] = not bool(findings)
            cycle_history.append(c_meta)
            if c_meta.get("compiler_info") is not None:
                layer_log.append(c_meta["compiler_info"])
        else:
            for attempt in range(max_clover_cycles):
                cycle_num = attempt + 1
                emit({"type": "plan", "round": len(answers) + 1, "stage": "c1_inference", "model": "LFM3-Omni-2B", "cycle": cycle_num})
                emit({"type": "plan", "round": len(answers) + 1, "stage": "fused_c1_c2_c3", "model": "LFM3-Omni-2B", "cycle": cycle_num})
                ans = self._ask(cur_prompt, cur_hist, max_new)
                answers.append(ans)
                emit({"type": "plan", "round": len(answers), "stage": "fused_done", "model": "LFM3-Omni-2B",
                      "ms": round((ans.get("seconds") or 0) * 1000), "cycle": cycle_num})
                reply = ans.get("reply") or ""
                if "ontology" in layers and kit is not None:
                    reply, changed = kit.grounding_filter(query, reply)
                    layer_log.append({"layer": "ontology_grounding", "changed": changed, "cycle": cycle_num})
                op_plan.complete_stage("execution")
                op_plan.start_stage("outcome", f"Cycle {cycle_num} Outcome: verifying output against reality…")
                findings, c_meta = C.verify_turn_outcome(query, reply, [], kit=kit, layers=layers, meaning=meaning)
                c_meta["cycle"] = cycle_num
                c_meta["findings"] = list(findings)
                c_meta["ok"] = not bool(findings)
                cycle_history.append(c_meta)
                if c_meta.get("compiler_info") is not None:
                    layer_log.append(c_meta["compiler_info"])
                if not findings or (cancel and cancel.is_set()):
                    break
                if attempt + 1 >= max_clover_cycles:
                    break
                op_plan.retry_from_context(cycle_num + 1, findings[0])
                emit({"type": "cycle", "step": "outcome_retry", "cycle": cycle_num + 1, "findings": findings})
                cur_hist = cur_hist + [{"role": "user", "content": cur_prompt}, {"role": "assistant", "content": reply}]
                cur_prompt = C.AUTO_CYCLE_FIX.format(findings="\n".join("- " + f for f in findings))

        if len(cycle_history) > 1 or (cycle_history and cycle_history[0].get("findings")):
            lesson_txt = (
                C.lesson_text(query, [f[:240] for f in (cycle_history[0].get("findings") or [])[:2]], "", passed=True)
                if len(cycle_history) > 1 and cycle_history[-1].get("ok")
                else ""
            )
            cyc_info = {
                "layer": "cycle",
                "trigger": "auto",
                "action": "corrected" if (len(cycle_history) > 1 and cycle_history[-1].get("ok")) else "unconfirmed",
                "cycles": len(cycle_history),
                "cycles_used": len(cycle_history),
                "repaired": len(cycle_history) > 1 and bool(cycle_history[-1].get("ok")),
                "first_findings": cycle_history[0].get("findings", []),
                "final_findings": cycle_history[-1].get("findings", []),
                "lesson": lesson_txt,
                "lesson_kind": "verified" if lesson_txt else "unconfirmed",
            }
            layer_log.append(cyc_info)
            emit(dict(cyc_info, type="layer"))
        last_c = cycle_history[-1] if cycle_history else {"ok": False, "checks": []}
        op_plan.complete_stage(
            "outcome",
            C.outcome_label(last_c["checks"], len(cycle_history)) if last_c["ok"]
            else f"Checked across {len(cycle_history)} cycles (unconfirmed)",
        )
        layer_log.append({"layer": "outcome", "checks": list(last_c["checks"]), "passed": bool(last_c["ok"])})
        op_plan.start_stage("growth", "Aligning epistemic state & recording learned knowledge…")
        gen_list = [
            dict(self._generation(a), **({"c1": "inference only"} if a.get("c1") == "inference only" else {}))
            for a in answers
        ]
        rnd_n = len(answers) or 1
        if kit is not None and not (cancel and cancel.is_set()):
            try:
                c1_state, c1_info = kit.c1_infer(
                    reply, stats=gen_list, model="LFM3-Omni-2B", layers_desc="C1 trained LFM3-Omni-2B (C engine, 4-bit)"
                )
                layer_log.append(c1_info)
                emit(dict(c1_info, type="layer"))
                emit({"type": "plan", "round": rnd_n, "stage": "c1_done", "ms": c1_info.get("ms")})
                planned_t = [rd["tool"] for rd in rounds if rd.get("ran")] or (list(only_tools) if only_tools is not None else [])
                emit({"type": "plan", "round": rnd_n, "stage": "c2_plan", "model": "HybridAgent"})
                c2_state, c2_info = kit.c2_plan_state(query, c1_state=c1_state, offered_tools=planned_t)
                layer_log.append(c2_info)
                emit(dict(c2_info, type="layer"))
                emit({"type": "plan", "round": rnd_n, "stage": "c2_done", "ms": c2_info.get("ms")})
                if hasattr(kit, "jepa_record_outcome"):
                    kit.jepa_record_outcome(c1_obs_state=c1_state, is_baseline=(len(cycle_history) <= 1))
                if "ncm" in layers and (hasattr(kit, "c3_decode") or hasattr(kit, "ncm_render")):
                    emit({"type": "plan", "round": rnd_n, "stage": "ncm_decode", "model": "11-Level NCM"})
                    reply, ncm_info = kit.c3_decode(c2_state) if hasattr(kit, "c3_decode") else kit.ncm_render(reply)
                    layer_log.append(ncm_info)
                    emit(dict(ncm_info, type="layer"))
                    emit({"type": "plan", "round": rnd_n, "stage": "ncm_done",
                          "ms": round((ncm_info.get("s") or 0) * 1000)})
            except Exception:
                pass
        op_plan.complete_stage("growth", "Growth preserved · context updated")
        if on_event:
            on_event({"type": "token", "round": rnd_n, "kind": "answer", "text": reply})
        return {"reply": reply, "tool_calls": rounds, "rounds": len(answers),
                "generation": gen_list,
                "layers": layer_log + [self._fused_layer(answers)],
                "todo_list": op_plan.todos}

    def retry_turn(self, history, question, previous, feedback, *, media=(), layers=(), kit=None, on_event=None,
                   cancel=None, max_new=2048, context="", **kw):
        """The Clover cycle (clover_cycle.retry_cycle) on this runtime, as omni_native.retry_turn without its tools (no
        web search, no Python solution, no compiler check): each rewrite and each independent solution is one call.
        The first regression run had no cycle on this engine ("That's not right, try again" went as a new question)."""
        import clover_cycle as C
        import omni_native as ON
        emit = on_event or (lambda e: None)
        op_plan = C.OperatingCyclePlan(emit, model_name="LFM3-Omni-2B (C1+C2+C3)", query=question, offered_tools=())
        self.active_op_plan = op_plan
        op_plan.start_stage("context", f"Re-establishing Context from user feedback: {str(feedback)[:80]}")
        blocks, layer_log = self._context(layers, kit, question)  # Context gathered again for the question
        if media:
            sensory_msgs, sensory_layers, _, _ = ON.run_s3_sensory_context(
                self.base_dir, self.base_dir / "media", media, question, kit=kit, emit=emit
            )
            for sm in sensory_msgs:
                if sm.get("content"):
                    blocks.append(str(sm["content"]))
            layer_log.extend(sensory_layers)
        notes = ("\n\n" + "\n\n".join(blocks)) if blocks else ""
        op_plan.complete_stage("context")
        op_plan.start_stage("direction", "Setting corrected Direction for independent solve")
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution", "Running C1 -> C2 -> C3 retry cycle…")
        convo = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
        start, answers = len(convo), []

        def rewrite(instruction):  # the runtime sees the last messages of the conversation and this retry's attempts
            if cancel and cancel.is_set():
                return previous
            text = instruction + (notes if len(convo) == start else "")
            ans = self._ask(text, convo, max_new)
            answers.append(ans)
            new = (ans.get("reply") or "").strip() or previous
            convo.extend([{"role": "user", "content": text}, {"role": "assistant", "content": new}])
            return new

        def solve(k, note=None):  # the question alone, without the earlier answer
            if cancel and cancel.is_set():
                return ""
            ans = self._ask(C.second_solve_prompt(context + question + notes, question, k, note), [], max_new)
            answers.append(ans)
            return ans.get("reply") or ""

        answer, info = C.retry_cycle(question, previous, feedback, rewrite, solve=solve, emit=emit, used=[])
        info["tools"] = []
        layer_log.append(info)
        if "ontology" in layers and kit is not None:
            answer, changed = kit.grounding_filter(question, answer)
            layer_log.append({"layer": "ontology_grounding", "changed": changed})
        op_plan.complete_stage("execution")
        op_plan.start_stage("outcome", f"Outcome action: {info.get('action', 'corrected')}")
        op_plan.complete_stage("outcome")
        op_plan.start_stage("growth", "Growth preserved")
        op_plan.complete_stage("growth")
        cancelled = bool(cancel and cancel.is_set())
        reply = "(Stopped.)" if cancelled else answer
        emit({"type": "token", "round": 1, "kind": "answer", "text": reply})
        return {"reply": reply, "tool_calls": [], "rounds": len(answers),
                "generation": [self._generation(a) for a in answers], "reasoning": "", "cancelled": cancelled,
                "layers": layer_log + [self._fused_layer(answers)],
                "todo_list": op_plan.todos}

    def vscode_step(self, user_prompt, ws_info, prior_turns, tool_rounds, tools, **kw):
        res = self.run_turn([], user_prompt)
        return {"content": res["reply"], "tool_call": None, "trace": {"engine": "fused"}}

    def close(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
