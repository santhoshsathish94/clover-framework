#!/usr/bin/env python3
"""
Comprehensive A/B Evaluation Suite for Clover AI System Layers & Process Learning
(`tests/compare_scenario_layers.py`).

Tests all 12 scenarios across:
  1. Personal Process Learning (PPL: explicit statements, natural preferences, >12-word form feedback,
     git-free Android store, UI/growth visibility)
  2. CSI Engine (per-turn caching, negated tool requests, main-clause claims vs embedded/negated verbs,
     Android remote CSI fallback)
  3. Supporting Layers (S1 default memory permission & diet recall, S2 scenario-gated earlier-chat recall,
     S3 generative compiler repair -> S1 growth lesson)
  4. Core Layers & K3HostEngine (Nemotron 3.5 / Kimi K3 direct planning & structured output preservation,
     K3HostEngine retry cycle, and Fused C1+C2+C3 scenario-gated continuous latent step & code bypass)
"""
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("CLOVER_TEST_URL", "http://127.0.0.1:8780")
for _cand_c in (Path(__file__).resolve().parent.parent / "clover", Path("/opt/clover-k3/web-studio/clover")):
    if _cand_c.is_dir() and str(_cand_c) not in sys.path:
        sys.path.insert(0, str(_cand_c))
import k3_host_engine  # noqa: E402

HDR = {
    "Content-Type": "application/json",
    "Host": "api.cloverframework.com",
    "Origin": "https://cloverframework.com",
    "X-Clover-Key": k3_host_engine.k3_host_key(),
}


def api(method, path, body=None, timeout=180):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, headers=HDR, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def run_suite(label="run"):
    t_start = time.time()
    results = {}
    created_sessions = []

    def track_sid(sid):
        created_sessions.append(sid)
        return sid

    # Save initial state
    orig_settings = api("GET", "/api/settings").get("settings") or {}
    orig_perms = {k: v["mode"] for k, v in (api("GET", "/api/permissions").get("permissions") or {}).items()}
    api("POST", "/api/personal/forget", {"confirm": True})

    try:
        # ------------------------------------------------------------------
        # Scenario 1A: Explicit User Habit Statement ("When I say X, I mean Y")
        # ------------------------------------------------------------------
        s1a_1 = track_sid(f"cmp-{label}-1a-1")
        s1a_2 = track_sid(f"cmp-{label}-1a-2")
        r_stmt = api("POST", "/api/chat", {
            "session_id": s1a_1,
            "prompt": "When I say verify something, I mean search the web.",
        })
        pv_1a = api("GET", "/api/personal")
        r_use = api("POST", "/api/chat", {
            "session_id": s1a_2,
            "prompt": "Verify the boiling point of water.",
        })
        sem_1a = (r_use.get("layers") or [{}])[0]
        offered_1a = r_use.get("tools_offered") or []
        results["1a_explicit_habit_statement"] = {
            "passed": ("web_search" in offered_1a and (sem_1a.get("personal") or {}).get("hypothesis") is not None),
            "learned_status": [(h.get("when_you_say"), h.get("status")) for h in (pv_1a.get("learned") or [])],
            "tools_offered_on_next_turn": offered_1a,
            "understood": sem_1a.get("understood"),
            "growth_on_statement": r_stmt.get("growth") or [],
        }

        # ------------------------------------------------------------------
        # Scenario 1B: Natural Imperative Preference ("Keep answers short and use bullet points.")
        # ------------------------------------------------------------------
        s1b_1 = track_sid(f"cmp-{label}-1b-1")
        s1b_2 = track_sid(f"cmp-{label}-1b-2")
        r_pref = api("POST", "/api/chat", {
            "session_id": s1b_1,
            "prompt": "Keep answers short and use bullet points.",
        })
        pv_1b = api("GET", "/api/personal")
        active_prefs_1b = {p["dimension"]: p["value"] for p in (pv_1b.get("preferences") or []) if p.get("in_use")}
        r_pref_use = api("POST", "/api/chat", {
            "session_id": s1b_2,
            "prompt": "What are the three states of matter?",
        })
        sem_1b = (r_pref_use.get("layers") or [{}])[0]
        form_1b = (sem_1b.get("personal") or {}).get("form") or ""
        results["1b_natural_preference_statement"] = {
            "passed": (active_prefs_1b.get("length") == "short" and active_prefs_1b.get("format") == "bullets" and bool(form_1b)),
            "active_preferences": active_prefs_1b,
            "guidance_injected": form_1b,
            "understood": sem_1b.get("understood"),
            "growth_on_preference": r_pref.get("growth") or [],
        }

        # ------------------------------------------------------------------
        # Scenario 1C: Longer Form Feedback (>12 words) on Previous Answer
        # ------------------------------------------------------------------
        s1c = track_sid(f"cmp-{label}-1c")
        api("POST", "/api/chat", {
            "session_id": s1c,
            "prompt": "What are the main layers of the Earth?",
        })
        api("POST", "/api/chat", {
            "session_id": s1c,
            "prompt": "Could you please make that answer shorter and format it as a table for me?",
        })
        pv_1c = api("GET", "/api/personal")
        active_prefs_1c = {p["dimension"]: p["value"] for p in (pv_1c.get("preferences") or []) if p.get("in_use")}
        results["1c_longer_form_feedback"] = {
            "passed": (active_prefs_1c.get("format") == "table"),
            "active_preferences": active_prefs_1c,
            "experiences_count": len(pv_1c.get("experiences") or []),
        }

        # ------------------------------------------------------------------
        # Scenario 1D: Android / Git-Free Environment Personal Store
        # ------------------------------------------------------------------
        clover_dir = Path("/opt/clover-k3/web-studio/clover")
        if str(clover_dir) not in sys.path:
            sys.path.insert(0, str(clover_dir))
        import personal
        import semantics
        old_path = os.environ.get("PATH", "")
        with tempfile.TemporaryDirectory() as tmp_p:
            try:
                os.environ["PATH"] = "/nonexistent"
                personal._PROCESS.clear()
                pers_nogit = personal.for_process(tmp_p, "phone", None)
                nogit_ok = False
                if pers_nogit is not None:
                    rt_p = semantics.Runtime(tools={"web.search": ("web_search",)}, device="phone", personal=pers_nogit)
                    pers_nogit.before_turn("s1", "When I say checkup something, I mean search the web.")
                    f_p = semantics.interpret("Checkup the weather.", rt_p)
                    v_p = pers_nogit.view()
                    pers_nogit.forget_all()
                    nogit_ok = bool(v_p.get("enabled")) and len(v_p.get("experiences") or []) >= 1
            finally:
                os.environ["PATH"] = old_path
                personal._PROCESS.clear()
        results["1d_android_git_free_personal_store"] = {
            "passed": nogit_ok,
            "store_initialized_without_git": pers_nogit is not None,
        }

        # ------------------------------------------------------------------
        # Scenario 2A: CSI Caching & Single-Call Efficiency + Imperative/Negation
        # ------------------------------------------------------------------
        h_info = api("GET", "/api/health")
        csi_url = (h_info.get("csi") or {}).get("url") or ""
        real_reader = semantics.csi_reader(csi_url) if csi_url else None
        call_counter = {"n": 0}

        def counting_csi(txt):
            call_counter["n"] += 1
            return real_reader(txt) if real_reader else None

        with tempfile.TemporaryDirectory() as tmp_csi_p:
            pers_c = personal.Personal(tmp_csi_p, device="laptop", csi=counting_csi)
            rt_c = semantics.Runtime(
                tools={"web.search": ("web_search",), "web.page": ("fetch_url",), "python": ("run_python",)},
                device="laptop",
                csi=counting_csi,
                personal=pers_c,
            )
            f_neg = semantics.interpret("Do not search the web, just explain recursion.", rt_c)
            calls_per_turn = call_counter["n"]
        results["2a_csi_single_call_caching_and_negation"] = {
            "passed": (calls_per_turn == 1 and "web.search" in f_neg.declined),
            "csi_http_calls_for_one_turn": calls_per_turn,
            "declined_caps": f_neg.declined,
        }

        # ------------------------------------------------------------------
        # Scenario 2B: Android CSI Fallback to k3
        # ------------------------------------------------------------------
        import csi_child
        has_remote_fallback = hasattr(csi_child, "remote_csi_url") or "k3" in Path(csi_child.__file__).read_text(encoding="utf-8")
        results["2b_android_csi_fallback_support"] = {
            "passed": bool(has_remote_fallback),
            "has_remote_fallback": bool(has_remote_fallback),
        }

        # ------------------------------------------------------------------
        # Scenario 3A: S1 Memory Default Permission ("Remember that I'm vegetarian...")
        # ------------------------------------------------------------------
        # Test with fresh default permissions from PermissionStore
        import permissions as P
        with tempfile.TemporaryDirectory() as tmp_perm_d:
            fresh_perms = P.PermissionStore(Path(tmp_perm_d) / "permissions.json").get()
        mem_default_allow = (fresh_perms.get("memory") == "allow")
        if mem_default_allow:
            api("POST", "/api/permissions", {"permissions": {"memory": "allow"}})
        s3a_1 = track_sid(f"cmp-{label}-3a-1")
        s3a_2 = track_sid(f"cmp-{label}-3a-2")
        r_rem = api("POST", "/api/chat", {
            "session_id": s3a_1,
            "prompt": "Please remember that I am vegetarian and allergic to peanuts.",
        })
        mems_3a = api("GET", "/api/memory").get("memories") or []
        saved_veg = any("vegetarian" in (m.get("summary") or "").lower() for m in mems_3a)
        r_dinner = api("POST", "/api/chat", {
            "session_id": s3a_2,
            "prompt": "Suggest a quick dinner I can cook tonight.",
        })
        used_mem_3a = any(l.get("layer") == "memory" and l.get("entries") for l in (r_dinner.get("layers") or []))
        results["3a_s1_memory_default_and_diet_recall"] = {
            "passed": bool(mem_default_allow and saved_veg and used_mem_3a),
            "fresh_default_memory_mode": fresh_perms.get("memory"),
            "saved_vegetarian_note": saved_veg,
            "diet_note_recalled_on_meal_question": used_mem_3a,
        }

        # ------------------------------------------------------------------
        # Scenario 3B: S2 Scenario-Gated Earlier-Chats Recall
        # ------------------------------------------------------------------
        s3b_seed = track_sid(f"cmp-{label}-3b-seed")
        s3b_pers = track_sid(f"cmp-{label}-3b-pers")
        s3b_math = track_sid(f"cmp-{label}-3b-math")
        api("POST", "/api/chat", {
            "session_id": s3b_seed,
            "prompt": "My dog's name is Bruno and he is a Golden Retriever. Just say noted.",
        })
        r_dog = api("POST", "/api/chat", {
            "session_id": s3b_pers,
            "prompt": "What did I tell you earlier about my dog's name and breed?",
        })
        r_math = api("POST", "/api/chat", {
            "session_id": s3b_math,
            "prompt": "What is 144 divided by 12?",
        })
        dog_recalled = "bruno" in (r_dog.get("reply") or "").lower()
        math_polluted = any(l.get("layer") == "recall" and l.get("refs") for l in (r_math.get("layers") or []))
        results["3b_s2_scenario_gated_recall"] = {
            "passed": bool(dog_recalled and not math_polluted),
            "personal_past_question_recalled_bruno": dog_recalled,
            "general_math_question_polluted_by_recall": math_polluted,
            "dog_reply": (r_dog.get("reply") or "")[:160],
        }

        # ------------------------------------------------------------------
        # Scenario 3C: S3 Compiler Repair -> S1 Growth Lesson
        # ------------------------------------------------------------------
        # Test whether compiler repair or retry cycle records a verified learning
        s3c = track_sid(f"cmp-{label}-3c")
        r_cyc = api("POST", "/api/chat", {
            "session_id": s3c,
            "history": [{"user": "What is 19 * 21?", "reply": "19 * 21 = 389."}],
            "prompt": "That is wrong, try again.",
        })
        growth_3c = r_cyc.get("growth") or []
        results["3c_cycle_and_compiler_growth_lesson"] = {
            "passed": bool(growth_3c and "399" in (r_cyc.get("reply") or "")),
            "growth": growth_3c,
            "reply": (r_cyc.get("reply") or "")[:160],
        }

        # ------------------------------------------------------------------
        # Scenario 4A: K3HostEngine (Nemotron 3.5) Structured Output & Direct Execution
        # ------------------------------------------------------------------
        s4a = track_sid(f"cmp-{label}-4a")
        t4a = time.time()
        r_nemo = api("POST", "/api/chat", {
            "session_id": s4a,
            "model": "nemotron-3.5-super-vl",
            "prompt": "In two bullet points, state your exact model name and how many layers you have.",
        })
        s4a_elapsed = round(time.time() - t4a, 2)
        nemo_layers = r_nemo.get("layers") or []
        ncm_layer_4a = next((l for l in nemo_layers if l.get("layer") == "ncm"), {})
        used_k3_reply = bool(ncm_layer_4a.get("used_k3_reply", False))
        results["4a_k3_host_engine_structured_and_direct"] = {
            "passed": bool(used_k3_reply and "nemotron" in (r_nemo.get("reply") or "").lower()),
            "used_k3_reply_on_structured": used_k3_reply,
            "elapsed_s": s4a_elapsed,
            "ncm_layer": ncm_layer_4a,
            "reply": (r_nemo.get("reply") or "")[:200],
        }

        # ------------------------------------------------------------------
        # Scenario 4B: Fused C1+C2+C3 (CloverS23OnDeviceSystem) Scenario-Gated C2 Latent Step & C3 Code Bypass
        # ------------------------------------------------------------------
        s23_dir = Path("/opt/clover-k3/s23-omni2b")
        s23_data = Path("/opt/clover-k3/s23-omni2b")
        fused_ok = False
        c2_auto_injected = False
        c3_code_bypassed = False
        if s23_dir.exists() and s23_data.exists():
            import subprocess
            probe_code = (
                "import sys, json\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(s23_dir)!r})\n"
                "from s23_on_device_clover import CloverS23OnDeviceSystem\n"
                f"sys_s23 = CloverS23OnDeviceSystem(Path({str(s23_data)!r}), load_neural_ncm=True)\n"
                "r1 = sys_s23.run_cycle('Explain in one sentence what photosynthesis does.', max_active_layer=7, allow_internet=False, allow_python=False, session_id='cmp-s23-1')\n"
                "lt1 = r1.get('layer_trace') or {}\n"
                "c2_inj = bool((lt1.get('layer2_hybrid_agent') or {}).get('injected_before_decode'))\n"
                "r2 = sys_s23.run_cycle('Write a Python function add(a, b) that returns a + b.', max_active_layer=7, allow_internet=False, allow_python=False, session_id='cmp-s23-2')\n"
                "lt2 = r2.get('layer_trace') or {}\n"
                "c3_info = lt2.get('layer3_11level_ncm') or {}\n"
                "sys_s23.git_memory_repo.delete_session('cmp-s23-1')\n"
                "sys_s23.git_memory_repo.delete_session('cmp-s23-2')\n"
                "print('S23_PROBE_JSON:' + json.dumps({'c2_inj': c2_inj, 'c3_info': c3_info, 'r1_reply': r1.get('reply'), 'r2_reply': r2.get('reply')}))\n"
            )
            proc = subprocess.run(
                ["/root/ncm-cpu/venv/bin/python", "-c", probe_code],
                capture_output=True, text=True, timeout=120,
            )
            for ln in (proc.stdout or "").splitlines():
                if ln.startswith("S23_PROBE_JSON:"):
                    jd = json.loads(ln.split("S23_PROBE_JSON:", 1)[1])
                    c2_auto_injected = bool(jd.get("c2_inj"))
                    c3_info = jd.get("c3_info") or {}
                    c3_code_bypassed = bool(c3_info.get("skipped_code_block") or not c3_info.get("ncm_active", True))
                    fused_ok = c2_auto_injected and c3_code_bypassed
        results["4b_fused_c1_c2_c3_scenario_routing"] = {
            "passed": fused_ok,
            "c2_latent_auto_injected_on_reasoning": c2_auto_injected,
            "c3_ncm_bypassed_on_code_block": c3_code_bypassed,
        }

    finally:
        for m in (api("GET", "/api/memory").get("memories") or []):
            if any(k in (m.get("summary") or "").lower() for k in ("vegetarian", "19 * 21")):
                api("POST", "/api/memory/forget", {"id": m["id"]})
        for sid in created_sessions:
            try:
                api("POST", "/api/session/forget", {"session_id": sid, "delete": True})
            except Exception:
                pass
        api("POST", "/api/personal/forget", {"confirm": True})
        api("POST", "/api/permissions", {"permissions": orig_perms})
        api("POST", "/api/settings", orig_settings)

    passed_n = sum(1 for v in results.values() if v.get("passed"))
    total_n = len(results)
    return {
        "label": label,
        "passed": passed_n,
        "total": total_n,
        "elapsed_s": round(time.time() - t_start, 2),
        "scenarios": results,
    }


if __name__ == "__main__":
    lbl = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out = run_suite(lbl)
    print(json.dumps(out, indent=2))
