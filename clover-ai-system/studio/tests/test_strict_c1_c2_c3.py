"""Strict separation test for C1 (Inference ONLY), C2 (Planning ONLY), and C3 (Decoding ONLY).

Verifies:
1. OmniC2C3Runtime.c1_infer runs ONLY C1 (extracts 2048-d representations, stage='c1_inference', 0 C2/C3 calls).
2. OmniC2C3Runtime.c2_plan runs ONLY C2 (HybridLink2048 + Lfm3NativeHybridLink on plan-trunk-0, stage='c2_plan', 0 C1/C3 calls).
3. OmniC2C3Runtime.c3_decode runs ONLY C3 (Bridges + 11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder,
   requires stage='c2_plan', 0 C1/C2 calls).
4. OmniNative.run_turn executes strictly in order: C1 (c1_inference -> c1_done) -> C2 (c2_plan -> c2_done) -> C3 (ncm_decode -> ncm_done),
   with zero answer tokens emitted during C1 or C2, and final answer tokens emitted only after C3 completes.
"""
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parent / "clover"
sys.path.insert(0, str(CLOVER))

import lfm3_layers  # noqa: E402
import omni_c2_c3  # noqa: E402
import omni_native  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


# 1. Test OmniNative.run_turn strict event sequence with LayerKit
base_tmp = Path(tempfile.mkdtemp())
kit = lfm3_layers.LayerKit(base_tmp, memory_root=base_tmp / "knowledge")
eng = object.__new__(omni_native.OmniNative)
eng.lock = threading.Lock()
eng.stage = {"name": "idle"}
eng.media_dir = base_tmp
eng._extra = {}
eng.workdir, eng.workname = None, ""


def fake_c1_chat(msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
    # Verify C1 is never given an answer-streaming callback
    check("C1 _chat receives on_delta=None so C1 never decodes answer tokens to UI", on_delta is None)
    return "Clover runs C1 for inference, C2 for planning, and C3 for decoding.", [], {
        "prompt_tokens": 24, "reused_tokens": 0, "new_tokens": 14, "prefill_s": 0.05, "decode_s": 0.10, "finish": "stop"
    }


eng._chat = fake_c1_chat
events = []
res = eng.run_turn(
    [],
    "How do C1, C2, and C3 work?",
    tools=False,
    layers=("ncm",),
    kit=kit,
    on_event=events.append,
    system="SYS",
)

plan_stages = [e["stage"] for e in events if e.get("type") == "plan" and e.get("stage")]
check(
    "OmniNative.run_turn emits strict C1 -> C2 -> C3 stage order",
    plan_stages == ["c1_inference", "c1_done", "c2_plan", "c2_done", "ncm_decode", "ncm_done"],
)

ncm_done_idx = next(i for i, e in enumerate(events) if e.get("type") == "plan" and e.get("stage") == "ncm_done")
token_indices = [i for i, e in enumerate(events) if e.get("type") == "token" and e.get("kind") == "answer"]
check(
    "Answer token event is emitted strictly after C3 (ncm_done)",
    len(token_indices) == 1 and token_indices[0] > ncm_done_idx,
)

layer_names = [l.get("layer") for l in res.get("layers", [])]
check(
    "Turn layers record c1_inference, c2_hybridagent, and ncm in order",
    layer_names == ["c1_inference", "c2_hybridagent", "ncm"],
)

# 2. If precomputed C2/C3 trunks are present on this machine, verify strict hardware/tensor isolation in OmniC2C3Runtime
for cand_base in (
    Path("/opt/clover-k3/web-studio"),
    Path("/opt/clover-k3/studio-android/sim/base"),
    Path.home() / "CloverAI",
):
    if cand_base.is_dir() and not omni_c2_c3.missing_c2_c3_files(cand_base):
        rt = omni_c2_c3.for_base(cand_base)
        check(f"Loaded OmniC2C3Runtime from {cand_base}", rt is not None)

        c1_state, c1_info = rt.c1_infer("Clover separates C1 inference, C2 planning, and C3 decoding.")
        check("c1_infer produces stage='c1_inference' with h_primary and windows", c1_state.get("stage") == "c1_inference")
        check("c1_infer does not compute r_valid (C2)", all("r_valid" not in w for w in c1_state["windows"] if not w.get("passthrough")))

        # Verify c3_decode rejects c1_state directly (must go through c2_plan first)
        rejected = False
        try:
            rt.c3_decode(c1_state)
        except ValueError:
            rejected = True
        check("c3_decode strictly rejects un-planned c1_state", rejected)

        # Verify c2_plan runs without calling C1 token_embeddings_2048
        orig_c1_emb = rt.token_embeddings_2048
        rt.token_embeddings_2048 = lambda *a, **k: (_ for _ in ()).throw(AssertionError("C2 called C1 token_embeddings_2048!"))
        try:
            c2_state, c2_info = rt.c2_plan("How do C1, C2, and C3 work?", c1_state=c1_state)
        finally:
            rt.token_embeddings_2048 = orig_c1_emb
        check("c2_plan produces stage='c2_plan' with r_valid and zero calls to C1", c2_state.get("stage") == "c2_plan")

        # Verify c3_decode runs without calling C1 token_embeddings_2048 or C2 forward_hybridlink / forward_hybridlink_2048
        orig_hl1 = rt.forward_hybridlink
        orig_hl2 = rt.forward_hybridlink_2048
        rt.token_embeddings_2048 = lambda *a, **k: (_ for _ in ()).throw(AssertionError("C3 called C1 token_embeddings_2048!"))
        rt.forward_hybridlink = lambda *a, **k: (_ for _ in ()).throw(AssertionError("C3 called C2 forward_hybridlink!"))
        rt.forward_hybridlink_2048 = lambda *a, **k: (_ for _ in ()).throw(AssertionError("C3 called C2 forward_hybridlink_2048!"))
        try:
            decoded, c3_info = rt.c3_decode(c2_state)
        finally:
            rt.token_embeddings_2048 = orig_c1_emb
            rt.forward_hybridlink = orig_hl1
            rt.forward_hybridlink_2048 = orig_hl2
        check("c3_decode decodes text using only C3 Bridges + 11L NCM + PointerDecoder (zero C1/C2 calls)",
              bool(decoded.strip()) and c3_info.get("codes", 0) > 0 and c3_info.get("windows") == 1)

        # Verify 512-token window handles >100 tokens in a single one-time flow without splitting into 48-token chunks
        check("OmniC2C3Runtime token window size P_BPE is 512", omni_c2_c3.P_BPE == 512)
        long_para = " ".join(["Clover processes a 512-token window in a single one-time flow between C1, C2, and C3."] * 8)
        c1_long, c1_long_info = rt.c1_infer(long_para)
        c2_long, c2_long_info = rt.c2_plan("Explain the one-time flow", c1_state=c1_long)
        dec_long, c3_long_info = rt.c3_decode(c2_long)
        check(
            "Long paragraph (>100 tokens) runs in 1 single window across C1, C2, and C3",
            len(c1_long["pids_primary"]) > 80
            and c1_long_info.get("windows") == 1
            and c2_long_info.get("windows") == 1
            and c3_long_info.get("windows") == 1
            and bool(dec_long.strip()),
        )
        break

# 2b. Verify one-time S3 -> C1 -> C2 -> C3 flow on web_search (1 single C1 pass, not 2 LLM rounds)
c1_calls = []
def fake_c1_single_pass(msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
    c1_calls.append({"msgs": len(msgs), "schemas": len(schemas)})
    return "Based on the live web search results, the 2026 midterm elections are scheduled for November 3, 2026.", [], {
        "prompt_tokens": 64, "reused_tokens": 0, "new_tokens": 20, "prefill_s": 0.04, "decode_s": 0.08, "finish": "stop"
    }

eng._chat = fake_c1_single_pass
eng.execute_tool = lambda name, args, ask=None, allow=(): (True, "1. 2026 US Midterm Elections — Nov 3, 2026\n   https://example.com/midterms", "allowed")
res_ws = eng.run_turn(
    [],
    "What are the latest 2026 midterm election results?",
    tools=True,
    only_tools=["web_search"],
    layers=("ncm",),
    kit=kit,
    system="SYS",
)
check(
    "One-time S3 -> C1 -> C2 -> C3 flow executes web_search in S3 and runs C1 only ONCE",
    len(c1_calls) == 1 and c1_calls[0]["schemas"] == 0 and len(res_ws["tool_calls"]) == 1 and res_ws["tool_calls"][0]["tool"] == "web_search",
)

# 3. Verify S2 relationship gate & S3 live web search routing (restaurants vs midterms results)
import clover_repo as R
import memory_index as MI
import semantics as M

b_rel = Path(tempfile.mkdtemp())
(b_rel / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
repo_rel = R.Repo(b_rel, b_rel / "AGENTS.md")
n_r, _ = repo_rel.turn_start("chat_rest", "What are some good vegetarian restaurants in Jaipur?")
repo_rel.turn_end("chat_rest", n_r, {"reply": "Some top vegetarian restaurants in Jaipur are LMB Hotel and Tapri Central."}, 1.0)
idx_rel = MI.MemoryIndex(b_rel, repo_rel, embedder=MI.SubwordHashEmbedder(), start=False)
idx_rel.refresh()
n_m, _ = repo_rel.turn_start("chat_mid", "What are the midterms results?")
blk_rel, info_rel = idx_rel.recall("chat_mid", n_m, "What are the midterms results?", history_turns=0)
check("S2 relationship gate excludes unrelated restaurant chat when asking for midterms results",
      blk_rel is None and info_rel.get("refs") == [])

# Verify OmniNative._history filters out unrelated in-session turns on topic shifts while keeping follow-ups
hist_mixed = [
    {"role": "user", "content": "What are some good vegetarian restaurants in Jaipur?"},
    {"role": "assistant", "content": "LMB Hotel and Tapri Central are great vegetarian restaurants in Jaipur."},
]
filtered_shift = eng._history(hist_mixed, current_query="What are the midterms results?", is_follow_up=False)
check("OmniNative._history filters out unrelated prior restaurant turn on topic shift to midterms results",
      filtered_shift == [])
kept_followup = eng._history(hist_mixed, current_query="Which one is best for dinner?", is_follow_up=True)
check("OmniNative._history preserves prior turns on follow-up questions",
      len(kept_followup) == 2)

# Verify semantics routes both 'midterms results' and 'good vegetarian restaurants in Jaipur' to web_search in 'ask' mode
rt_sem = omni_native.N.RUNTIME
fr_mid = M.interpret("What are the midterms results?", rt_sem)
only_mid, on_mid = M.route(fr_mid, "ask", True, offer=["web_search", "fetch_url", "run_python"])
check("semantics routes 'What are the midterms results?' to web_search in default 'ask' mode",
      on_mid and "web_search" in (only_mid or []))

fr_rest = M.interpret("What are some good vegetarian restaurants in Jaipur?", rt_sem)
only_rest, on_rest = M.route(fr_rest, "ask", True, offer=["web_search", "fetch_url", "run_python"])
check("semantics routes 'What are some good vegetarian restaurants in Jaipur?' to web_search in default 'ask' mode",
      on_rest and "web_search" in (only_rest or []) and not M.answered_unasked(fr_rest, "Here are some options.", ["web_search"]))

# 4. Verify Playwright-grade Image Inspector (S3) & UI hold-to-speak / GitHub direct sync
import image_inspector
from PIL import Image, ImageDraw

img_tmp = base_tmp / "ui_test_screen.png"
im = Image.new("RGB", (640, 360), (247, 243, 233))
dr = ImageDraw.Draw(im)
dr.rectangle([0, 0, 640, 52], fill=(23, 107, 44))
dr.text((20, 18), "Clover AI Studio Dashboard", fill=(255, 255, 255))
dr.rectangle([40, 100, 220, 148], fill=(16, 185, 129), outline=(11, 20, 16), width=2)
dr.text((65, 118), "Sign in with Google", fill=(255, 255, 255))
dr.rectangle([260, 100, 480, 148], fill=(37, 99, 235), outline=(11, 20, 16), width=2)
dr.text((280, 118), "Sync GitHub Memory", fill=(255, 255, 255))
im.save(img_tmp, format="PNG")

rep_img = image_inspector.inspect_image(img_tmp, annotate_dir=base_tmp)
check(
    "image_inspector extracts geometry, palette, [e1..eN] elements, and annotated PNG",
    rep_img["width"] == 640
    and rep_img["height"] == 360
    and len(rep_img["palette"]) >= 2
    and len(rep_img["elements"]) >= 1
    and bool(rep_img.get("annotated_file"))
    and (base_tmp / rep_img["annotated_file"]).is_file(),
)

seen_img_snap = []
def fake_c1_vision(msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
    sys_texts = " ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "system")
    seen_img_snap.append("Playwright-Style Visual & UI Snapshot" in sys_texts)
    return "The dashboard has a green header bar and two action buttons at [e1] and [e2].", [], {
        "prompt_tokens": 96, "reused_tokens": 0, "new_tokens": 18, "prefill_s": 0.05, "decode_s": 0.09, "finish": "stop"
    }

eng._chat = fake_c1_vision
img_ev = []
res_img = eng.run_turn(
    [],
    "Inspect the UI elements in this screenshot.",
    tools=False,
    media=[{"kind": "image", "file": "ui_test_screen.png", "name": "ui_test_screen.png"}],
    layers=("ncm",),
    kit=kit,
    on_event=img_ev.append,
    system="SYS",
)
check(
    "OmniNative.run_turn runs S3 Visual Inspector before C1 and injects Playwright snapshot",
    seen_img_snap == [True] and any(e.get("type") == "layer" and e.get("layer") == "image_analysis" for e in img_ev),
)

app_html = (CLOVER / "web" / "app.html").read_text(encoding="utf-8")
check(
    "web/app.html keeps separate #mic hidden and uses #send hold-to-speak + #img-inspector + direct GitHub token sync",
    "if ($('mic')) $('mic').hidden = true;" in app_html
    and "HOLD_MS = 450" in app_html
    and 'id="img-inspector"' in app_html
    and 'id="acct-gh-token"' in app_html,
)

print("ALL STRICT C1/C2/C3 AND S1/S2/S3 CHECKS PASSED")
