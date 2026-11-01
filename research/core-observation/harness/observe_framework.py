"""Starts the framework runtime (clover-ai-system/mobile/s23_on_device_clover.py) with its layers observed.
Usage: python observe_framework.py --framework <clone>/clover-ai-system --base <base-dir> --port 8791 --trace <file>"""
import argparse
import sys
from pathlib import Path

import tracer as T

ap = argparse.ArgumentParser()
ap.add_argument("--framework", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--port", type=int, default=8791)
ap.add_argument("--trace", required=True)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.framework) / "mobile"))
import s23_on_device_clover as F  # noqa: E402  (its own sys.path set-up runs here)
import semantics  # noqa: E402

T.open_trace(a.trace)
T.mark_turns(("/api/chat",))


def ids_text(items):
    out = {}
    for k, v in items:
        if k in ("prompt_ids", "new_ids") and isinstance(v, (list, tuple)):
            dec = T.decoders.get("ids")
            out[k] = {"n": len(v), "text": T.brief(dec(list(v)), limit=8000) if dec else None}
        else:
            out[k] = T.brief(v)
    return out


def ids_result(res):
    """(ids, ...) tuples from the engine: the generated ids as text."""
    if isinstance(res, tuple) and res and isinstance(res[0], (list, tuple)):
        dec = T.decoders.get("ids")
        head = {"n_ids": len(res[0]), "text": T.brief(dec(list(res[0])), limit=6000) if dec else None}
        return {"ids": head, "rest": [T.brief(x) for x in res[1:]]}
    return T.brief(res)


# Semantics (common file)
for fn in ("interpret", "route", "align", "social", "non_answer"):
    T.wrap(semantics, fn, "semantics")
T.wrap(F, "align_turn", "semantics", after=lambda items, res: {"reply_after": T.brief(dict(items).get("res", {}).get("reply"))})

# Turns
S = F.CloverS23OnDeviceSystem
T.wrap(S, "run_cycle", "turn", result=lambda r: {k: T.brief(v) for k, v in r.items() if k != "layer_trace"}
       | {"layer_trace_keys": list((r.get("layer_trace") or {}).keys())})
T.wrap(S, "social_turn", "turn")
T.wrap(S, "run_retry_cycle", "turn")

# C1 inference (4-bit C engine)
E = F.S23Lfm3NativeEngine
for fn in ("prefill_cached", "append_cached"):
    T.wrap(E, fn, "C1", args=ids_text)
for fn in ("step_embedding_cached", "step_token_cached"):
    T.wrap(E, fn, "C1")
for fn in ("decode_from_cache", "decode_from_cache_with_hidden", "generate_chat"):
    T.wrap(E, fn, "C1", result=ids_result)
T.wrap(S, "_build_system_prompt", "C1")

# C2 plan (HybridLink latent steps, tool decision) and C3 decode (NCM)
H = F.S23HybridAgentAndNCM
T.wrap(H, "run_continuous_latent_step", "C2")
T.wrap(H, "decode_hybrid_agent_turn", "C2", result=ids_result)
T.wrap(S, "_parse_lfm3_tool_call_xml", "C2")
T.wrap(S, "looping", "C2")
T.wrap(S, "_clean_model_text", "C3")
for fn in ("_generate_from_zpref_kv_exact", "_forward_pointer_decoder", "_dec_model_aligned"):
    T.wrap(H, fn, "C3")

# S3 tools and permissions
T.wrap(S, "_check_tool_permission", "S3")
for fn in ("_execute_python_subprocess", "_execute_shell_subprocess", "_execute_ssh_subprocess"):
    T.wrap(S, fn, "S3")
T.wrap(S, "_compile_verify_model_code_only", "S3-compiler")
T.wrap(S, "extract_python_from_reply", "S3-compiler")
T.wrap(S, "_audit_reply_grounding", "S2-grounding")

# Objects created inside the system's constructor: wrapped on their classes before construction
import clover_git_memory_repo as GR  # noqa: E402
import memory_index as MI  # noqa: E402
import web_search as WS  # noqa: E402
import k3_memory_graph as MG  # noqa: E402
import k3_ontology_graph as OG  # noqa: E402

for fn in ("search_repo_relevant", "record_task_or_session_turn", "read_file", "write_file", "remember_note", "user_notes"):
    if hasattr(GR.CloverMemoryGitRepo, fn):  # remember_note and user_notes: added after the first observation
        T.wrap(GR.CloverMemoryGitRepo, fn, "S1")
T.wrap(MG, "retrieve_relevant_memory_package", "S1-graph")
T.wrap(MI.MemoryIndex, "recall", "S2-recall")
T.wrap(OG.CloverOntologyRetriever, "retrieve", "S2-ontology")
T.wrap(OG, "render_evidence", "S2-ontology")
for fn in ("check_turing_self_reference_stratification", "match_formula_csm_lmc"):
    if hasattr(OG.CloverSymbolicVerifier, fn):
        T.wrap(OG.CloverSymbolicVerifier, fn, "S2-ontology")
for fn in ("search", "execute", "research"):
    T.wrap(WS.WebSearch, fn, "S3-web")

_init = S.__init__


def init_observed(self, *args, **kw):
    _init(self, *args, **kw)
    T.decoders["ids"] = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)
    T.emit({"layer": "startup", "fn": "CloverS23OnDeviceSystem", "phase": "ready", "setup_ms": self.setup_ms,
            "settings": self.settings, "permissions": self.perm_modes, "engine_threads_note": "see C engine",
            "hybrid_ncm_loaded": self.hybrid_ncm is not None,
            "modules": T.module_files(["s23_on_device_clover", "semantics", "clover_cycle", "clover_semantics",
                                       "clover_git_memory_repo", "memory_index", "web_search", "local_terminal",
                                       "k3_memory_graph", "k3_ontology_graph", "k3_generative_compiler",
                                       "ncm_runtime", "models", "models.next_concept", "models.vqvae"])})


S.__init__ = init_observed
T.emit({"layer": "startup", "fn": "observe_framework", "phase": "start", "base": a.base, "port": a.port})
F.serve_s23_http(Path(a.base), host="127.0.0.1", port=a.port)
