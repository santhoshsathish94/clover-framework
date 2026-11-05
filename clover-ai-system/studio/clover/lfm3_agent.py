"""Standalone Clover engine for LFM3-1.7B on the laptop (Windows, CPU).

Layer 1 + 3: Lfm3NCMRenderer (verified bit-identical to k3). Layers 4-6 and the agent loop: clover_loop.py, verbatim
from clover-framework. Layer 2 is a slot: base LFM3 (text planner, no latent steps) or a trained LFM3 HybridAgent
checkpoint (adapter.pt present: HybridLink latent steps as in Clover's CPU path).

Mapping from CloverK3AgentEngine (Kimi K3 is not on the laptop):
  k3_tok      -> LFM3 tokenizer adapter (window splitting / token-length checks in LFM3's vocabulary)
  k3_experts  -> Layer 1 MCQ scoring with LFM3's C engine + tied head (score_mcq_options)
  render_ncm_with_k3_experts -> Lfm3NCMRenderer.render_plans_in_model_vocab(plans, cands=1), as the chat server does for lfm3
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clover_loop as CL  # noqa: E402
import lfm3_clover as LC  # noqa: E402
from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_CLOSE, THINK_OPEN  # noqa: E402
from hybrid_agent.model import HybridAgentModel  # noqa: E402
from windows_python_tool import PythonTool  # noqa: E402

torch.backends.quantized.engine = "onednn"


class QLinear(nn.Module):
    """Dynamic int8 linear on fp32 activations, returning the caller's dtype."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, x):
        return self.q(x.float()).to(x.dtype)


def load_planner_backbone(model_dir, int8_ffn=True):
    """Clover's CPU planner precision (clover-agent-engine.py: fp32 load, SwiGLU MLP blocks int8): weights stream from
    safetensors into a meta skeleton, feed_forward linears go straight to int8, so no fp32 copy of them is ever held."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.float32)
    qcfg = torch.ao.quantization.per_channel_dynamic_qconfig
    n_q = 0
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as st:
        keys = set(st.keys())
        if int8_ffn:
            for i, layer in enumerate(model.model.layers):
                for name in ("w1", "w2", "w3"):
                    meta = getattr(layer.feed_forward, name)
                    lin = nn.Linear(meta.in_features, meta.out_features, bias=False)
                    lin.weight = nn.Parameter(st.get_tensor(f"model.layers.{i}.feed_forward.{name}.weight").float(), requires_grad=False)
                    lin.qconfig = qcfg
                    setattr(layer.feed_forward, name, QLinear(torch.ao.nn.quantized.dynamic.Linear.from_float(lin)))
                    del lin
                    n_q += 1
        for pname, p in list(model.named_parameters()):
            if p.device.type != "meta":
                continue
            owner_name, attr = pname.rsplit(".", 1)
            owner = model.get_submodule(owner_name)
            if pname in keys:
                owner._parameters[attr] = nn.Parameter(st.get_tensor(pname).float(), requires_grad=False)
    if getattr(cfg, "tie_word_embeddings", False) or model.lm_head.weight.device.type == "meta":
        model.lm_head.weight = model.model.embed_tokens.weight
    for mname, mod in list(model.named_modules()):
        if "Rotary" in type(mod).__name__ and any(b.device.type == "meta" for b in mod.buffers(recurse=False)):
            parent = model.get_submodule(mname.rsplit(".", 1)[0]) if "." in mname else model
            setattr(parent, mname.rsplit(".", 1)[-1], type(mod)(config=cfg))
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"planner tensors left on meta: {left[:8]}")
    model.eval()
    return model, n_q


class Layer1MCQ:
    """Stands in for K3GPUExpertsRunner.score_mcq_options: same prompt stem, LFM3 Layer 1 instead of Kimi K3."""

    def __init__(self, renderer):
        self.renderer = renderer

    def score_mcq_options(self, prompt_text, _tok=None):
        best, probs, _ms = self.renderer.score_mcq_options(prompt_text)
        return best, probs


class Layer1Head:
    """Layer 1 without the NCM renderer: LFM3 tokenizer adapter, C engine and tied head (what planning needs)."""
    score_mcq_options = LC.Lfm3NCMRenderer.score_mcq_options

    def __init__(self, data_dir, engine_dir):
        import numpy as np
        data_dir = Path(data_dir)
        pruned = data_dir / "lfm3" / "pruned"
        self.tok = LC.HFModelTokenizerAdapter(data_dir / "lfm3" / "hf", LC.V_LFM3, eos_id=124900, pad_id=124893, semicolon_id=26)
        self.c_engine = LC.Lfm3NativeCPUEngine(pruned, engine_dir)
        emb_mm = np.memmap(pruned / "global-0" / "embed_tokens.bf16.bin", dtype=np.uint16, mode="c", shape=(LC.V_LFM3, LC.D_LFM3))
        self.lm_head_w = torch.from_numpy(emb_mm).view(torch.bfloat16)


class RecordReplaySearch:
    """Live search with a query->result cache so arms run on different days see the same tool observations."""

    def __init__(self, inner, cache_path, mode="record"):
        self.inner = inner
        self.path = Path(cache_path)
        self.mode = mode
        self.cache = {}
        self.stats = {"hits": 0, "live": 0}
        if self.path.exists():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    self.cache[rec["query"]] = rec["result"]

    def execute(self, query):
        if self.mode != "live" and query in self.cache:
            self.stats["hits"] += 1
            return self.cache[query]
        res = self.inner.execute(query)
        self.stats["live"] += 1
        if self.mode != "live":
            self.cache[query] = res
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"query": query, "result": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
        return res


class CloverLFM3Engine:
    def __init__(self, base_dir, planner_dir, search_cache=None, search_mode="record", int8_ffn=True, max_latent_steps=None,
                 load_renderer=True, load_planner=True, gate_tools=False, progress=None, planner_backend="torch"):
        t0 = time.perf_counter()
        base_dir = Path(base_dir)
        self.dev = self.dev_ha = torch.device("cpu")
        self.lock = threading.Lock()
        self.plan_replay = None
        self.stage = {"name": "loading", "since": time.time()}
        self._progress = progress or (lambda msg: None)
        self._progress("Layer 1 + 3: LFM3 C engine and NCM renderer")

        if load_renderer:
            self.renderer = LC.Lfm3NCMRenderer(base_dir / "data", base_dir / "engine")
            layer1 = self.renderer
        else:
            self.renderer = None
            layer1 = Layer1Head(base_dir / "data", base_dir / "engine")
        self.k3_tok = layer1.tok
        self.k3_experts = Layer1MCQ(layer1)

        from transformers import AutoTokenizer
        from hybrid_agent.model import add_anchor_tokens
        planner_dir = Path(planner_dir)
        self.planner_dir = str(planner_dir)
        self.has_hybridlink = (planner_dir / "adapter.pt").is_file()
        self.ha_tok = AutoTokenizer.from_pretrained(str(planner_dir), trust_remote_code=True)
        if self.has_hybridlink:
            add_anchor_tokens(self.ha_tok)
        self.ha_tok.padding_side = "right"
        self.ha_model, self.int8_linears = None, 0
        lib = layer1.c_engine.lib
        c_ok = hasattr(lib, "lfm3_cache_feed")
        if planner_backend == "auto":
            planner_backend = "c" if (c_ok and not self.has_hybridlink) else "torch"
        if planner_backend == "c" and (self.has_hybridlink or not c_ok):
            raise RuntimeError("C-engine planner needs the base LFM3 planner and an engine DLL with lfm3_cache_*")
        self.planner_backend = planner_backend if load_planner else "none"
        if load_planner and planner_backend == "c":
            from c_planner import c_engine_planner
            self._progress("Layer 2: LFM3 planner on the C engine")
            self.ha_model = c_engine_planner(lib, layer1.lm_head_w, max_tokens=8192)
        elif load_planner:
            self._progress("Layer 2: LFM3 planner")
            backbone, self.int8_linears = load_planner_backbone(planner_dir, int8_ffn=int8_ffn)
            self.ha_model = HybridAgentModel(backbone)
            if self.has_hybridlink:
                state = torch.load(str(planner_dir / "adapter.pt"), map_location="cpu", weights_only=True)
                self.ha_model.hybridlink.load_state_dict(state, strict=True)
            self.ha_model.eval()
        vocab = self.ha_tok.get_vocab()
        self.lat_open_id = vocab.get(ENTRY_ANCHOR)
        self.lat_close_id = vocab.get(EXIT_ANCHOR)
        self.think_open_id = vocab[THINK_OPEN]
        self.think_close_id = vocab[THINK_CLOSE]
        self.max_latent_steps = (14 if self.has_hybridlink else 0) if max_latent_steps is None else max_latent_steps

        self._progress("Layers 4-6: memory graph, ontology, compiler")
        self.py_tool = PythonTool(python_path=sys.executable, timeout=15, max_output_chars=250_000, memory_limit_gb=4.0)
        live = CL.K3KnowledgeSearchTool()
        self.gate_tools = bool(gate_tools)
        if self.gate_tools:
            import permissions as P
            self.py_tool = P.GatedPython(self.py_tool)
            live.live_web = P.GatedWeb(live.live_web)
            gc_globals = CL.CloverPythonGenerativeCompiler.check_spec_doctests.__globals__
            if not isinstance(gc_globals["subprocess"], P.GatedSubprocess):
                gc_globals["subprocess"] = P.GatedSubprocess(gc_globals["subprocess"])
        self.search_tool = RecordReplaySearch(live, search_cache, search_mode) if search_cache else live
        self.ontology_retriever = CL.CloverOntologyRetriever()
        self.symbolic_verifier = CL.CloverSymbolicVerifier(self.ontology_retriever)
        self.enable_ontology = True
        self.memory_graph = CL.ensure_default_memory_tree()
        self.enable_memory_graph = True
        self.enable_memory_auto_stage = False
        self.py_gc = CL.CloverPythonGenerativeCompiler(allow_undefined=True)
        self.rust_gc = None
        self.history = []
        self._last_decoded_gen = ""
        self.setup_s = round(time.perf_counter() - t0, 2)
        self._set_stage("idle")

    def _set_stage(self, name):
        self.stage = {"name": name, "since": time.time()}

    def describe(self):
        return {"planner_dir": self.planner_dir, "hybridlink": self.has_hybridlink, "max_latent_steps": self.max_latent_steps,
                "planner_backend": self.planner_backend, "int8_feed_forward_linears": self.int8_linears,
                "threads": torch.get_num_threads(), "setup_s": self.setup_s, "tools_gated": self.gate_tools}

    def run_agent_planning(self, user_prompt, history=None, **kw):
        if self.plan_replay is not None:
            return self.plan_replay[user_prompt]
        kw.setdefault("max_lat", self.max_latent_steps)
        self._set_stage("planning")
        return CL.run_agent_planning(self, user_prompt, history=history, **kw)

    def try_direct_k3_ha_ncm_decode(self, user_prompt, history=None):
        return None  # clover-agent-engine.py try_direct_k3_ha_ncm_decode returns None unconditionally

    def render_ncm_with_k3_experts(self, plans, cands=8):
        self._set_stage("rendering")
        return self.renderer.render_plans_in_model_vocab(plans, cands=1)

    def agent_reply(self, user_prompt, cands=8, remember=True):
        with self.lock, torch.inference_mode():
            try:
                return CL.agent_reply(self, user_prompt, cands=cands, remember=remember)
            finally:
                self._set_stage("idle")

    def plan_only(self, user_prompt):
        """Phase 1 of agent_reply for split runs: everything before rendering (no history, as remember=False)."""
        with self.lock, torch.inference_mode():
            t0 = time.perf_counter()
            plans, lat_steps, exit_reason, tool_calls = self.run_agent_planning(user_prompt, history=None)
            return {"plans": plans, "latent_steps": lat_steps, "exit_reason": exit_reason, "tool_calls": tool_calls,
                    "plan_s": round(time.perf_counter() - t0, 2), "planner_text": self._last_decoded_gen}
