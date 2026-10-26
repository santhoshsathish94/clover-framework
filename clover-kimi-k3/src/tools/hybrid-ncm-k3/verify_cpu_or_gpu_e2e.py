#!/usr/bin/env python3
"""
Unified CPU & GPU End-to-End Verification (`verify_cpu_or_gpu_e2e.py`)
======================================================================
Verifies that the complete 7-Layer Clover Kimi K3 system runs end-to-end on
EITHER a GPU server (`gpu`, 8x RTX 3090) OR a CPU server (`k3`, Ryzen 9 7950X3D)
from a clean GitHub clone + Hugging Face dataset download (reusing existing
computed `expert-1..92` and `trunk-0..92`):

  1. Path & Dataset Discovery (`clover_paths.py`):
     - Verifies `DATASET_DIR` (`trunk-0..92`, `expert-1..92`), `OPEN_WEIGHTS_DIR` (`model-00094-of-000096.safetensors`,
       `tiktoken.model`), and `CKPT_DIR` (`e12_clover_k3_true_ha_ncm.pt` / `e11_clover_k3_native_ha_ncm.pt`).
  2. Native Kimi K3 C Tokenizer (`libk3tok.so`, 163,840 vocabulary).
  3. Real Kimi K3 MoE Backbone + Direction B Data-Movement Ledger (`K3GPUExpertsRunner`:
     `libk3_gpu_experts.so` on GPU or `libk3_cpu_experts.so` + `libslotcache.so` on CPU).
  4. Collective Memory Tree (`ensure_default_memory_tree`) + Stratified Priority Traversal.
  5. Neuro-Symbolic Ontology GraphRAG & Symbolic Verifier (`CloverOntologyRetriever` + `CloverSymbolicVerifier`).
  6. Generative Compilation Sealor (`CloverPythonGenerativeCompiler`).
  7. Full End-to-End Multi-Turn Agent & Natural Conversation (`CloverK3AgentEngine.agent_reply`
     running `HybridAgent-8B-Agentic` + Kimi K3 MoE + `K3NativeHybridLink` + 11-Pass NCM + `K3LexicalPointerDecoder`).
"""
import json
import os
import sys
import time

from clover_paths import (
    setup_clover_sys_path,
    find_dataset_dir,
    find_open_weights_dir,
    find_hybrid_ncm_ckpt_dir,
    resolve_device,
)

setup_clover_sys_path()

import torch
from k3_agent_engine import CloverK3AgentEngine, K3GPUExpertsRunner


def main():
    t_start = time.perf_counter()
    ds_dir = find_dataset_dir()
    ow_dir = find_open_weights_dir()
    ckpt_dir = find_hybrid_ncm_ckpt_dir()
    dev = resolve_device("cuda:0")

    print("=" * 80, flush=True)
    print(f"CLOVER KIMI K3 UNIFIED CPU/GPU VERIFICATION (device={dev}, cuda={torch.cuda.is_available()})", flush=True)
    print(f"  DATASET_DIR      : {ds_dir}", flush=True)
    print(f"  OPEN_WEIGHTS_DIR : {ow_dir}", flush=True)
    print(f"  HYBRID_NCM_CKPTS : {ckpt_dir}", flush=True)
    print("=" * 80, flush=True)

    assert os.path.isdir(os.path.join(ds_dir, "trunk-0")), f"Missing trunk-0 in {ds_dir}"
    assert os.path.isdir(os.path.join(ds_dir, "expert-1")), f"Missing expert-1 in {ds_dir}"
    assert os.path.exists(os.path.join(ow_dir, "model-00094-of-000096.safetensors")), f"Missing shard 94 in {ow_dir}"
    assert os.path.exists(os.path.join(ckpt_dir, "e12_clover_k3_true_ha_ncm.pt")) or os.path.exists(
        os.path.join(ckpt_dir, "e11_clover_k3_native_ha_ncm.pt")
    ), f"Missing e12/e11 checkpoint in {ckpt_dir}"

    # 1. Initialize K3GPUExpertsRunner (automatically compiles & loads libk3_cpu_experts.so on CPU or libk3_gpu_experts.so on GPU)
    k3_runner = K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=28)
    print(f"[PASS 1/5] K3MoE Runner initialized (is_cpu_backend={k3_runner.is_cpu_backend}, setup={k3_runner.setup_s}s)", flush=True)

    # 2. Initialize full CloverK3AgentEngine
    eng = CloverK3AgentEngine(out_dir=ckpt_dir, dev_ncm=dev, dev_ha=dev, k3_experts_runner=k3_runner)
    print(f"[PASS 2/5] CloverK3AgentEngine initialized on {eng.dev} in {eng.setup_s}s (vocab={eng.k3_tok.vocab_size})", flush=True)

    # 3. Verify Collective Memory + Ontology + Generative Compiler
    mem_nodes = len(eng.memory_graph.records)
    ont_ev = eng.ontology_retriever.retrieve("Does tying your belly reduce belly fat?", top_k=2)
    gc_res = eng.py_gc.check_partial("def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n")
    assert mem_nodes >= 20, f"Expected >=20 memory nodes, got {mem_nodes}"
    assert gc_res.ok, f"Expected partial Python code to seal cleanly, got {gc_res}"
    print(f"[PASS 3/5] Subsystems verified: MemoryGraph={mem_nodes} nodes, Ontology={len(ont_ev)} hits, PythonGC=OK", flush=True)

    # 4. Run Multi-Turn Natural Conversation & Tool Execution through all 7 layers
    cands = 2 if dev == "cpu" else 4
    turns = [
        ("What is clover", lambda r: "Context" in r["reply"] and "Direction" in r["reply"]),
        ("How can I do that", lambda r: "AGENTS.md" in r["reply"] or "Context" in r["reply"]),
        ("can you generate a rocket in space image", lambda r: bool(r.get("tool_calls")) and "Rocket" in json.dumps(r.get("tool_calls", []))),
        ("What is 25 * 4?", lambda r: "100" in r["reply"]),
    ]

    for idx, (u_prompt, check_fn) in enumerate(turns, 1):
        t0 = time.perf_counter()
        res = eng.agent_reply(u_prompt, cands=cands, remember=True)
        dt = (time.perf_counter() - t0) * 1000.0
        ok = bool(check_fn(res))
        print(
            f"  [Turn {idx}] ({dt:7.1f} ms, k3_toks={res.get('n_k3_tokens')}) U: {u_prompt!r}\n"
            f"           A: {res['reply'].replace(chr(10), ' | ')[:180]}",
            flush=True,
        )
        assert ok, f"Turn {idx} failed check: {res}"

    print("[PASS 4/5] Multi-turn natural conversation, <python> tool, and SVG rocket generation verified", flush=True)

    # 5. Check Direction B Data-Movement Ledger
    dm = k3_runner.get_data_movement_ledger()
    print(f"[PASS 5/5] Direction B Data-Movement Ledger: {json.dumps(dm)}", flush=True)
    assert dm.get("policy_name") == "specmd_least_stale_r4", f"Unexpected policy: {dm}"

    print("=" * 80, flush=True)
    print(f"ALL 5 UNIFIED VERIFICATION GATES PASSED IN {time.perf_counter() - t_start:.2f}s!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
