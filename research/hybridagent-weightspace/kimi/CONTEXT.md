# Kimi K3 HybridAgent

This second investigation began only after Nemotron's trials, rejection and
independent reversal/source audit completed. Scope is the native 7168-dimensional
K3 HybridLink, not a separately loaded Qwen-based HybridAgent model.

The hf-push checkout's 135-byte checkpoint files are Git LFS pointers. The real
combined Kimi checkpoint was located under
/mnt/nvme1/clover-k3/datasets/hybrid-ncm-k3. The runtime prefers
checkpoint-clover-true-hybrid-agent-ncm.pt and loads its k3_hybridlink and
k3_cbridge. A 32-window staged-input dataset is available beside it.

First verify native-link tensor values and available input/target shapes on
physical copies. Use the native class from the saved source without executing
unrelated training or Qwen-model loaders. Do not invent missing full-model
states or call proxy/staged inputs a verified generalization benchmark.

## Existing runtime resolved

The human selected using the existing full Kimi runtime. The real CPU library
and final-tail shard were located under /opt/clover-k3/clover-framework and
copied with hashes into the private runtime/ directory. The library's default
CPU path opens only four MoE layers despite its full93 API name, so capture
explicitly sets K3_CPU_LAYERS=all and requires 92 active MoE layers.

The first preflight at /mnt/nvme1/clover-k3/datasets/computed found 47 missing
trunk indexes and stopped before inference. The other layers live under
/opt/clover-k3/datasets/computed. A private symlink view joins these roots,
deduplicating only identical resolved paths; original files were not moved.
Both roots are mounted read-only during capture. Full large expert payloads
were not duplicated or independently hashed; the index hashes, copied runtime
library and copied tail weights are pinned, and capture coverage is recorded.

The eight-token smoke succeeded with 92 active MoE layers and finite states.
Before the larger run, twelve windows and the discovery/validation split were
frozen: four discovery windows, eight reserved windows, eight tokens per window.
All twelve captures succeeded through the configured 93-layer CPU path. The
first window reproduced the smoke hash exactly. This is existing-runtime
execution, not an independent proof of parity with an official reference.

## Native gain trials

Only k3_hybridlink.out_scale was changed, by 0.92 or 1.08, in separate physical
copies of the combined checkpoint. The entire remaining state dictionary was
hash-audited unchanged. Both candidates failed the discovery gate requiring
lower MSE without per-window MSE/cosine regression and improvement over a
zero-vector control. No candidate was selected; reserved windows were not used
for candidate evaluation. All trial checkpoints were restored byte-exactly.

No improved Kimi model is claimed. Preserve the real captures, rejected patches,
baseline and runtime-view manifest for end-to-end investigation. The observed
input windows are reserved within this calibration, not proven absent from the
original adapter training corpus. The separate Qwen planner was not loaded.