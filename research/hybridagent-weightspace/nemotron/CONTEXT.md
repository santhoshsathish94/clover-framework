# Nemotron HybridAgent

The first sequential investigation used a physical copy of
/opt/clover-k3/nemotron-computed/hybridagent-8gpu/nemotron_hybridagent_final.pt.
Its SHA-256 is 21f72954f07a67b4cc68443d6bfe4661afd711c98cece3f3e33a369f27d9e5cb.
The copy contains 23 tensors. Seventeen required stage/tokenizer files were
physically copied with hashes (8,205,586,387 bytes).

The saved training graph is three stages (Mamba, MoE, attention), not all 88
Nemotron layers. Only six of 63 full-model shards were found in the known model
store. The live workstation loader also prefers HybridLink weights in the bridge
checkpoint over the standalone HybridSFT file. We did not claim full-model or
current-live-pipeline equivalence, and did not deploy or modify the live loader.

The structural two-pass smoke succeeded with original model paths inaccessible.
We froze eight procedural arithmetic trajectories for discovery and sixteen for
reserved validation, using the saved Nemotron data builder and exact token labels.
This is diagnostic data, not a claimed reproduction of the original training
holdout. The tested single-component factors were 0.92 and 1.08 on the standard
HybridLink post-norm pair or on the native HybridLink output scale.

All four candidates failed the per-example NTP/text-loss/alignment/correct-token
non-degradation gate. Mean discovery NTP was 5.595701 original, 5.595140 standard
0.92, 5.601898 standard 1.08, 5.597499 native 0.92, and 5.595040 native 1.08.
Tiny average reductions did not override individual regressions. No candidate
was selected and the sixteen reserved cases remained unopened. Every physical
trial copy was restored to the baseline bytes. The run completed in 64.021 seconds.

No improved Nemotron dataset is ready for end-to-end use. Preserve the baseline,
protocol, raw records and rejected patch specifications for inspection. Kimi work
starts only after this outcome and the source/reversal audit are complete.