# Sequential HybridAgent Weight-Space Study

## Direction

The human confirmed on 2026-11-12: investigate Nemotron 3.5 Super VL first,
then Kimi K3, one HybridAgent at a time, without rushing. Produce a final pushed
research branch and precise baseline/candidate dataset locations for independent
end-to-end testing. Commit/push of research code and documentation is authorized;
weights stay on K3. No production replacement, release or gradient training is
authorized. Existing model files and prior frozen studies are immutable.

Worktree: C:\personal\oss\clover-astra-weightspace-wt.
Branch: research/astra-lfm3-weightspace-20261111.
New local record: research/hybridagent-weightspace.
New K3 outputs: /opt/clover-k3/hybridagent-weightspace-20261112.

## Evidence correction

The previous LFM3 report's "zero degradation" claim was too strong. Its g0.92
ablation worsened entry CE, exit cosine loss and forced-rollout exit rank, and
did not meet the frozen backbone-loss significance gate. Its QA scorer accepted
some answer occurrences, and training-split exclusion was not established from
an actual Omni training manifest. The reports were corrected, without changing
their frozen outputs. No model may inherit that overstated acceptance claim.

## Nemotron context

The located checkpoint is
/opt/clover-k3/nemotron-computed/hybridagent-8gpu/nemotron_hybridagent_final.pt.
The saved training code uses a reduced Mamba -> MoE -> attention stage model,
not all 88 layers. It exports a standard HybridLink, a native HybridLink,
special-token embedding/head deltas and a final-norm delta. Both latent paths
contribute to a blended feedback input in the saved training objective.

First hypothesis: a parameter calibration can only affect end-to-end behavior
if the target runtime actually loads and uses that parameter. Check exact
checkpoint namespaces/shapes/values and trace consumption before choosing an
edit. First executable check: known-value statistics, then hash-verified physical
copy and inventory. Do not treat reduced-stage loss or isolated link alignment
as proof of full-model, downstream planning, or deployment improvement.

No cross-model weight scaling is assumed valid.

## Completed outcomes

Nemotron completed first. All four gain changes failed per-example guards;
40 discovery evaluations and 18 source/copy records were audited. All trial
files were restored and reserved validation stayed unopened. The test is a
reduced-stage diagnostic, not a full-model or current-live-pipeline improvement.

Kimi then completed using a native checkpoint copy and the existing CPU runtime.
The staged NCM data was not misused as C2 state. Missing reference captures were
replaced by new runtime captures after the human selected that route. The
private dataset view resolves split storage and aliases; K3_CPU_LAYERS=all
ensures all 92 MoE layers are active. Twelve eight-token windows passed finite
state/coverage checks. Both gain candidates failed the zero-control discovery
gate, so reserved windows were not evaluated for a candidate. All trials were
restored and sources verified unchanged.

The handoff contains no accepted candidate for either model. Baseline paths,
checksums, tested component keys and evidence locations are in HANDOFF.json;
handoff.py verified them on K3. No production service or original model changed.
The evidence-only archive SHA-256 is
31cc59409adabeca25fa2dfc082de61c60a882631d0be63b679327fed6b77d4f;
it was checked before local extraction. Captured arrays and all weights remain
on K3. Commit/push delivery is the final step, explicitly authorized by the human.