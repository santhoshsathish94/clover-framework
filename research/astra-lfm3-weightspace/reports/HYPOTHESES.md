# Ranked intervention hypotheses

Status: mathematical hypotheses supported by discovery associations; not yet experimentally verified improvements.

Target tensor: `model.language_model.embedding_norm.weight` (final shared RMS gain, after backbone layer 23 and before the tied text head).

## Derivation

For pre-norm state $r$, $z=r/\sqrt{\operatorname{mean}(r^2)+\epsilon}$, gain $g$, hidden $h=g\odot z$, and logits $\ell=Eh$.
Changing one gain $g_j'=(1-\alpha)g_j$ gives $\Delta\ell_y=-\alpha E_{y,j}h_j$ in real arithmetic.
For gold $a$ and the baseline strongest incorrect letter $b$, $\Delta(\ell_a-\ell_b)=-\alpha(E_{a,j}-E_{b,j})h_j=-\alpha c_j$.
BF16 gain multiplication/head rounding, competitor switching and autoregressive continuation can invalidate that local prediction.

Shared procedure: 5%, 10% (predeclared primary), 20% attenuation; one coordinate in one tensor per fresh copied checkpoint. No gradients or other edits.
Risk: lower margins for correct cases, label bias, worse unrelated text/media reasoning, ASR or speech-depth conditioning. The selected coordinate is not assumed to be a defective outlier.
Reversal: restore the recorded original two BF16 bytes in the copy, then require the entire checkpoint SHA-256 to equal the sealed original. Never reverse by approximate inverse scaling.

## H1: coordinate 36

Exact flat index 36; original gain 3.03125; selection score 0.0707615.
Observed mean contribution: failures -0.0835954, successes 0.00641696.
Predicted 10% mean margin change: failures +0.00835954, successes -0.000641696.
Selected by the preregistered discovery-only rule, not by weight extremeness or validation outcomes.

## H2: coordinate 916

Exact flat index 916; original gain 2.96875; selection score 0.062697.
Observed mean contribution: failures -0.258464, successes 0.0978834.
Predicted 10% mean margin change: failures +0.0258464, successes -0.00978834.
H1 is the sole execution candidate. H2/H3 are lower-ranked alternatives and will not be tried after seeing validation results.

## H3: coordinate 248

Exact flat index 248; original gain 2.90625; selection score 0.053879.
Observed mean contribution: failures -0.0709955, successes 0.00855826.
Predicted 10% mean margin change: failures +0.00709955, successes -0.000855826.
H1 is the sole execution candidate. H2/H3 are lower-ranked alternatives and will not be tried after seeing validation results.

## Supporting failures

- `discovery-001`: generated `A`; correct label `B`; first-token margin -17.9062.
- `discovery-002`: generated `B`; correct label `C`; first-token margin -10.6406.
- `discovery-003`: generated `C`; correct label `D`; first-token margin -7.1875.
- `discovery-004`: generated `A`; correct label `B`; first-token margin -17.3359.
- `discovery-005`: generated `A`; correct label `C`; first-token margin -15.6875.
- `discovery-008`: generated `A`; correct label `C`; first-token margin -15.9863.

These are exploratory comparisons across 2048 coordinates. No significance or general improvement is inferred from choosing a maximum. The held-out paired experiment is the causal test.
