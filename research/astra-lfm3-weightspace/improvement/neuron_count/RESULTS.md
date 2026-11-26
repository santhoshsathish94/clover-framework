# Edited-neuron-count experiment

This increases the number of existing layer22 MLP neurons whose output columns are edited. It does not add architectural neurons or retrain the model.
Same layer, discovered direction, native runtime and tuning cases; fixed-strength and matched-total-norm tracks separate coverage from perturbation size.

| Edited neurons | Track | Strength | Delta norm | Arithmetic /144 | Free /36 | Selection /48 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Original | 0 | 0 | 74 | 32 | 48 |
| 64 | fixed | 4 | 1.99636 | 79 | 32 | 48 |
| 128 | fixed | 4 | 2.28348 | 79 | 32 | 48 |
| 128 | norm_matched | 3.49699 | 1.9962 | 79 | 32 | 48 |
| 256 | fixed | 4 | 2.63901 | 77 | 32 | 48 |
| 256 | norm_matched | 3.02585 | 1.99615 | 78 | 32 | 48 |
| 512 | fixed | 4 | 3.09614 | 77 | 32 | 48 |
| 512 | norm_matched | 2.57903 | 1.99613 | 76 | 32 | 48 |
| 1024 | fixed | 4 | 3.6236 | 77 | 32 | 48 |
| 1024 | norm_matched | 2.20348 | 1.9961 | 77 | 32 | 48 |

The64-neuron control exactly matched the previous tested checkpoint hash and all tuning outputs. Each trial changed exactly one tensor and its declared columns, with byte-exact reversal afterward.

## Outcome

No wider candidate beat both the original and64-neuron control while preserving free-response and selection totals. No fresh confirmation was opened and no wider model was retained.

## Limits and artifacts

Tuning results are exploratory. Wider coverage can also cause larger interference; more edited neurons is not automatically more capability. Old final data were not reused for confirmation.
Originals remain immutable, no gradients or deployment. Trial sources, frozen neuron sets, actual rounded perturbation norms, raw outputs and all tensor/reversal audits are retained. A failed confirmation leaves the original as the retained final model.
