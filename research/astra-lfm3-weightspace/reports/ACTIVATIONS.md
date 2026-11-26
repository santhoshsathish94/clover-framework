# Activation and direction analysis

Discovery: 15/32 correct, 17/32 incorrect. Validation examples excluded from selection.

Observed association, not a causal explanation. Final-token vectors are compared within this small constructed arithmetic sample.

| Location | Failure-minus-success contrast L2 | Leave-one-out centroid AUC |
|---|---:|---:|
| layer_0 | 0 | 0.5000 |
| layer_2 | 0.015794 | 0.5608 |
| layer_4 | 0.0212167 | 0.5804 |
| layer_11 | 0.0979656 | 0.5882 |
| layer_17 | 0.614251 | 0.6196 |
| layer_23 | 1.45971 | 0.6510 |
| final_hidden | 12.1338 | 0.6118 |

AUC is cross-fitted inside discovery, not independent validation. Directions and standardized coordinate contrasts are in `discovery-directions.npz`.
Float64 contribution-sum versus logged BF16 margin maximum absolute difference: 0.0578804.
Thus the real-arithmetic sensitivity must not be represented as an exact prediction of rounded model logits.

## Cohort confounds

```json
{
  "family": {
    "addition": {
      "success": 3,
      "failure": 5
    },
    "subtraction": {
      "success": 2,
      "failure": 6
    },
    "multiplication": {
      "success": 6,
      "failure": 2
    },
    "division": {
      "success": 4,
      "failure": 4
    }
  },
  "gold_letter": {
    "A": {
      "success": 8,
      "failure": 0
    },
    "B": {
      "success": 2,
      "failure": 6
    },
    "C": {
      "success": 1,
      "failure": 7
    },
    "D": {
      "success": 4,
      "failure": 4
    }
  }
}
```

Operation, answer label, sequence length and prompt content can explain apparent directions. This analysis does not prove a defect or a general failure representation.
Image/audio/video/speech activations are captured in the baseline archive but are not used to fit this text-derived intervention. Their small samples do not establish modality-specific failure subspaces.
