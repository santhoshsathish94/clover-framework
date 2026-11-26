# Weight distribution report

Observed: 868/868 stored tensors analyzed in 33.40 seconds on K3 (8 threads).
Nonfinite values: 0. Original hashes matched before and after: True.

## Methods and limits

Exact float64 accumulated moments over every element; exact row/column L2 norms and counts beyond six tensor-specific standard deviations.
Value quantiles use all elements up to 65536, otherwise 65536 seeded uniform draws with replacement. Histograms use that same sample.
Convolution tensors are flattened with axis 0 as rows and remaining axes as columns. The saved NPZ holds every row/column norm and concentration count.
Full float32 SVD is used when the smaller dimension is <=256 and element count <=8 million. Other spectra use randomized top-16, oversampling 8, two power iterations, seed 20261111.
The truncated spectra are not full-rank estimates. Pair correlations sample 4096 aligned elements within equal-shape/component groups; row-pair checks use top-norm and evenly spaced rows.
Exploratory multiple comparisons are not evidence of defects. No normalization, clipping, rank reduction or automatic correction was performed.

| Component | Six-std elements | Matrices surveyed | Median top-16/full reported energy |
|---|---:|---:|---:|
| audio_input_encoder | 3,681 | 115 | 0.1359 |
| audio_to_backbone_projector | 5,464 | 2 | 0.1992 |
| shared_backbone | 150,668 | 166 | 0.0599 |
| shared_embedding_and_text_output | 449 | 1 | 0.0531 |
| speech_code_to_backbone_embedding | 118 | 1 | 0.5119 |
| speech_output_depth_decoder | 22,179 | 37 | 0.8505 |
| speech_waveform_codec | 8,482 | 116 | 0.2820 |
| vision_encoder_image_and_video | 12,828 | 91 | 0.0670 |
| vision_to_backbone_projector | 0 | 2 | 0.0175 |

Energy medians mix full small-matrix and truncated large-matrix reports; compare per-tensor method fields, not this aggregate as a rank claim.

## Final shared RMS gain

Observed range 0.069335938 to 5.5625; mean 2.9044223; std 0.32266311; excess kurtosis 11.869.
This heterogeneous learned scaling is not itself a defect. Candidate selection requires behavioural activation evidence and is not based on these extremes.

## Highest sampled same-shape correlations

- `audio_tokenizer/encoder.lfm.layers.2.self_attn.gate_proj.weight` versus `audio_tokenizer/encoder.lfm.layers.6.self_attn.gate_proj.weight`: sampled Pearson 0.459120.
- `main/model.depth_decoder.depth_embeddings.2.embedding.weight` versus `main/model.depth_decoder.depth_embeddings.7.embedding.weight`: sampled Pearson 0.256923.
- `main/model.depth_decoder.depth_embeddings.2.embedding.weight` versus `main/model.depth_decoder.depth_embeddings.5.embedding.weight`: sampled Pearson 0.208122.
- `main/model.depth_decoder.depth_embeddings.1.embedding.weight` versus `main/model.depth_decoder.depth_embeddings.8.embedding.weight`: sampled Pearson 0.198003.
- `main/model.depth_decoder.model.layers.1.self_attn.gate_proj.weight` versus `main/model.depth_decoder.model.layers.2.self_attn.gate_proj.weight`: sampled Pearson 0.187991.

These may reflect shared learned structures. Their functional necessity has not been established by the weight survey.
Per-tensor evidence: `weight-survey-v2/tensors.jsonl`; full profiles: `weight-survey-v2/norms_and_concentrations.npz`.
