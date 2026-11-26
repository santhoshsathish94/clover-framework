# HybridSFT

Hybrid supervised fine-tuning turns a pre-trained base model into a HybridAgent that
coordinates its backbone with HybridLink for stable tool-integrated hybrid generation.

<p align="center"><img src="../assets/hybrid_sft.png" width="100%" alt="Self-conditioned HybridSFT"></p>

## Stages

| Stage | `--stage` | Trainable | Objective |
|---|---|---|---|
| Format activation | `format_activation` | backbone | cross-entropy on the hybrid trajectories: the backbone learns when to emit the entry and exit anchors `<latent>` / `</latent>` |
| HybridLink activation | `hybridlink_activation` | HybridLink | regression of the HybridLink outputs at latent positions onto their target input embeddings |
| Self-conditioned HybridSFT | `self_conditioned` | backbone + HybridLink | joint training of backbone and HybridLink on the model's own latent inputs, closing the train–inference gap |

## Data

One JSON object per line. Latent spans are enclosed by the anchors inside the assistant text; tool observations are kept in `tool_result` and serve as context only.

```json
{"id": 0,
 "system": "<system prompt>",
 "question": "...",
 "rounds": [
   {"assistant": "<think>\nKey step.<latent>routine reasoning</latent>Decision.\n</think><python>...</python>",
    "tool": "python", "tool_result": "<result>...</result>"},
   {"assistant": "<think>\n...<latent>...</latent>...\n</think><answer>\\boxed{...}</answer>",
    "tool": null, "tool_result": null}]}
```

`--data` accepts a local JSONL file or a Hugging Face dataset repo id (with `--data_file` naming the file in the repo, default `train.jsonl`). The released data are `togethercomputer/HybridAgent-HybridSFT-v1` and `togethercomputer/HybridAgent-HybridSFT-v2`; v2 is the upgraded version, produced by the same annotation pipeline with a stronger labeling model. Each record's `system` field is used as the system prompt; when it is absent, the default prompt in `hybrid_agent/prompts.py` is used.

## Training

```bash
MODEL=Qwen/Qwen3-8B-Base DATA=<jsonl or HF dataset id> OUT=outputs/hybrid_sft bash scripts/run_hybrid_sft.sh all
```

The launcher runs the three stages in order, each initialized from the previous output. Recommended hardware: 8 or 16 GPUs. The provided accelerate configs assume one node with 8 GPUs (batch size 1 × gradient accumulation 2 × 8 = 16); with a different GPU count adjust `--grad_accum` and the matching `gradient_accumulation_steps` in the config so the effective batch stays the same. For the 14B model use the CPU-offload config for the last stage: `CONFIG_SELF_CONDITIONED=hybrid_sft/configs/accelerate_zero2_offload.yaml`. A single stage can be run directly:

```bash
accelerate launch --config_file hybrid_sft/configs/accelerate_zero2.yaml -m hybrid_sft.train \
    --stage hybridlink_activation --model outputs/hybrid_sft/format_activation \
    --data <jsonl or HF dataset id> --output_dir outputs/hybrid_sft/hybridlink_activation
```

Epochs default to 1 / 2 / 1 for the three stages (`--epochs`; 1 / 1 / 1 and 1 / 1 / 2 are good practice as well); the other settings are the trainer defaults, listed by `python -m hybrid_sft.train --help`. Evaluate the saved HybridSFT checkpoints, e.g. on AIME and MATH-500, and take the best one as the initialization for HybridRL.

## Output

Every checkpoint directory (`checkpoint-N/` and the final output) is a complete HybridAgent checkpoint: the backbone in Hugging Face format, the tokenizer with the anchor tokens, and the HybridLink weights (`adapter.pt`, `adapter_config.json`). It loads with `hybrid_agent.load_hybrid_agent` and is the initialization for HybridRL.
