# Architecture and tensor inventory

Observed on K3 from the pinned original checkpoint and its source; not the trained Clover variant.

Revision: `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.

| Component | Stored tensors | Stored elements |
|---|---:|---:|
| audio_input_encoder | 170 | 109,505,344 |
| audio_to_backbone_projector | 6 | 5,772,800 |
| shared_backbone | 285 | 1,466,657,536 |
| shared_embedding_and_text_output | 1 | 262,144,000 |
| speech_code_to_backbone_embedding | 1 | 49,569,792 |
| speech_output_depth_decoder | 64 | 83,898,560 |
| speech_waveform_codec | 203 | 73,559,090 |
| vision_encoder_image_and_video | 132 | 164,206,336 |
| vision_to_backbone_projector | 6 | 25,174,016 |

Stored elements include checkpoint buffers; runtime parameter/buffer identities are in benchmark metadata.
Tied aliases are not counted twice. The complete tensor-level shapes, dtypes, offsets and shard hashes are in `inventory.json`.

## Source-defined paths

- Shared backbone: width 2048, 24 layers: 13 short-convolution, 6 full-attention, 5 sliding-attention. Gated MLPs and sandwich RMS norms.
- Text input/output: one 128000 x 2048 embedding, also used by the tied text head. Runtime tying checked by storage pointer.
- Images and video: the same width-1024, 14-layer vision tower and two-layer projector feed the shared backbone. Video contains timestamped temporal patch groups.
- Audio input: three stride-2 spectrogram convolutions, a width-768, 17-layer encoder, and a separate projector into the shared backbone.
- Speech output: a width-1024, three-layer bidirectional depth decoder predicts 12 codebooks per frame. Each codebook's embedding is also its output head.
- Waveform codec: separate LiquidCodec checkpoint; 24 kHz output, 1920 samples/frame, 12 codebooks. It is included in the inventory and weight survey.
- The final shared RMS gain is after all 24 backbone blocks. Changing it can affect text, multimodal answers, and the speech-depth input.

## Checkpoint hashes

- main/model.safetensors: `76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763` (4,333,941,640 bytes).
- audio_tokenizer/model.safetensors: `45984a872fdcf0b94cd0ec68822bbf26f0c6fcb580cfaba0f7fa3ee60355137f` (147,143,540 bytes).

All upstream Git/LFS digests were verified during acquisition; the analysis rechecked the sealed original before and after each complete phase.
