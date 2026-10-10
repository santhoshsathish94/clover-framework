# Upstream engine investigation

Work involving engines other than Clover's own implementation.

## Kimi K3 itself, as its authors and its serving engines define it

Added after the discovery that every measurement in `clover-intelegence` was of
`kimi-k3-in-c`, a reimplementation, while the documents said "the model".

- [Official repository and tech report](moonshot-kimi-k3-official.md) — architecture,
  the chunkwise KDA form, KCP, and the decode regime the authors describe the way
  we measured it
- [vLLM execution](vllm-kimi-k3-execution.md) — how it is actually served:
  pipeline, tensor, sequence, expert and context parallelism

## kimi-k3-in-c

- [Engine explanation](fareed-khan-kimi-k3-in-c-explanation.md)
- [Measured results and limits](kimi-k3-measurements.md)
- [Local evidence record](kimi-k3-local-evidence.json)
- [Historical benchmark campaign](kimi-k3-bench-run.sh)

The campaign script is preserved unchanged. It can download large model artifacts and
perform machine setup; relocation is not permission or a recommendation to run it.
Its original commands and results remain in the [benchmark context](../context/CONTEXT-kimi-k3-benchmark.md).

[Investigation index](../README.md).