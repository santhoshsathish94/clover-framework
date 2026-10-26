#!/usr/bin/env bash
set -euo pipefail
/root/ncm-cpu/venv/bin/python - << 'PY'
import inspect
from transformers.models.olmoe import modeling_olmoe
print("=== OlmoeTopKRouter ===")
print(inspect.getsource(modeling_olmoe.OlmoeTopKRouter))
print("=== OlmoeSparseMoeBlock ===")
print(inspect.getsource(modeling_olmoe.OlmoeSparseMoeBlock))
print("=== OlmoeDecoderLayer.forward ===")
print(inspect.getsource(modeling_olmoe.OlmoeDecoderLayer.forward))
PY
