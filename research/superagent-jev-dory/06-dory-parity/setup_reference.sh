#!/bin/bash
# Dory parity, step 1: independent reference (llama.cpp PR #30306, brianchmielNV:dory-support), CPU build, BF16 GGUF.
# Writes only under $D. Detached; progress in $D/setup.log.
set -uo pipefail
D=/opt/clover-k3/superagent-jev-dory-study/dory-parity
mkdir -p "$D" && cd "$D" || exit 1
exec >> setup.log 2>&1
stamp() { echo "$(date -u +%FT%TZ) $*"; }
stamp "start"
if [ ! -d llama.cpp-dory/.git ]; then
  git clone --branch dory-support --single-branch https://github.com/brianchmielNV/llama.cpp.git llama.cpp-dory || { stamp "FAILED clone"; exit 1; }
fi
git -C llama.cpp-dory log -1 --format='%H %cI %s' > pr_head.txt
stamp "head $(cat pr_head.txt)"
if [ ! -x venv/bin/python ]; then python3.12 -m venv venv || { stamp "FAILED venv"; exit 1; }; fi
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q cmake ninja || { stamp "FAILED cmake"; exit 1; }
venv/bin/pip install -q -r llama.cpp-dory/requirements/requirements-convert_hf_to_gguf.txt || { stamp "FAILED convert requirements"; exit 1; }
venv/bin/python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
export PATH="$D/venv/bin:$PATH"
cmake -S llama.cpp-dory -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF || { stamp "FAILED configure"; exit 1; }
nice -n 10 cmake --build build -j 16 --target llama-server llama-cli || { stamp "FAILED build"; exit 1; }
stamp "built"
if [ ! -s dory-4b-bf16.gguf ]; then
  nice -n 10 venv/bin/python llama.cpp-dory/convert_hf_to_gguf.py /opt/clover-k3/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
    --outfile dory-4b-bf16.gguf --outtype bf16 || { stamp "FAILED convert"; exit 1; }
fi
sha256sum dory-4b-bf16.gguf > dory-4b-bf16.gguf.sha256
stamp "SETUP_DONE"
