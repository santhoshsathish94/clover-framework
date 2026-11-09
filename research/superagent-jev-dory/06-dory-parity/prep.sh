#!/bin/bash
# Build the logits tool against the pull request's libllama, then write prompts.json + ids.txt.
D=/opt/clover-k3/superagent-jev-dory-study/dory-parity
cd "$D" || exit 1
g++ -O2 -std=c++17 dory_logits.cpp -I llama.cpp-dory/include -I llama.cpp-dory/ggml/include \
  -L build/bin -lllama -lggml -lggml-base -Wl,-rpath,"$D/build/bin" -o dory_logits || { echo "BUILD_FAILED"; exit 1; }
/root/hybridagent-venv/bin/python run_parity.py prompts || { echo "PROMPTS_FAILED"; exit 1; }
sha256sum prompts.json ids.txt PREREGISTRATION.md run_parity.py dory_logits.cpp
tail -n 2 convert_f32.log
echo PREP_DONE
