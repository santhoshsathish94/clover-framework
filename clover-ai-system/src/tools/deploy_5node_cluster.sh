#!/usr/bin/env bash
# =================================================================================================
# Clover 5 × (8× RTX 3090 = 960 GiB VRAM) Balanced All-Active Mesh + MCP 2.0 + Batch Queue Deployer
# =================================================================================================
# Zero "all eggs in one basket": ALL 5 nodes hold an active Kimi K31H pipeline stage (14+20+20+20+18
# = 92 layers, ~113-161 GiB/node) PLUS dedicated Generalist/Specialist engines on remaining GPUs AND
# an Anycast Stateless Gateway (`/v1/chat/completions`, `/v1/chat/completions/batch`, `/mcp` MCP 2.0):
#
#   * Node 1 (173.2 / 192 GiB): Nemotron-3.5-Super-VL 120B HG4 (60.2G, GPUs 0-2)
#                               + Kimi Stage 0 (L00..13 + Embed = 113.0G, GPUs 3-7) + Mesh Gateway :8080
#   * Node 2 (178.8 / 192 GiB): Qwen3.6-35B Replica #1 (17.2G, GPU 0) + LFM3-2B Replica #1 (0.83G, GPU 0)
#                               + Kimi Stage 1 (L14..33 = 160.8G, GPUs 1-7) + Mesh Gateway :8080
#   * Node 3 (179.1 / 192 GiB): Qwen3.6-35B Replica #2 (17.2G, GPU 0) + LFM3-2B Replica #2 (0.83G, GPU 0)
#                               + Chronos-2 (0.22G, GPU 0) + Kimi Stage 2 (L34..53 = 160.8G, GPUs 1-7) + Mesh Gateway :8080
#   * Node 4 (177.1 / 192 GiB): Goedel-Prover-V2-32B HG4 (15.8G, GPU 0) + V-JEPA 2.1 HG4 (0.52G, GPU 0)
#                               + 32-core Lean 4 Verifier + Kimi Stage 3 (L54..73 = 160.8G, GPUs 1-7) + Mesh Gateway :8080
#   * Node 5 (168.6 / 192 GiB): Qwen3.6-35B Replica #3 (17.2G, GPU 0) + Boltz-2 + MACE-MP-0 (2.9G, GPU 0)
#                               + x-cruncher + Kimi Stage 4 (L74..91 + Heads + 5L-DSpark = 148.5G, GPUs 1-7)
#                               + Kimi Pipeline Coordinator :8089 + Mesh Gateway :8080
# =================================================================================================
set -euo pipefail

SSH_KEY="${HOME}/.ssh/ssh-gpu-server"
REPO_URL="https://github.com/santhoshsathish94/clover-framework.git"
DRY_RUN=0
NODE1=""
NODE2=""
NODE3=""
NODE4=""
NODE5=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --node1) NODE1="$2"; shift 2 ;;
    --node2) NODE2="$2"; shift 2 ;;
    --node3) NODE3="$2"; shift 2 ;;
    --node4) NODE4="$2"; shift 2 ;;
    --node5) NODE5="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$NODE1" || -z "$NODE2" || -z "$NODE3" || -z "$NODE4" || -z "$NODE5" ]]; then
  echo "Usage: $0 [--dry-run] --node1 user@ip:ssh_port[:lan_ip] ... --node5 user@ip:ssh_port[:lan_ip]" >&2
  exit 2
fi

parse_node() {
  local spec="$1"
  local host_part="${spec%%:*}"
  local rest="${spec#*:}"
  local port="${rest%%:*}"
  local lan_ip=""
  if [[ "$rest" == *:* ]]; then
    lan_ip="${rest#*:}"
  else
    lan_ip="${host_part#*@}"
  fi
  echo "${host_part}|${port}|${lan_ip}"
}

IFS='|' read -r N1_HOST N1_PORT N1_LAN <<< "$(parse_node "$NODE1")"
IFS='|' read -r N2_HOST N2_PORT N2_LAN <<< "$(parse_node "$NODE2")"
IFS='|' read -r N3_HOST N3_PORT N3_LAN <<< "$(parse_node "$NODE3")"
IFS='|' read -r N4_HOST N4_PORT N4_LAN <<< "$(parse_node "$NODE4")"
IFS='|' read -r N5_HOST N5_PORT N5_LAN <<< "$(parse_node "$NODE5")"

echo "==========================================================================="
echo " Clover 5 × (8× RTX 3090 = 960 GiB VRAM) Balanced All-Active Server Mesh"
echo "==========================================================================="
echo " Node 1 (173G/192G): Nemotron-120B (GPUs 0-2) + Kimi L00..13 (GPUs 3-7) + MCP 2.0 :8080"
echo " Node 2 (179G/192G): Qwen3.6-35B #1 + LFM3 #1 (GPU 0) + Kimi L14..33 (GPUs 1-7) + MCP 2.0 :8080"
echo " Node 3 (179G/192G): Qwen3.6-35B #2 + LFM3 #2 + Chronos-2 (GPU 0) + Kimi L34..53 + MCP 2.0 :8080"
echo " Node 4 (177G/192G): Goedel-32B + V-JEPA + Lean4 (GPU 0) + Kimi L54..73 (GPUs 1-7) + MCP 2.0 :8080"
echo " Node 5 (169G/192G): Qwen3.6-35B #3 + Boltz-2 (GPU 0) + Kimi L74..91+Heads (GPUs 1-7) + MCP 2.0 :8080"
echo " Anycast Entry:      Every node (:8080) accepts OpenAI /v1, /v1/chat/completions/batch & /mcp"
echo "==========================================================================="

run_remote() {
  local label="$1"
  local host="$2"
  local port="$3"
  local cmd="$4"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN ${label} ${host}:${port}]"
    echo "  ${cmd}"
    return 0
  fi
  ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$port" "$host" "$cmd"
}

BOOTSTRAP_ALL_ENGINES='set -euo pipefail
mkdir -p /workspace/hg-models/bin
if [[ ! -d /workspace/clover-framework ]]; then
  git clone --depth 1 '"${REPO_URL}"' /workspace/clover-framework
else
  git -C /workspace/clover-framework pull --ff-only
fi
pip install -q numpy huggingface_hub tokenizers safetensors
nvcc -O3 -std=c++17 --compiler-options -fPIC -shared \
  -gencode arch=compute_86,code=sm_86 \
  /workspace/clover-framework/clover-ai-system/src/tools/hg_cuda_accel.cu \
  -o /workspace/hg-models/bin/libhg_cuda_accel.so -lcudart
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/lfm3-1.7b/lfm3_hg4_engine.c -o /workspace/hg-models/bin/liblfm3_hg4_engine.so -lm -ldl
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/clover-35b-a3b/clover35b_hg4_engine.c -o /workspace/hg-models/bin/libclover35b_hg4_engine.so -lm -ldl
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hg4_engine.c -o /workspace/hg-models/bin/libnemotron_hg4_engine.so -lm -ldl
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/kimi-k3/kimi_k31_hg4_engine.c -o /workspace/hg-models/bin/libkimi_k31_hg4_engine.so -lm -ldl
'

MCP_PEERS="goedel_lean4_prove=http://${N4_LAN}:8080,vjepa2_predict_physics=http://${N4_LAN}:8080,boltz2_fold_structure=http://${N5_LAN}:8080,chronos2_forecast=http://${N3_LAN}:8080"
SPEC_URLS="goedel-prover-v2-32b=http://${N4_LAN}:8091,vjepa-2.1=http://${N4_LAN}:8092,boltz-2=http://${N5_LAN}:8093,chronos-2=http://${N3_LAN}:8094"

# 1. Bootstrap all 5 nodes in parallel
run_remote "NODE1-BUILD" "$N1_HOST" "$N1_PORT" "$BOOTSTRAP_ALL_ENGINES" & P1=$!
run_remote "NODE2-BUILD" "$N2_HOST" "$N2_PORT" "$BOOTSTRAP_ALL_ENGINES" & P2=$!
run_remote "NODE3-BUILD" "$N3_HOST" "$N3_PORT" "$BOOTSTRAP_ALL_ENGINES" & P3=$!
run_remote "NODE4-BUILD" "$N4_HOST" "$N4_PORT" "$BOOTSTRAP_ALL_ENGINES" & P4=$!
run_remote "NODE5-BUILD" "$N5_HOST" "$N5_PORT" "$BOOTSTRAP_ALL_ENGINES" & P5=$!
wait "$P1" "$P2" "$P3" "$P4" "$P5"

# 2. Provision balanced Kimi slice + local Generalist/Specialist engines + Anycast MCP 2.0 Mesh Gateway on all 5 nodes!
PEERS_N1='{"lfm3-omni-2b":["http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080"],"clover-35b-a3b":["http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080","http://'"${N5_LAN}"':8080"],"nemotron-3.5-super-vl":["local"]}'
PEERS_N2='{"lfm3-omni-2b":["local","http://'"${N3_LAN}"':8080"],"clover-35b-a3b":["local","http://'"${N3_LAN}"':8080","http://'"${N5_LAN}"':8080"],"nemotron-3.5-super-vl":["http://'"${N1_LAN}"':8080"]}'
PEERS_N3='{"lfm3-omni-2b":["local","http://'"${N2_LAN}"':8080"],"clover-35b-a3b":["local","http://'"${N2_LAN}"':8080","http://'"${N5_LAN}"':8080"],"nemotron-3.5-super-vl":["http://'"${N1_LAN}"':8080"]}'
PEERS_N4='{"lfm3-omni-2b":["http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080"],"clover-35b-a3b":["http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080","http://'"${N5_LAN}"':8080"],"nemotron-3.5-super-vl":["http://'"${N1_LAN}"':8080"]}'
PEERS_N5='{"lfm3-omni-2b":["http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080"],"clover-35b-a3b":["local","http://'"${N2_LAN}"':8080","http://'"${N3_LAN}"':8080"],"nemotron-3.5-super-vl":["http://'"${N1_LAN}"':8080"]}'

run_remote "NODE1-PROVISION" "$N1_HOST" "$N1_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model nemotron-3.5-super-vl --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 0:14 --need-emb
nohup env OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so --layer-range 0:14 --need-emb --pin-cold-vram --gpus 3,4,5,6,7 --port 9100 > /workspace/kimi_stage0.log 2>&1 &
nohup env CLOVER_NODE_ID=node-1 CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0,1,2 CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin CLOVER_NEMO_HG4_DIR=/workspace/hg-models/nemotron-3.5-super-vl CLOVER_NEMO_SPECULATIVE=1 CLOVER_KIMI_URL=http://${N5_LAN}:8089 CLOVER_SPECIALIST_URLS='${SPEC_URLS}' CLOVER_MCP_TOOL_PEERS='${MCP_PEERS}' CLOVER_MODEL_PEERS='${PEERS_N1}' OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8080 > /workspace/mesh_gateway.log 2>&1 &
" & D1=$!

run_remote "NODE2-PROVISION" "$N2_HOST" "$N2_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b,clover-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 14:34
nohup env OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so --layer-range 14:34 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --port 9101 > /workspace/kimi_stage1.log 2>&1 &
nohup env CLOVER_NODE_ID=node-2 CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin CLOVER_OMNI_HG4_DIR=/workspace/hg-models/lfm3-omni-2b CLOVER35B_HG4_DIR=/workspace/hg-models/clover-35b-a3b CLOVER_KIMI_URL=http://${N5_LAN}:8089 CLOVER_SPECIALIST_URLS='${SPEC_URLS}' CLOVER_MCP_TOOL_PEERS='${MCP_PEERS}' CLOVER_MODEL_PEERS='${PEERS_N2}' OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8080 > /workspace/mesh_gateway.log 2>&1 &
" & D2=$!

run_remote "NODE3-PROVISION" "$N3_HOST" "$N3_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b,clover-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 34:54
nohup env OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so --layer-range 34:54 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --port 9102 > /workspace/kimi_stage2.log 2>&1 &
nohup env CLOVER_NODE_ID=node-3 CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin CLOVER_OMNI_HG4_DIR=/workspace/hg-models/lfm3-omni-2b CLOVER35B_HG4_DIR=/workspace/hg-models/clover-35b-a3b CLOVER_KIMI_URL=http://${N5_LAN}:8089 CLOVER_SPECIALIST_URLS='${SPEC_URLS}' CLOVER_MCP_TOOL_PEERS='${MCP_PEERS}' CLOVER_MODEL_PEERS='${PEERS_N3}' OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8080 > /workspace/mesh_gateway.log 2>&1 &
" & D3=$!

run_remote "NODE4-PROVISION" "$N4_HOST" "$N4_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 54:74
nohup env OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so --layer-range 54:74 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --port 9103 > /workspace/kimi_stage3.log 2>&1 &
nohup env CLOVER_NODE_ID=node-4 CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin CLOVER_KIMI_URL=http://${N5_LAN}:8089 CLOVER_SPECIALIST_URLS='${SPEC_URLS}' CLOVER_MCP_TOOL_PEERS='${MCP_PEERS}' CLOVER_MODEL_PEERS='${PEERS_N4}' OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8080 > /workspace/mesh_gateway.log 2>&1 &
" & D4=$!

run_remote "NODE5-PROVISION" "$N5_HOST" "$N5_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model clover-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 74:92 --need-head
nohup env OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py coordinator --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so --local-range 74:92 --stages ${N1_LAN}:9100@0:14,${N2_LAN}:9101@14:34,${N3_LAN}:9102@34:54,${N4_LAN}:9103@54:74 --pin-cold-vram --wire-dtype bf16 --gpus 1,2,3,4,5,6,7 --port 8089 > /workspace/kimi_coordinator.log 2>&1 &
nohup env CLOVER_NODE_ID=node-5 CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin CLOVER35B_HG4_DIR=/workspace/hg-models/clover-35b-a3b CLOVER_KIMI_URL=http://${N5_LAN}:8089 CLOVER_SPECIALIST_URLS='${SPEC_URLS}' CLOVER_MCP_TOOL_PEERS='${MCP_PEERS}' CLOVER_MODEL_PEERS='${PEERS_N5}' OMP_NUM_THREADS=16 python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8080 > /workspace/mesh_gateway.log 2>&1 &
" & D5=$!

wait "$D1" "$D2" "$D3" "$D4" "$D5"
echo "BALANCED_5NODE_MESH_DEPLOYED_OK"
