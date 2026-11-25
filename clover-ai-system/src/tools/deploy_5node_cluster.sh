#!/usr/bin/env bash
# =================================================================================================
# Clover 5 × (8× RTX 3090) All-VRAM Multi-Server Cluster Provisioner
# =================================================================================================
# Deploys the entire 8-model Clover Specialist + Generalist Cluster across 5 same-region
# 8× RTX 3090 (192 GiB VRAM each = 960 GiB total VRAM) nodes with <1 MB inter-node TCP hops:
#
#   * Node 1 (192 GiB VRAM): All 7 Generalist & Specialist Engines + `clover-fusion-cluster` Router
#       - GPU 0 (24 GiB): LFM3-Omni-2B (0.83G) + V-JEPA 2.1 (0.52G) + Chronos-2 (0.22G) + Boltz-2 (2.8G)
#       - GPU 1 (24 GiB): Qwen3.6-35B-A3B HG4-16R (17.2G) — Layer-2 Coordinator & Fast Coding Agent
#       - GPU 2 (24 GiB): Goedel-Prover-V2-32B HG4-16R (15.8G) — Lean 4 Formal Theorem Prover
#       - GPUs 3-6 (96 GiB): Nemotron-3.5-Super-VL 120B HG4 + 2L-MTP (60.2G) — Tier-2 Escalation
#       - GPU 7 (24 GiB): Spare / KV Burst Headroom (94 GiB total free VRAM across Node 1)
#
#   * Node 2 (192 GiB VRAM): Kimi-K3.1-Pro 2.8T K31H Pipeline Stage 0 (Layers 0..22, ~185 GiB VRAM)
#   * Node 3 (192 GiB VRAM): Kimi-K3.1-Pro 2.8T K31H Pipeline Stage 1 (Layers 23..45, ~185 GiB VRAM)
#   * Node 4 (192 GiB VRAM): Kimi-K3.1-Pro 2.8T K31H Pipeline Stage 2 (Layers 46..68, ~185 GiB VRAM)
#   * Node 5 (192 GiB VRAM): Kimi-K3.1-Pro 2.8T K31H Pipeline Stage 3 (Layers 69..91 + Heads, ~188 GiB)
#                            + OpenAI `/v1/chat/completions` Pipeline Coordinator on port 8089
#
# Usage:
#   bash deploy_5node_cluster.sh \
#     --ssh-key ~/.ssh/ssh-gpu-server \
#     --node1 root@IP1:PORT1:INTERNAL_IP1 \
#     --node2 root@IP2:PORT2:INTERNAL_IP2 \
#     --node3 root@IP3:PORT3:INTERNAL_IP3 \
#     --node4 root@IP4:PORT4:INTERNAL_IP4 \
#     --node5 root@IP5:PORT5:INTERNAL_IP5
#
#   Pass `--dry-run` to validate arguments and inspect generated per-node commands at $0 cost.
# =================================================================================================
set -euo pipefail

SSH_KEY="${HOME}/.ssh/ssh-gpu-server"
REPO_URL="https://github.com/santhoshsathish94/clover-framework.git"
KIMI_HF_REPO="Santhoshsathish94/Clover-Kimi-K3.1-Pro-2.8T-K31H-HG4-HG2"
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
    --kimi-repo) KIMI_HF_REPO="$2"; shift 2 ;;
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

echo "====================================================================="
echo " Clover 5 × (8× RTX 3090 = 960 GiB VRAM) All-VRAM Cluster Plan"
echo "====================================================================="
echo " Node 1 (Router + 7 Models): ${N1_HOST}:${N1_PORT} (LAN ${N1_LAN}:8080)"
echo " Node 2 (Kimi L00..22):      ${N2_HOST}:${N2_PORT} (TCP ${N2_LAN}:9100)"
echo " Node 3 (Kimi L23..45):      ${N3_HOST}:${N3_PORT} (TCP ${N3_LAN}:9101)"
echo " Node 4 (Kimi L46..68):      ${N4_HOST}:${N4_PORT} (TCP ${N4_LAN}:9102)"
echo " Node 5 (Kimi L69..91+Head): ${N5_HOST}:${N5_PORT} (HTTP ${N5_LAN}:8089)"
echo " Wire payload per token:     231 KB/hop (BF16) = 693 KB total across 3 hops"
echo "====================================================================="

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

BOOTSTRAP_BUILD='set -euo pipefail
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
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared \
  -I/workspace/clover-framework/clover-ai-system/src/models \
  /workspace/clover-framework/clover-ai-system/src/models/kimi-k3/kimi_k31_hg4_engine.c \
  -o /workspace/hg-models/bin/libkimi_k31_hg4_engine.so -lm -ldl
'

# 1. Bootstrap all 5 nodes in parallel (clone + compile CUDA sm_86 + C11 SIMD)
run_remote "NODE1-BUILD" "$N1_HOST" "$N1_PORT" "$BOOTSTRAP_BUILD
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/lfm3-1.7b/lfm3_hg4_engine.c -o /workspace/hg-models/bin/liblfm3_hg4_engine.so -lm -ldl
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/clover-35b-a3b/clover35b_hg4_engine.c -o /workspace/hg-models/bin/libclover35b_hg4_engine.so -lm -ldl
gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hg4_engine.c -o /workspace/hg-models/bin/libnemotron_hg4_engine.so -lm -ldl
" &
P1=$!
run_remote "NODE2-BUILD" "$N2_HOST" "$N2_PORT" "$BOOTSTRAP_BUILD" &
P2=$!
run_remote "NODE3-BUILD" "$N3_HOST" "$N3_PORT" "$BOOTSTRAP_BUILD" &
P3=$!
run_remote "NODE4-BUILD" "$N4_HOST" "$N4_PORT" "$BOOTSTRAP_BUILD" &
P4=$!
run_remote "NODE5-BUILD" "$N5_HOST" "$N5_PORT" "$BOOTSTRAP_BUILD" &
P5=$!
wait "$P1" "$P2" "$P3" "$P4" "$P5"

# 2. Download weights in parallel across all 5 independent NICs (each node downloads <=188 GB!)
run_remote "NODE2-STAGE0" "$N2_HOST" "$N2_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice \
  --out /workspace/hg-models/kimi-slice --layer-range 0:23 --need-emb
nohup env OMP_NUM_THREADS=32 \
  python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker \
  --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so \
  --layer-range 0:23 --need-emb --pin-cold-vram --gpus 0,1,2,3,4,5,6,7 --port 9100 > /workspace/kimi_stage0.log 2>&1 &
" &
D2=$!

run_remote "NODE3-STAGE1" "$N3_HOST" "$N3_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice \
  --out /workspace/hg-models/kimi-slice --layer-range 23:46
nohup env OMP_NUM_THREADS=32 \
  python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker \
  --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so \
  --layer-range 23:46 --pin-cold-vram --gpus 0,1,2,3,4,5,6,7 --port 9101 > /workspace/kimi_stage1.log 2>&1 &
" &
D3=$!

run_remote "NODE4-STAGE2" "$N4_HOST" "$N4_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice \
  --out /workspace/hg-models/kimi-slice --layer-range 46:69
nohup env OMP_NUM_THREADS=32 \
  python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py worker \
  --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so \
  --layer-range 46:69 --pin-cold-vram --gpus 0,1,2,3,4,5,6,7 --port 9102 > /workspace/kimi_stage2.log 2>&1 &
" &
D4=$!

run_remote "NODE5-TAIL" "$N5_HOST" "$N5_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice \
  --out /workspace/hg-models/kimi-slice --layer-range 69:92 --need-head
nohup env OMP_NUM_THREADS=32 \
  python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py coordinator \
  --model-dir /workspace/hg-models/kimi-slice --lib-path /workspace/hg-models/bin/libkimi_k31_hg4_engine.so \
  --local-range 69:92 \
  --stages ${N2_LAN}:9100@0:23,${N3_LAN}:9101@23:46,${N4_LAN}:9102@46:69 \
  --pin-cold-vram --wire-dtype bf16 --gpus 0,1,2,3,4,5,6,7 --port 8089 > /workspace/kimi_coordinator.log 2>&1 &
" &
D5=$!

run_remote "NODE1-ROUTER" "$N1_HOST" "$N1_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model clover-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model nemotron-3.5-super-vl --download-only --cache-dir /workspace/hg-models
nohup env \
  CLOVER_NATIVE_LIB_DIR=/workspace/hg-models/bin \
  CLOVER_OMNI_HG4_DIR=/workspace/hg-models/lfm3-omni-2b \
  CLOVER35B_HG4_DIR=/workspace/hg-models/clover-35b-a3b \
  CLOVER_NEMO_HG4_DIR=/workspace/hg-models/nemotron-3.5-super-vl \
  CLOVER_NEMO_SPECULATIVE=1 \
  CLOVER_KIMI_URL=http://${N5_LAN}:8089 \
  OMP_NUM_THREADS=32 \
  python3 /workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py \
  --host 0.0.0.0 --port 8080 > /workspace/cluster_gateway.log 2>&1 &
" &
D1=$!

wait "$D1" "$D2" "$D3" "$D4" "$D5"
echo "CLUSTER_DEPLOYMENT_DISPATCHED_OK"
