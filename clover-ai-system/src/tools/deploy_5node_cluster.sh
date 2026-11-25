#!/usr/bin/env bash
# =================================================================================================
# Clover 6-Node "5-Petal Clover Leaf + Center k3 Hub" Provisioner (`deploy_5node_cluster.sh`)
# =================================================================================================
# Deploys Clover's 6-Node Topology (Opus 5.5 audited & verified against 21.9 GiB/GPU placeable VRAM):
#
#   * Center Hub `k3` (Permanent $0/hr 7950X3D AVX-512 + 128 GB DDR5 + Dual NVMe):
#       - Public Ingress, Global `ClusterBatchQueue` (Priority + T=0 Clubbing + Prefix Affinity)
#       - Dynamic Petal Lease Registry (`POST /v1/cluster/register`), Durable SQLite `tasks/*`
#       - C11 `x-cruncher` Exact Math, Lean 4 Verifier, Epistemic Memory
#       - Local CPU Replica #3: `Qwen3.6-35B` + `LFM3-2B` (18 GiB DDR5 — stays up even at 0 GPU Petals!)
#
#   * Petal 1 (8x RTX 3090): GPUs 0-2 Nemotron-120B (60.5/65.7G)  | GPUs 3-7 Kimi L00:14+Emb (104.4/109.5G)
#   * Petal 2 (8x RTX 3090): GPU 0 35B #1 + LFM3 #1 (18.3/21.9G)  | GPUs 1-7 Kimi L14:33     (151.7/153.3G)
#   * Petal 3 (8x RTX 3090): GPU 0 35B #2 + LFM3 #2 + Chronos     | GPUs 1-7 Kimi L33:52     (151.7/153.3G)
#   * Petal 4 (8x RTX 3090): GPU 0 Goedel-32B + V-JEPA + Boltz-2  | GPUs 1-7 Kimi L52:71     (151.8/153.3G)
#   * Petal 5 (8x RTX 3090): GPUs 0-7 Kimi L71:93 (ALL 93L 0..92!) + lm_head + 5L-DSpark + Coordinator :8089
# =================================================================================================
set -euo pipefail

SSH_KEY="${HOME}/.ssh/ssh-gpu-server"
REPO_URL="https://github.com/santhoshsathish94/clover-framework.git"
API_KEY="${CLOVER_GATEWAY_API_KEY:-}"
CLUSTER_SECRET="${CLOVER_CLUSTER_SECRET:-}"
ALLOW_PUBLIC_BIND=0
RUNTIME_MODE="auto"   # `auto` (uses `Dockerfile.hg-server` when `docker info` succeeds, else container-native `clover_docker_entry.py`), `docker`, or `native`
DRY_RUN=0
K3_CENTER=""
NODE1=""
NODE2=""
NODE3=""
NODE4=""
NODE5=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --cluster-secret) CLUSTER_SECRET="$2"; shift 2 ;;
    --runtime) RUNTIME_MODE="$2"; shift 2 ;;
    --docker) RUNTIME_MODE="docker"; shift ;;
    --bare-metal) RUNTIME_MODE="native"; shift ;;
    --k3-center) K3_CENTER="$2"; shift 2 ;;
    --node1|--petal1) NODE1="$2"; shift 2 ;;
    --node2|--petal2) NODE2="$2"; shift 2 ;;
    --node3|--petal3) NODE3="$2"; shift 2 ;;
    --node4|--petal4) NODE4="$2"; shift 2 ;;
    --node5|--petal5) NODE5="$2"; shift 2 ;;
    --allow-public-bind) ALLOW_PUBLIC_BIND=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$NODE1" || -z "$NODE2" || -z "$NODE3" || -z "$NODE4" || -z "$NODE5" ]]; then
  echo "Usage: $0 [--dry-run] [--k3-center user@k3:22:wg_ip] --petal1 user@ip:port:wg_ip ... --petal5 user@ip:port:wg_ip" >&2
  exit 2
fi

if [[ -z "$API_KEY" ]]; then
  API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(24))' 2>/dev/null || openssl rand -hex 24)"
fi
if [[ -z "$CLUSTER_SECRET" ]]; then
  CLUSTER_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))' 2>/dev/null || openssl rand -hex 32)"
fi

parse_node() {
  local spec="$1"
  local host_part="${spec%%:*}"
  local rest="${spec#*:}"
  local port="${rest%%:*}"
  local wg_ip=""
  if [[ "$rest" == *:* ]]; then
    wg_ip="${rest#*:}"
  elif [[ "$ALLOW_PUBLIC_BIND" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
    wg_ip="${host_part#*@}"
  else
    echo "ERROR: Node '$spec' omits private/WireGuard IP (:wg_ip). Pass user@pub_ip:ssh_port:10.x.x.x or --allow-public-bind." >&2
    exit 2
  fi
  echo "${host_part}|${port}|${wg_ip}"
}

IFS='|' read -r N1_HOST N1_PORT N1_WG <<< "$(parse_node "$NODE1")"
IFS='|' read -r N2_HOST N2_PORT N2_WG <<< "$(parse_node "$NODE2")"
IFS='|' read -r N3_HOST N3_PORT N3_WG <<< "$(parse_node "$NODE3")"
IFS='|' read -r N4_HOST N4_PORT N4_WG <<< "$(parse_node "$NODE4")"
IFS='|' read -r N5_HOST N5_PORT N5_WG <<< "$(parse_node "$NODE5")"

echo "================================================================================"
echo " Clover 6-Node Topology: Center Hub k3 + 5 x (8x RTX 3090) GPU Petals (93/93L)"
echo "================================================================================"
echo " Petal 1 (GPUs 0-2 Nemotron-120B | GPUs 3-7 Kimi L00:14+Emb): ${N1_WG}"
echo " Petal 2 (GPU 0 35B #1 + LFM3 #1 | GPUs 1-7 Kimi L14:33):     ${N2_WG}"
echo " Petal 3 (GPU 0 35B #2 + Chronos | GPUs 1-7 Kimi L33:52):     ${N3_WG}"
echo " Petal 4 (GPU 0 Goedel+VJEPA+Bio | GPUs 1-7 Kimi L52:71):     ${N4_WG}"
echo " Petal 5 (GPUs 0-7 Kimi L71:93 + Heads + Coordinator :8089):  ${N5_WG}"
[[ -n "$K3_CENTER" ]] && echo " Center Hub k3 (Global Queue + Clubbing + MCP 2.0 + 35B #3):  ${K3_CENTER}"
echo " Auth: Bearer API key + HMAC-SHA256 TCP worker handshake active"
echo "================================================================================"

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
mkdir -p /workspace/hg-models/bin /workspace/hg-models/clover-context
if [[ ! -d /workspace/clover-framework ]]; then
  git clone --depth 1 '"${REPO_URL}"' /workspace/clover-framework
else
  chmod u+w /workspace/clover-framework/AGENTS.md 2>/dev/null || true
  git -C /workspace/clover-framework pull --ff-only
fi
# Runtime enforcement (AGENTS.md §4): mount/lock AGENTS.md read-only (0444) so no agent can rewrite its governing rules
chmod 0444 /workspace/clover-framework/AGENTS.md
if [[ "'"${RUNTIME_MODE}"'" == "docker" ]] || { [[ "'"${RUNTIME_MODE}"'" == "auto" ]] && docker info >/dev/null 2>&1; }; then
  docker build -f /workspace/clover-framework/Dockerfile.hg-server -t clover-hg-server:latest /workspace/clover-framework
  echo "DOCKER_READY=1" > /workspace/.clover_runtime
else
  pip install -q numpy huggingface_hub tokenizers safetensors
  nvcc -O3 -std=c++17 --compiler-options -fPIC -shared \
    -gencode arch=compute_86,code=sm_86 \
    /workspace/clover-framework/clover-ai-system/src/tools/hg_cuda_accel.cu \
    -o /workspace/hg-models/bin/libhg_cuda_accel.so -lcudart
  gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/lfm3-1.7b/lfm3_hg4_engine.c -o /workspace/hg-models/bin/liblfm3_hg4_engine.so -lm -ldl
  gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/clover-35b-a3b/clover35b_hg4_engine.c -o /workspace/hg-models/bin/libclover35b_hg4_engine.so -lm -ldl
  gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hg4_engine.c -o /workspace/hg-models/bin/libnemotron_hg4_engine.so -lm -ldl
  gcc -O3 -march=native -mavx2 -mfma -mf16c -fopenmp -fPIC -shared -I/workspace/clover-framework/clover-ai-system/src/models /workspace/clover-framework/clover-ai-system/src/models/kimi-k3/kimi_k31_hg4_engine.c -o /workspace/hg-models/bin/libkimi_k31_hg4_engine.so -lm -ldl
  echo "DOCKER_READY=0" > /workspace/.clover_runtime
fi
cat << "WRAP" > /workspace/run_clover_role.sh
#!/usr/bin/env bash
set -euo pipefail
ROLE_NAME="$1"; shift
source /workspace/.clover_runtime
if [[ "${DOCKER_READY:-0}" == "1" ]]; then
  docker rm -f "clover-${ROLE_NAME}" >/dev/null 2>&1 || true
  ENV_ARGS=()
  for kv in CLOVER_NODE_ID CLOVER_GATEWAY_API_KEY CLOVER_CLUSTER_SECRET CLOVER_SHARE_CPU_LOCK CLOVER_CUDA_GPUS CLOVER_ACTIVE_MODELS CLOVER_NEMO_SPECULATIVE CLOVER_SPECIALIST_URLS CLOVER_OMNI_HG4_DIR CLOVER35B_HG4_DIR CLOVER_NEMO_HG4_DIR OMP_NUM_THREADS OMP_WAIT_POLICY; do
    [[ -n "${!kv:-}" ]] && ENV_ARGS+=(-e "${kv}=${!kv}")
  done
  docker run -d --name "clover-${ROLE_NAME}" --restart unless-stopped --gpus all --network host \
    -v /workspace/hg-models:/workspace/hg-models \
    -v /workspace/hg-models:/models \
    -v /workspace/clover-framework/AGENTS.md:/app/AGENTS.md:ro \
    -e CLOVER_AGENTS_MD=/app/AGENTS.md \
    -e CLOVER_CONTEXT_DIR=/models/clover-context \
    -e OMP_WAIT_POLICY=PASSIVE \
    "${ENV_ARGS[@]}" clover-hg-server:latest "$@"
else
  nohup env OMP_WAIT_POLICY=PASSIVE CLOVER_AGENTS_MD=/workspace/clover-framework/AGENTS.md CLOVER_CONTEXT_DIR=/workspace/hg-models/clover-context \
    python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_docker_entry.py "$@" > "/workspace/${ROLE_NAME}.log" 2>&1 &
fi
WRAP
chmod +x /workspace/run_clover_role.sh
'

# 1. Bootstrap all 5 GPU Petals in parallel
run_remote "PETAL1-BUILD" "$N1_HOST" "$N1_PORT" "$BOOTSTRAP_ALL_ENGINES" & P1=$!
run_remote "PETAL2-BUILD" "$N2_HOST" "$N2_PORT" "$BOOTSTRAP_ALL_ENGINES" & P2=$!
run_remote "PETAL3-BUILD" "$N3_HOST" "$N3_PORT" "$BOOTSTRAP_ALL_ENGINES" & P3=$!
run_remote "PETAL4-BUILD" "$N4_HOST" "$N4_PORT" "$BOOTSTRAP_ALL_ENGINES" & P4=$!
run_remote "PETAL5-BUILD" "$N5_HOST" "$N5_PORT" "$BOOTSTRAP_ALL_ENGINES" & P5=$!
wait "$P1" "$P2" "$P3" "$P4" "$P5"

# 2. Download exact slices + launch Kimi workers & LOCAL-ONLY Petal Gateways via `run_clover_role.sh` (Docker-first!)
run_remote "PETAL1-START" "$N1_HOST" "$N1_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model nemotron-3.5-super-vl --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 0:14 --need-emb
CLOVER_CLUSTER_SECRET='${CLUSTER_SECRET}' OMP_NUM_THREADS=16 /workspace/run_clover_role.sh kimi-stage0 kimi-worker --model-dir /workspace/hg-models/kimi-slice --layer-range 0:14 --need-emb --pin-cold-vram --gpus 3,4,5,6,7 --host ${N1_WG} --port 9100
CLOVER_NODE_ID=petal-1 CLOVER_GATEWAY_API_KEY='${API_KEY}' CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0,1,2 CLOVER_ACTIVE_MODELS=nemotron-3.5-super-vl CLOVER_NEMO_HG4_DIR=/workspace/hg-models/nemotron-3.5-super-vl CLOVER_NEMO_SPECULATIVE=1 OMP_NUM_THREADS=16 /workspace/run_clover_role.sh petal-gw gateway --mode real --host ${N1_WG} --port 8780
" & D1=$!

run_remote "PETAL2-START" "$N2_HOST" "$N2_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b,qwen3.6-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 14:33
CLOVER_CLUSTER_SECRET='${CLUSTER_SECRET}' OMP_NUM_THREADS=16 /workspace/run_clover_role.sh kimi-stage1 kimi-worker --model-dir /workspace/hg-models/kimi-slice --layer-range 14:33 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --host ${N2_WG} --port 9101
CLOVER_NODE_ID=petal-2 CLOVER_GATEWAY_API_KEY='${API_KEY}' CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_ACTIVE_MODELS=lfm3-omni-2b,clover-35b-a3b CLOVER_OMNI_HG4_DIR=/workspace/hg-models/lfm3-omni-2b CLOVER35B_HG4_DIR=/workspace/hg-models/qwen3.6-35b-a3b OMP_NUM_THREADS=16 /workspace/run_clover_role.sh petal-gw gateway --mode real --host ${N2_WG} --port 8780
" & D2=$!

run_remote "PETAL3-START" "$N3_HOST" "$N3_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b,qwen3.6-35b-a3b --download-only --cache-dir /workspace/hg-models
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"amazon/chronos-bolt-base\", local_dir=\"/workspace/hg-models/chronos-2\")' || true
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 33:52
CLOVER_GATEWAY_API_KEY='${API_KEY}' /workspace/run_clover_role.sh spec-chronos2 specialist --model chronos-2 --model-dir /workspace/hg-models/chronos-2 --gpu 0 --host 127.0.0.1 --port 8094
CLOVER_CLUSTER_SECRET='${CLUSTER_SECRET}' OMP_NUM_THREADS=16 /workspace/run_clover_role.sh kimi-stage2 kimi-worker --model-dir /workspace/hg-models/kimi-slice --layer-range 33:52 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --host ${N3_WG} --port 9102
CLOVER_NODE_ID=petal-3 CLOVER_GATEWAY_API_KEY='${API_KEY}' CLOVER_SHARE_CPU_LOCK=0 CLOVER_CUDA_GPUS=0 CLOVER_ACTIVE_MODELS=lfm3-omni-2b,clover-35b-a3b CLOVER_SPECIALIST_URLS='chronos-2=http://127.0.0.1:8094' CLOVER_OMNI_HG4_DIR=/workspace/hg-models/lfm3-omni-2b CLOVER35B_HG4_DIR=/workspace/hg-models/qwen3.6-35b-a3b OMP_NUM_THREADS=16 /workspace/run_clover_role.sh petal-gw gateway --mode real --host ${N3_WG} --port 8780
" & D3=$!

run_remote "PETAL4-START" "$N4_HOST" "$N4_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 52:71
python3 -c 'from huggingface_hub import snapshot_download; [snapshot_download(r, local_dir=d) for r, d in [(\"Santhoshsathish94/Goedel-Prover-V2-32B-HG4-16R\", \"/workspace/hg-models/goedel-prover-v2-32b\"), (\"Santhoshsathish94/VJEPA2-ViTg-HG4-16R\", \"/workspace/hg-models/vjepa-2.1\"), (\"boltz-community/boltz-2\", \"/workspace/hg-models/boltz-2\")]]' || true
CLOVER_GATEWAY_API_KEY='${API_KEY}' /workspace/run_clover_role.sh spec-goedel specialist --model goedel-prover-v2-32b --model-dir /workspace/hg-models/goedel-prover-v2-32b --gpu 0 --host 127.0.0.1 --port 8091
CLOVER_GATEWAY_API_KEY='${API_KEY}' /workspace/run_clover_role.sh spec-vjepa specialist --model vjepa-2.1 --model-dir /workspace/hg-models/vjepa-2.1 --gpu 0 --host 127.0.0.1 --port 8092
CLOVER_GATEWAY_API_KEY='${API_KEY}' /workspace/run_clover_role.sh spec-boltz specialist --model boltz-2 --model-dir /workspace/hg-models/boltz-2 --gpu 0 --host 127.0.0.1 --port 8093
CLOVER_CLUSTER_SECRET='${CLUSTER_SECRET}' OMP_NUM_THREADS=16 /workspace/run_clover_role.sh kimi-stage3 kimi-worker --model-dir /workspace/hg-models/kimi-slice --layer-range 52:71 --pin-cold-vram --gpus 1,2,3,4,5,6,7 --host ${N4_WG} --port 9103
CLOVER_NODE_ID=petal-4 CLOVER_GATEWAY_API_KEY='${API_KEY}' CLOVER_SHARE_CPU_LOCK=0 CLOVER_ACTIVE_MODELS='' CLOVER_SPECIALIST_URLS='goedel-prover-v2-32b=http://127.0.0.1:8091,vjepa-2.1=http://127.0.0.1:8092,boltz-2=http://127.0.0.1:8093' OMP_NUM_THREADS=16 /workspace/run_clover_role.sh petal-gw gateway --mode real --host ${N4_WG} --port 8780
" & D4=$!

run_remote "PETAL5-START" "$N5_HOST" "$N5_PORT" "
python3 /workspace/clover-framework/clover-ai-system/src/tools/kimi_pipeline_cluster.py download-slice --out /workspace/hg-models/kimi-slice --layer-range 71:93 --need-emb --need-head
CLOVER_CLUSTER_SECRET='${CLUSTER_SECRET}' CLOVER_GATEWAY_API_KEY='${API_KEY}' OMP_NUM_THREADS=24 /workspace/run_clover_role.sh kimi-coord kimi-coordinator --model-dir /workspace/hg-models/kimi-slice --local-range 71:93 --stages ${N1_WG}:9100@0:14,${N2_WG}:9101@14:33,${N3_WG}:9102@33:52,${N4_WG}:9103@52:71 --pin-cold-vram --wire-dtype fp32 --gpus 0,1,2,3,4,5,6,7 --host ${N5_WG} --port 8089
" & D5=$!

wait "$D1" "$D2" "$D3" "$D4" "$D5"

# 3. If `--k3-center` is provided, start/update Center Hub `k3` and launch 30s lease heartbeats on Petals 1..4!
if [[ -n "$K3_CENTER" ]]; then
  IFS='|' read -r K3_HOST K3_PORT K3_WG <<< "$(parse_node "$K3_CENTER")"
  K3_PEERS='{"lfm3-omni-2b":["local"],"clover-35b-a3b":["local"]}'
  run_remote "K3-CENTER-HUB" "$K3_HOST" "$K3_PORT" "
    mkdir -p /mnt/nvme1/clover-hub/context_files
    chmod 0444 /mnt/nvme1/clover-framework-repo/AGENTS.md 2>/dev/null || true
    nohup env CLOVER_NODE_ID=k3-center CLOVER35B_HG4_DIR=/mnt/nvme1/hg_specialists/qwen3.6-35b-a3b-hg4 CLOVER_AGENTS_MD=/mnt/nvme1/clover-framework-repo/AGENTS.md CLOVER_CONTEXT_DIR=/mnt/nvme1/clover-hub/context_files CLOVER_GATEWAY_API_KEY='${API_KEY}' CLOVER_TASKS_DB=/mnt/nvme1/clover-hub/mcp_tasks.sqlite3 CLOVER_KIMI_URL=http://${N5_WG}:8089 CLOVER_MODEL_PEERS='${K3_PEERS}' OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY=0-15 python3 /mnt/nvme1/clover-framework-repo/clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8790 > /mnt/nvme1/clover-hub/k3_hub.log 2>&1 &
  "
  start_hb() {
    local lbl="$1" h="$2" p="$3" pid="$4" wg="$5" mdls="$6" tls="$7"
    run_remote "$lbl" "$h" "$p" "nohup bash -c 'while true; do curl -fsS -m 5 -X POST http://${K3_WG}:8790/v1/cluster/register -H \"Authorization: Bearer ${API_KEY}\" -H \"Content-Type: application/json\" -d \"{\\\"petal_id\\\":\\\"${pid}\\\",\\\"url\\\":\\\"http://${wg}:8780\\\",\\\"models\\\":${mdls},\\\"mcp_tools\\\":${tls}}\" >/dev/null 2>&1 || true; sleep 30; done' > /workspace/petal_hb.log 2>&1 &"
  }
  start_hb "PETAL1-HB" "$N1_HOST" "$N1_PORT" "petal-1" "$N1_WG" '["nemotron-3.5-super-vl"]' '[]'
  start_hb "PETAL2-HB" "$N2_HOST" "$N2_PORT" "petal-2" "$N2_WG" '["lfm3-omni-2b","clover-35b-a3b"]' '[]'
  start_hb "PETAL3-HB" "$N3_HOST" "$N3_PORT" "petal-3" "$N3_WG" '["lfm3-omni-2b","clover-35b-a3b","chronos-2"]' '["chronos2_forecast"]'
  start_hb "PETAL4-HB" "$N4_HOST" "$N4_PORT" "petal-4" "$N4_WG" '["goedel-prover-v2-32b","vjepa-2.1","boltz-2"]' '["goedel_lean4_prove","vjepa2_predict_physics","boltz2_fold_structure"]'
fi

echo "CLOVER_6NODE_5PETAL_K3_DEPLOYED_OK"
