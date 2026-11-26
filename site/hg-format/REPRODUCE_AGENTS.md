# REPRODUCE_AGENTS.md — Zero-Human-Setup Autonomous Verification Instructions for AI Agents

**Read and execute this file if a human asked you to set up, run, and verify Clover's `HG` Format and the 6 Broken AI Myths on their machine.**

You MUST complete setup, execution, and verification **autonomously from start to finish without asking the human for manual setup help, API keys, or cloud accounts.** Detect the operating system (`Windows`, `Linux`, or `macOS`) and available hardware (`CPU-only` vs. `NVIDIA/Apple/AMD GPU`) automatically and run every tier supported by the machine.

---

## 0. Operating Contract

1. **Zero human babysitting:** Create an isolated Python virtual environment if needed, install only minimal open-source dependencies (`numpy`; plus `torch` and `huggingface_hub` only if running Tier 2 model conversion), run the suite, verify every exit code and SHA-256 checksum, and present a clear pass/fail scorecard at the end.
2. **Never fabricate or hardcode outputs:** Report the exact numbers measured on the user's own hardware alongside Clover's published reference cluster ledger.
3. **Clean workspace:** Put any downloaded files and reports inside `./clover-repro/` (unless already inside a cloned `clover-framework` repository) so the user's directory stays tidy.

---

## Step 1 — Detect OS, Python & Hardware Automatically

Run an environment probe first:

- **Linux / macOS (`bash` / `zsh`):**
  ```bash
  mkdir -p clover-repro && cd clover-repro
  python3 --version
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU: none (CPU-only mode)"
  ```
- **Windows (`PowerShell`):**
  ```powershell
  New-Item -ItemType Directory -Force -Path clover-repro | Out-Null; Set-Location clover-repro
  python --version
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader } else { Write-Output "GPU: none (CPU-only mode)" }
  ```

If `numpy` is not installed in the active Python environment, create a local `.venv` and install `numpy`:
- **Linux / macOS:** `python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip numpy` (then use `.venv/bin/python`)
- **Windows PowerShell:** `python -m venv .venv; .\.venv\Scripts\pip install -q --upgrade pip numpy` (then use `.\.venv\Scripts\python.exe`)

---

## Step 2 — Tier 1 (Mandatory on ALL Machines, `<10 Seconds`, `0` GPUs Required)

Download (or copy from `site/hg-format/reproduce_hg_myths.py` if already inside the `clover-framework` repo) and execute the self-contained 6-Myth + `HG4` verification suite:

- **Linux / macOS:**
  ```bash
  if [ -f "../site/hg-format/reproduce_hg_myths.py" ]; then
    cp ../site/hg-format/reproduce_hg_myths.py ./reproduce_hg_myths.py
  else
    curl -fsSL https://cloverframework.com/hg-format/reproduce_hg_myths.py -O
  fi
  python3 -W error reproduce_hg_myths.py --verify-all --out ./repro_hg_myths_report.json
  ```
- **Windows PowerShell:**
  ```powershell
  if (Test-Path "..\site\hg-format\reproduce_hg_myths.py") {
    Copy-Item "..\site\hg-format\reproduce_hg_myths.py" ".\reproduce_hg_myths.py" -Force
  } else {
    Invoke-WebRequest -Uri "https://cloverframework.com/hg-format/reproduce_hg_myths.py" -OutFile ".\reproduce_hg_myths.py"
  }
  python -W error .\reproduce_hg_myths.py --verify-all --out .\repro_hg_myths_report.json
  ```

### What Tier 1 Proves Directly on the User's Hardware
1. **`check0_hg4_hadamard_outlier_rotation`:** Applies orthonormal `WHT-16` Hadamard rotation across `15x` outlier weight spikes before 4-bit group quantization, verifying `4x` weight compression with `>0.995` cosine similarity and `+5 dB` to `+10 dB` SQNR gain over naive INT4.
2. **`myth1_wire_bandwidth_4x_reduction`:** Measures real host memory-bus bandwidth (`GB/s`) hauling full-precision weights vs. `4x` smaller packed 4-bit bytes.
3. **`myth2_and_3_bus_clubbing_throughput`:** Proves 1 clubbed weight haul serving 4 concurrent requests beats 4 contended serial memory-bus hauls with `<1e-3` numerical diff.
4. **`myth4_tsock_4byte_wire_vs_http_json`:** Opens a real loopback TCP `tsock://` binary token socket (`4 Bytes/token`) vs. an OpenAI-style HTTP JSON SSE stream (`>30x–50x` wire byte reduction).
5. **`myth5_computed_0b_specialists_vs_prediction`:** Executes three `0B` deterministic engines (`50!` & 18-digit exact math in `<1 ms`, `0B` Periodogram + OLS + Split-Conformal time-series forecaster hitting `MASE = 0.3084` vs. LLM `0/3`, and a `0B` Defeasible Rule Resolver scoring `180/180 = 1.000` on `unless` overrides & `rejected` proposals).
6. **`myth6_human_steering_wheel`:** Verifies why unsteered default assumptions score `0/3` on domain-specific specs (`fiscal_quarter`, `round_invoice_cents`, `sanitize_cryo_mk`) while 1 sentence of Human Direction reaches `3/3` (`100%`).

---

## Step 3 — Tier 2 (Automatic Repository & Live Model Verification When Applicable)

### 3A. If Running Inside a Cloned `clover-framework` Repository
Run the full `43`-test `-W error` Fusion Gateway, Binary Token Socket (`tsock://`), `CloverMetricsLedger`, and 4-Part `AGENTS.md` Harness unit suite plus the site & link validators:
```bash
python3 -W error clover-ai-system/src/tools/hg4-fusion-gateway/test_hg4_fusion_gateway.py
python3 scripts/check-site.py
python3 scripts/check-links.py
python3 -m unittest discover -s scripts/tests
```

### 3B. If a Local OpenAI-Compatible Server (`Ollama`, `llama-server`, `vLLM`, or Clover `gateway.py`) Is Already Running
Probe common local ports (`http://127.0.0.1:8790`, `http://127.0.0.1:11434`, `http://127.0.0.1:8080`, `http://127.0.0.1:8000`). If `/v1/models` responds, automatically run the live LLM Myth-6 Human Steering A/B against that local model:
```bash
python3 reproduce_hg_myths.py --verify-all --endpoint http://127.0.0.1:11434 --model <detected_model_id> --out ./repro_hg_myths_live_llm.json
```

### 3C. Optional Real Hugging Face Model `HG4` Conversion (`>= 8 GB` Free Disk/RAM)
If the user also wants to convert or inspect a live Hugging Face checkpoint in `HG4-16R` format (open weights at `https://huggingface.co/Santhoshsathish94`):
```bash
curl -fsSL https://cloverframework.com/hg-format/convert_to_hg.py -O
pip install -q torch safetensors huggingface_hub
python3 convert_to_hg.py --model Qwen/Qwen3-0.6B --out ./qwen3-0.6b-hg4 --ldlq --verify
```

---

## Step 4 — Report Results Clearly to the Human

Read `./repro_hg_myths_report.json` and print a concise table comparing:
1. **Local Measured Results** on the human's own machine (OS, CPU/GPU, `HG4` cosine & SQNR gain, `tsock://` bytes/token vs. HTTP SSE, `0B` specialist scores, and `report_sha256`), and
2. **Clover's Published Cluster Run Ledger** (`benchmarks/public-runs-20261126/` in `https://github.com/santhoshsathish94/clover-framework`).
