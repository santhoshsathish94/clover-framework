#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/specmd/w2"
/root/ncm-cpu/venv/bin/python "$W2/validate_sim.py"
