#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
mkdir -p "$BASE/work"

export PATH="/root/ncm-cpu/venv/bin:$PATH"
echo "=== 1. Run dmoews/logic-theorist verify.py & test.sh in a work copy ==="
rm -rf "$BASE/work/dmoews_ipli_run"
cp -r "$BASE/repo/dmoews-logic-theorist" "$BASE/work/dmoews_ipli_run"
cd "$BASE/work/dmoews_ipli_run"
$PY verify.py > /dev/null && echo "verify.py: OK (rc=0)"
cat test.sh
bash test.sh && echo "dmoews/logic-theorist test.sh: OK (rc=0)"

echo "=== 2. Diff orig-logic-routines.txt vs logic-routines.txt (1956 P-868 Section III errata) ==="
diff -u orig-logic-routines.txt logic-routines.txt || true

echo "=== 3. Run dmoews/ipl-v-logic-theorist tests in a work copy ==="
rm -rf "$BASE/work/dmoews_iplv_run"
cp -r "$BASE/repo/dmoews-ipl-v-lt" "$BASE/work/dmoews_iplv_run"
cd "$BASE/work/dmoews_iplv_run/interpreter"
cat test.sh
bash test.sh && echo "dmoews/ipl-v-logic-theorist test.sh: OK (rc=0)"
bash test-stefferud.sh && echo "1963 Stefferud IPL-V exact match: OK (rc=0)"

echo "=== 4. Read and run duplys/logic-theorist ==="
rm -rf "$BASE/work/duplys_run"
cp -r "$BASE/repo/duplys-logic-theorist" "$BASE/work/duplys_run"
cd "$BASE/work/duplys_run"
cat logic_theorist.py
$PY logic_theorist.py || echo "duplys rc=$?"
