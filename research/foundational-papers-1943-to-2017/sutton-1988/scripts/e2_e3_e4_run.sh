#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/sutton-1988

# Create a tiny headless mock matplotlib so reference repos run without GUI/matplotlib package
mkdir -p "$BASE/work/mock_mpl/matplotlib"
cat << 'EOF' > "$BASE/work/mock_mpl/matplotlib/__init__.py"
class _Dummy:
    def __getattr__(self, name):
        return lambda *a, **kw: _Dummy()
    def __call__(self, *a, **kw):
        return _Dummy()
    def __iter__(self):
        return iter([_Dummy(), _Dummy()])
pyplot = _Dummy()
EOF
cat << 'EOF' > "$BASE/work/mock_mpl/matplotlib/pyplot.py"
class _Dummy:
    def __getattr__(self, name):
        return lambda *a, **kw: _Dummy()
    def __call__(self, *a, **kw):
        return _Dummy()
    def __iter__(self):
        return iter([_Dummy(), _Dummy()])
def __getattr__(name):
    if name == "subplots":
        return lambda *a, **kw: (_Dummy(), _Dummy())
    return lambda *a, **kw: _Dummy()
EOF

export PYTHONPATH="$BASE/work/mock_mpl:$PYTHONPATH"

echo "=== 1. Run robosangli/sutton-1988-reproduction/sutton1988.py ==="
rm -rf "$BASE/work/robosangli_run"
cp -r "$BASE/repo/robosangli-sutton-1988-reproduction" "$BASE/work/robosangli_run"
cd "$BASE/work/robosangli_run"
$PY sutton1988.py
echo ">>> Pre-computed CSV results in robosangli/results/best_alpha_summary.csv:"
cat results/best_alpha_summary.csv

echo "=== 2. Inspect & run MrShininnnnn/Temporal-Differences-Learning/main.py ==="
rm -rf "$BASE/work/mrshinin_run"
cp -r "$BASE/repo/MrShininnnnn-Temporal-Differences-Learning" "$BASE/work/mrshinin_run"
cd "$BASE/work/mrshinin_run"
sed -i 's/np.int/int/g' main.py
sed -i 's/np.float/float/g' main.py
echo ">>> Lines 125-210 of MrShininnnnn/main.py:"
sed -n '125,210p' main.py

echo "=== 3. Inspect vjp23/TD_Lambda/td_learning.py ==="
cd "$BASE/repo/vjp23-TD_Lambda"
cat td_learning.py

echo "=== 4. Run our verified sutton1988_experiments.py (E2, E3, E4) ==="
cd "$BASE/work"
$PY sutton1988_experiments.py
echo "FINISHED"
