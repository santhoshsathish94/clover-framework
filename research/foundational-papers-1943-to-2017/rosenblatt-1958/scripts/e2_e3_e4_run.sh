#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"

echo "=== 1. Repair Python 2 syntax in Honghe/perceptron work copy and run ==="
cd "$BASE/work/honghe_run"
$PY -c "
for fname in ('perceptron/perceptron.py', 'perceptron/generate_data.py', 'perceptron/test.py'):
    s = open(fname).read()
    s = s.replace('xrange', 'range')
    s = s.replace(\"print 'iterations: %s' % iteration\", \"print('iterations: %s' % iteration)\")
    s = s.replace(\"print 'not hit.'\", \"print('not hit.')\")
    s = s.replace('from matplotlib.pylab import *', 'import numpy as np\nnorm = np.linalg.norm\ndef plot(*a, **k): pass\ndef show(): pass')
    open(fname, 'w').write(s)
"
PYTHONPATH=perceptron $PY perceptron/test.py && echo "Honghe/perceptron Python 3 repair: OK (rc=0)"

echo "=== 2. Run E2, E3, and E4 (rosenblatt1958_models.py) ==="
cd "$BASE/work"
$PY rosenblatt1958_models.py
