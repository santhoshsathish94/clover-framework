#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
mkdir -p "$W2"
rm -rf "$W2/e1_repo"
cp -a /root/dm-study/pebbling/repo/pebbling "$W2/e1_repo"

echo "=== 1. REPO METADATA (COMMIT, URL, LICENSE, FILES, LINES) ==="
cd "$W2/e1_repo"
git rev-parse HEAD
git log -n 5 --oneline
git remote -v
head -n 5 LICENSE
wc -l README.md pyproject.toml pebbling_engine.py pebbling_agent.py pebbling-game.jsx
du -sh .

echo ""
echo "=== 2. PAPER TEXT VERIFICATION ==="
PAPER="/root/dm-study/pebbling/work/paper/1981-stoc-hong-kung.txt"
if [ -f "$PAPER" ]; then
    wc -l -c "$PAPER"
    grep -n "=== PAGE" "$PAPER"
else
    echo "Paper not at $PAPER, searching..."
    find /root/dm-study/pebbling -name "*hong*" -o -name "*1981*"
fi

echo ""
echo "=== 3. RUN BUILT-IN ENGINE DEMO (pebbling_engine.py) ==="
/root/ncm-cpu/venv/bin/python pebbling_engine.py

echo ""
echo "=== 4. RUN BUILT-IN CLI AGENTS (heuristic, random, tournament) ==="
/root/ncm-cpu/venv/bin/python pebbling_agent.py --agent heuristic --nodes 10 --depth 4 --fan-in 2 --registers 3 --sram 4 --seed 42
/root/ncm-cpu/venv/bin/python pebbling_agent.py --agent random --nodes 10 --depth 4 --fan-in 2 --registers 3 --sram 4 --seed 42
/root/ncm-cpu/venv/bin/python pebbling_agent.py --agent tournament --nodes 12 --depth 5 --games 5 --seed 42

echo ""
echo "=== 5. SWEEP PROFILES AND DIFFICULTY PRESETS ==="
/root/ncm-cpu/venv/bin/python - << 'EOF'
from pebbling_engine import PebblingGame, benchmark_strategies, HARDWARE_PROFILES, MEMORY_TIERS
from pebbling_agent import HeuristicAgent, RandomAgent, run_agent_game

print("Hardware profiles:", HARDWARE_PROFILES)

presets = [
    ("tutorial_preset", 6, 3, 2, 3, 4),
    ("easy_preset",     8, 4, 2, 3, 5),
    ("medium_preset",  12, 5, 3, 3, 4),
    ("hard_preset",    16, 6, 3, 2, 3),
    ("expert_preset",  20, 7, 4, 2, 3),
]

print("\nPreset sweep (seed=42, profile=h100):")
print(f"{'Preset':16s} {'Nodes':>5s} {'Reg':>3s} {'SRAM':>4s} | {'GreedyCost':>10s} {'GreedyDone':>10s} | {'TopoCost':>8s} {'TopoDone':>8s} | {'HeurCost':>8s} {'HeurDone':>8s} {'HeurSteps':>9s}")
for name, n, d, fi, r, s in presets:
    b = benchmark_strategies(num_nodes=n, depth=d, max_fan_in=fi, register_slots=r, sram_slots=s, seed=42, hardware_profile="h100")
    g_res = b["greedy_cheapest"]
    t_res = b["topological_register"]
    h_res = run_agent_game(HeuristicAgent(), num_nodes=n, depth=d, max_fan_in=fi, register_slots=r, sram_slots=s, seed=42, verbose=False)
    # Check if sinks are actually pebbled at end of HeuristicAgent run
    game_check = PebblingGame.create(num_nodes=n, depth=d, max_fan_in=fi, register_slots=r, sram_slots=s, seed=42)
    ag = HeuristicAgent()
    ag.reset()
    while not game_check.done:
        game_check.step(ag.get_action(game_check))
    sinks_done = all(game_check.pebbles[sid] is not None for sid in game_check.dag.sink_ids)
    print(f"{name:16s} {n:5d} {r:3d} {s:4d} | {g_res['total_cost']:10d} {str(g_res['done']):>10s} | {t_res['total_cost']:8d} {str(t_res['done']):>8s} | {h_res['total_cost']:8d} {str(sinks_done):>8s} {h_res['steps']:9d}")

print("\nProfile sweep on medium_preset (nodes=12, depth=5, fan_in=3, reg=3, sram=4, seed=42):")
for prof in ["h100", "tutorial", "extreme"]:
    b = benchmark_strategies(num_nodes=12, depth=5, max_fan_in=3, register_slots=3, sram_slots=4, seed=42, hardware_profile=prof)
    h_res = run_agent_game(HeuristicAgent(), num_nodes=12, depth=5, max_fan_in=3, register_slots=3, sram_slots=4, hardware_profile=prof, seed=42, verbose=False)
    print(f"  profile={prof:10s}: greedy={b['greedy_cheapest']['total_cost']} pJ (steps={b['greedy_cheapest']['steps']}), topo={b['topological_register']['total_cost']} pJ (steps={b['topological_register']['steps']}), heuristic={h_res['total_cost']} pJ (steps={h_res['steps']})")
EOF
