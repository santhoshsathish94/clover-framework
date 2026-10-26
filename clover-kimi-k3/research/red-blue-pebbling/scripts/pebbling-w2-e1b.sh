#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2/e1_repo"
cd "$W2"

/root/ncm-cpu/venv/bin/python - << 'EOF'
from pebbling_engine import PebblingGame, Action, ActionType, greedy_cheapest_strategy, topological_register_strategy

# 1. Inspect topological_register_strategy on 10-node demo DAG (seed=42)
game = PebblingGame.create(num_nodes=10, depth=4, max_fan_in=2, register_slots=3, sram_slots=4, hardware_profile="h100", seed=42)
t_trans = topological_register_strategy(game)
print("=== topological_register_strategy on 10-node demo (seed=42) ===")
for i, step in enumerate(t_trans):
    nid = step["action"]["node_id"]
    is_sink = game.dag.nodes[nid].is_sink
    print(f"  step {i+1:2d}: {step['action']} cost={step['cost']} total={step['total_cost']} is_sink={is_sink}")
print("Final pebbles:", game.pebbles)
print("Sinks remaining:", [s for s in game.dag.sink_ids if game.pebbles[s] is None], "done:", game.done)

# 2. Inspect greedy_cheapest_strategy on medium_preset (12 nodes, depth=5, fan_in=3, reg=3, sram=4, seed=42, h100)
game_m = PebblingGame.create(num_nodes=12, depth=5, max_fan_in=3, register_slots=3, sram_slots=4, hardware_profile="h100", seed=42)
g_trans = greedy_cheapest_strategy(game_m)
print("\n=== greedy_cheapest_strategy on medium_preset (seed=42, h100) ===")
print(f"Total steps={game_m.step_count}, total_cost={game_m.total_cost}, done={game_m.done}, sinks_remaining={[s for s in game_m.dag.sink_ids if game_m.pebbles[s] is None]}")
print("First 12 steps:")
for i, s in enumerate(g_trans[:12]):
    print(f"  step {i+1:3d}: {s['action']} cost={s['cost']} total={s['total_cost']}")
print("Last 6 steps:")
for i, s in enumerate(g_trans[-6:], start=len(g_trans)-5):
    print(f"  step {i:3d}: {s['action']} cost={s['cost']} total={s['total_cost']}")

# 3. Test patched topological_register_strategy (do not evict sinks in free_register_if_needed)
def topological_register_fixed(game: PebblingGame):
    game.reset()
    topo_order = [nid for level in game.dag.levels for nid in level]
    def do_action(action):
        game.step(action)
    def free_register_if_needed():
        if game._tier_usage()["register"] < game.register_slots:
            return
        reg_nodes = [nid for nid, t in game.pebbles.items() if t == "register" and not game.dag.nodes[nid].is_sink]
        if not reg_nodes:
            return
        reg_nodes.sort(key=lambda nid: sum(1 for sid in game.dag.nodes[nid].successors if game.pebbles[sid] is None))
        victim = reg_nodes[0]
        unpebbled = sum(1 for sid in game.dag.nodes[victim].successors if game.pebbles[sid] is None)
        if unpebbled == 0:
            do_action(Action(ActionType.REMOVE, victim))
        elif game._has_capacity("sram"):
            do_action(Action(ActionType.TRANSFER, victim, "sram"))
        else:
            do_action(Action(ActionType.TRANSFER, victim, "hbm"))

    for nid in topo_order:
        if game.done or game.pebbles[nid] is not None or not game._can_compute(nid):
            continue
        free_register_if_needed()
        if game._has_capacity("register"):
            do_action(Action(ActionType.PLACE, nid, "register"))
        elif game._has_capacity("sram"):
            do_action(Action(ActionType.PLACE, nid, "sram"))
        else:
            do_action(Action(ActionType.PLACE, nid, "hbm"))
        for prev_nid in list(game.pebbles.keys()):
            if game.pebbles[prev_nid] is None or game.dag.nodes[prev_nid].is_sink:
                continue
            if all(game.pebbles[sid] is not None for sid in game.dag.nodes[prev_nid].successors):
                do_action(Action(ActionType.REMOVE, prev_nid))

print("\n=== Patched topological_register_fixed across presets (seed=42, h100) ===")
presets = [
    ("demo_10",          10, 4, 2, 3, 4),
    ("tutorial_preset",   6, 3, 2, 3, 4),
    ("easy_preset",       8, 4, 2, 3, 5),
    ("medium_preset",    12, 5, 3, 3, 4),
    ("hard_preset",      16, 6, 3, 2, 3),
    ("expert_preset",    20, 7, 4, 2, 3),
]
for name, n, d, fi, r, s in presets:
    g = PebblingGame.create(num_nodes=n, depth=d, max_fan_in=fi, register_slots=r, sram_slots=s, seed=42)
    topological_register_fixed(g)
    sinks_ok = all(g.pebbles[sid] is not None for sid in g.dag.sink_ids)
    print(f"  {name:16s}: cost={g.total_cost:4d} pJ, steps={g.step_count:2d}, done={g.done}, all_sinks_pebbled={sinks_ok}")
EOF
