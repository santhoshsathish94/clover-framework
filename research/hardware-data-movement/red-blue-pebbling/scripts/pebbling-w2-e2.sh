#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
mkdir -p "$W2"
rm -rf "$W2/e2_work"
cp -a /root/dm-study/pebbling/repo/pebbling "$W2/e2_work"
cp "$W2/paper_engine.py" "$W2/e2_work/paper_engine.py"
cd "$W2/e2_work"

/root/ncm-cpu/venv/bin/python - << 'EOF'
from pebbling_engine import DAG, Node, Edge, PebblingGame, Action, ActionType
from paper_engine import PaperPebbleEngine, PaperMove

# Build a fixed 4-node diamond DAG:
#   0 (input) -> 1, 2;  1, 2 -> 3 (output)
def make_diamond_repo(reg_slots=2, sram_slots=4):
    nodes = [
        Node(0, 0, [], [1, 2], is_source=True, is_sink=False),
        Node(1, 1, [0], [3], is_source=False, is_sink=False),
        Node(2, 1, [0], [3], is_source=False, is_sink=False),
        Node(3, 2, [1, 2], [], is_source=False, is_sink=True),
    ]
    edges = [Edge(0, 1), Edge(0, 2), Edge(1, 3), Edge(2, 3)]
    dag = DAG(nodes=nodes, edges=edges, levels=[[0], [1, 2], [3]])
    return PebblingGame(dag=dag, register_slots=reg_slots, sram_slots=sram_slots, hardware_profile="h100")

def make_diamond_paper(S=2):
    preds = {0: [], 1: [0], 2: [0], 3: [1, 2]}
    return PaperPebbleEngine(4, preds, S=S)

print("=== PROBE 1: INITIAL STATE AND INPUT RULE (R1) ===")
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
print(f"Repo initial pebbles: {rg.pebbles}")
print(f"Paper initial pebbles: red={pg.red}, blue={pg.blue}")
r_step = rg.step(Action(ActionType.PLACE, 0, "register"))
p_ok, p_cost, p_err = pg.step(PaperMove("R1", 0))
print(f"Load input 0 -> Repo PLACE(0, register): accepted={r_step.info['error'] is None}, cost={r_step.info['cost']} pJ | Paper R1(0): accepted={p_ok}, cost={p_cost} I/O")

print("\n=== PROBE 2: OUTPUT RULE (R2) — SIMULTANEOUS RED+BLUE VS MOVE ===")
# Now node 0 is in register (repo) / red (paper). Store node 0 to slow memory (sram / blue):
r_step2 = rg.step(Action(ActionType.TRANSFER, 0, "sram"))
# Note: in paper, node 0 already has blue (it's an input); let's compute node 1 first, then R2(1)!
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
rg.step(Action(ActionType.PLACE, 0, "register"))
rg.step(Action(ActionType.PLACE, 1, "register"))
pg.step(PaperMove("R1", 0))
pg.step(PaperMove("R3", 1))
print(f"Before R2(1): Repo reg usage={rg._tier_usage()['register']}/2, pebbles={rg.pebbles} | Paper red={pg.red}, blue={pg.blue}")
r_out = rg.step(Action(ActionType.TRANSFER, 1, "sram"))
p_ok, p_cost, p_err = pg.step(PaperMove("R2", 1))
print(f"After R2(1)/TRANSFER(1,sram): Repo accepted={r_out.info['error'] is None}, cost={r_out.info['cost']} pJ, reg_usage={rg._tier_usage()['register']}/2, pebbles={rg.pebbles}")
print(f"After R2(1)/TRANSFER(1,sram): Paper accepted={p_ok}, cost={p_cost} I/O, red={pg.red} (len={len(pg.red)}/2), blue={pg.blue}")
# Now reload 1 into register in repo vs R1 in paper (after deleting 1 from red in paper):
r_in = rg.step(Action(ActionType.TRANSFER, 1, "register"))
pg.step(PaperMove("R4", 1, "red"))
p_in_ok, p_in_cost, _ = pg.step(PaperMove("R1", 1))
print(f"Reload 1 to fast mem: Repo TRANSFER(1, register) cost={r_in.info['cost']} pJ | Paper R1(1) cost={p_in_cost} I/O")

print("\n=== PROBE 3: COMPUTE (R3) WITHOUT PREDECESSOR PEBBLED AT ALL (ILLEGAL IN BOTH) ===")
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
r_bad = rg.step(Action(ActionType.PLACE, 1, "register"))
p_ok, p_cost, p_err = pg.step(PaperMove("R3", 1))
print(f"Compute N1 with N0 unpebbled/blue-only: Repo accepted={r_bad.info['error'] is None} (err={r_bad.info['error']}) | Paper accepted={p_ok} (err={p_err})")

print("\n=== PROBE 4: COMPUTE (R3) WHEN PREDECESSOR IS IN SLOW MEMORY (BLUE / SRAM / HBM) ===")
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
# Place N0 in SRAM in repo; in paper N0 starts in blue (slow memory)
rg.step(Action(ActionType.PLACE, 0, "sram"))
r_slow_pred = rg.step(Action(ActionType.PLACE, 1, "register"))
p_ok, p_cost, p_err = pg.step(PaperMove("R3", 1))
print(f"Compute N1 when N0 is in slow mem (SRAM/blue): Repo accepted={r_slow_pred.info['error'] is None}, cost={r_slow_pred.info['cost']} pJ | Paper accepted={p_ok} (err={p_err})")

print("\n=== PROBE 5: DIRECT COMPUTE INTO SLOW MEMORY WITH S=0 FAST SLOTS ===")
rg0 = make_diamond_repo(reg_slots=0, sram_slots=4)
pg0 = make_diamond_paper(S=0)
r_s0 = rg0.step(Action(ActionType.PLACE, 0, "sram"))
r_s1 = rg0.step(Action(ActionType.PLACE, 1, "sram"))
p0_ok, _, p0_err = pg0.step(PaperMove("R1", 0))
print(f"With register_slots=0 / S=0: Repo PLACE(0,sram) accepted={r_s0.info['error'] is None}, PLACE(1,sram) accepted={r_s1.info['error'] is None} | Paper R1(0) accepted={p0_ok} (err={p0_err})")

print("\n=== PROBE 6: EXCEEDING RED / REGISTER CAPACITY S=2 ===")
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
rg.step(Action(ActionType.PLACE, 0, "register"))
rg.step(Action(ActionType.PLACE, 1, "register"))
r_cap = rg.step(Action(ActionType.PLACE, 2, "register"))
pg.step(PaperMove("R1", 0))
pg.step(PaperMove("R3", 1))
p_cap_ok, _, p_cap_err = pg.step(PaperMove("R3", 2))
print(f"3rd red/register pebble with S=2: Repo accepted={r_cap.info['error'] is None} (err={r_cap.info['error']}) | Paper accepted={p_cap_ok} (err={p_cap_err})")

print("\n=== PROBE 7: DELETE (R4) AND RECOMPUTATION ===")
# Delete N0 after computing N1, then re-load/recompute N0 and N1
rg = make_diamond_repo(reg_slots=2)
pg = make_diamond_paper(S=2)
rg.step(Action(ActionType.PLACE, 0, "register"))
rg.step(Action(ActionType.PLACE, 1, "register"))
rg.step(Action(ActionType.REMOVE, 1))
r_re = rg.step(Action(ActionType.PLACE, 1, "register"))
pg.step(PaperMove("R1", 0))
pg.step(PaperMove("R3", 1))
pg.step(PaperMove("R4", 1, "red"))
p_re_ok, p_re_cost, p_re_err = pg.step(PaperMove("R3", 1))
print(f"Delete N1 then recompute N1: Repo accepted={r_re.info['error'] is None}, cost={r_re.info['cost']} pJ | Paper accepted={p_re_ok}, cost={p_re_cost} I/O")

# What if we delete the input's blue pebble in paper vs REMOVE(0) in repo?
rg.step(Action(ActionType.REMOVE, 0))
r_in_re = rg.step(Action(ActionType.PLACE, 0, "register"))
pg.step(PaperMove("R4", 0, "red"))
pg.step(PaperMove("R4", 0, "blue"))
p_in_re_ok, _, p_in_re_err = pg.step(PaperMove("R1", 0))
print(f"Delete input N0 completely then re-input N0: Repo accepted={r_in_re.info['error'] is None} | Paper accepted={p_in_re_ok} (err={p_in_re_err})")

print("\n=== PROBE 8: ENDING / TERMINAL CONDITION ON COMPLETE DIAMOND SCHEDULE (S=3) ===")
rg3 = make_diamond_repo(reg_slots=3)
pg3 = make_diamond_paper(S=3)
for nid in [0, 1, 2]:
    rg3.step(Action(ActionType.PLACE, nid, "register"))
rg3.step(Action(ActionType.REMOVE, 0))
rg3.step(Action(ActionType.PLACE, 3, "register"))
print(f"Repo after placing sink N3 in register (with N1, N2 still in register): done={rg3.done}, total_cost={rg3.total_cost} pJ, pebbles={rg3.pebbles}")

pg3.step(PaperMove("R1", 0))
pg3.step(PaperMove("R3", 1))
pg3.step(PaperMove("R3", 2))
pg3.step(PaperMove("R4", 0, "red"))
pg3.step(PaperMove("R3", 3))
print(f"Paper at same point (N3 in red, N0 in blue, N1,N2 in red): is_terminal={pg3.is_terminal()}, outputs_all_blue={pg3.outputs_all_blue()}, io_cost={pg3.io_cost}")
pg3.step(PaperMove("R2", 3))
print(f"Paper after R2(3) (N3 in blue, but red={pg3.red}, blue={pg3.blue}): is_terminal={pg3.is_terminal()}, outputs_all_blue={pg3.outputs_all_blue()}, io_cost={pg3.io_cost}")
pg3.step(PaperMove("R4", 1, "red"))
pg3.step(PaperMove("R4", 2, "red"))
pg3.step(PaperMove("R4", 3, "red"))
pg3.step(PaperMove("R4", 0, "blue"))
print(f"Paper after cleaning up R4 on N1,N2,N3(red) and N0(blue): is_terminal={pg3.is_terminal()}, red={pg3.red}, blue={pg3.blue}, io_cost={pg3.io_cost}")
EOF
