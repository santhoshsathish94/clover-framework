"""
Minimal engine implementing the exact Red-Blue Pebble Game rules from
J.-W. Hong and H. T. Kung, "I/O complexity: The red-blue pebble game",
STOC 1981, Section 2, p. 327.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PaperMove:
    rule: str          # "R1" (input), "R2" (output), "R3" (compute), "R4" (delete)
    node_id: int
    color: str | None = None  # For R4: "red" or "blue"


class PaperPebbleEngine:
    """
    Hong & Kung (1981) Section 2 Red-Blue Pebble Game engine.
    - Configuration: (red: set[int], blue: set[int])
    - Initial configuration: red = empty, blue = inputs
    - Terminal configuration: red = empty, blue = outputs
    - R1 (Input): place red on v if v in blue and |red| < S (cost +1)
    - R2 (Output): place blue on v if v in red (cost +1)
    - R3 (Compute): place red on non-input v if pred(v) <= red and |red| < S (cost 0)
    - R4 (Delete): remove red or blue pebble from v (cost 0)
    """

    def __init__(self, num_nodes: int, preds: dict[int, list[int]], S: int,
                 inputs: set[int] | None = None, outputs: set[int] | None = None):
        self.num_nodes = num_nodes
        self.preds = {i: list(preds.get(i, [])) for i in range(num_nodes)}
        self.succs = {i: [] for i in range(num_nodes)}
        for u in range(num_nodes):
            for p in self.preds[u]:
                self.succs[p].append(u)
        self.inputs = set(inputs) if inputs is not None else {i for i in range(num_nodes) if not self.preds[i]}
        self.outputs = set(outputs) if outputs is not None else {i for i in range(num_nodes) if not self.succs[i]}
        self.S = S
        self.reset()

    def reset(self):
        self.red: set[int] = set()
        self.blue: set[int] = set(self.inputs)
        self.io_cost: int = 0
        self.step_count: int = 0

    def is_terminal(self) -> bool:
        """Exact paper terminal configuration: only outputs have pebbles, all blue."""
        return len(self.red) == 0 and self.blue == self.outputs

    def outputs_all_blue(self) -> bool:
        """Check whether all output vertices currently have blue pebbles."""
        return self.outputs.issubset(self.blue)

    def step(self, move: PaperMove) -> tuple[bool, int, str | None]:
        """
        Execute one transition. Returns (accepted, cost_delta, error_msg).
        Does not mutate state if rejected.
        """
        v = move.node_id
        if v < 0 or v >= self.num_nodes:
            return False, 0, f"Invalid vertex {v}"

        if move.rule == "R1":
            if v not in self.blue:
                return False, 0, f"R1 rejected: vertex {v} does not have a blue pebble"
            if v in self.red:
                return False, 0, f"R1 rejected: vertex {v} already has a red pebble"
            if len(self.red) >= self.S:
                return False, 0, f"R1 rejected: red pebble limit S={self.S} reached"
            self.red.add(v)
            self.io_cost += 1
            self.step_count += 1
            return True, 1, None

        elif move.rule == "R2":
            if v not in self.red:
                return False, 0, f"R2 rejected: vertex {v} does not have a red pebble"
            if v in self.blue:
                return False, 0, f"R2 rejected: vertex {v} already has a blue pebble"
            self.blue.add(v)
            self.io_cost += 1
            self.step_count += 1
            return True, 1, None

        elif move.rule == "R3":
            if v in self.inputs:
                return False, 0, f"R3 rejected: vertex {v} is an input (must use R1)"
            if v in self.red:
                return False, 0, f"R3 rejected: vertex {v} already has a red pebble"
            missing = [p for p in self.preds[v] if p not in self.red]
            if missing:
                return False, 0, f"R3 rejected: predecessors {missing} of {v} lack red pebbles"
            if len(self.red) >= self.S:
                return False, 0, f"R3 rejected: red pebble limit S={self.S} reached"
            self.red.add(v)
            self.step_count += 1
            return True, 0, None

        elif move.rule == "R4":
            if move.color == "red":
                if v not in self.red:
                    return False, 0, f"R4(red) rejected: vertex {v} has no red pebble"
                self.red.remove(v)
            elif move.color == "blue":
                if v not in self.blue:
                    return False, 0, f"R4(blue) rejected: vertex {v} has no blue pebble"
                self.blue.remove(v)
            else:
                return False, 0, f"R4 rejected: invalid color {move.color}"
            self.step_count += 1
            return True, 0, None

        return False, 0, f"Unknown rule {move.rule}"
