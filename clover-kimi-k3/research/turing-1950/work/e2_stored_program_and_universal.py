"""
E2 (Turing 1950, Sections 4 & 5, pp. 436-442 / pp. 4-8 of extract):
  1. Stored-Program Digital Computer (Store / Executive unit / Control) using 10-decimal-digit
     packets AAAABBBBCC (Section 4, pp. 4-5):
       - "Add the number stored in position 6809 to that in 4302 and put the result back into the
          latter storage position... coded in a form such as 6809430217. Here 17 says which of
          various possible operations is to be performed on the two numbers."
       - "Now obey the instruction stored in position 5606, and continue from there"
       - "If position 4505 contains 0 obey next the instruction stored in 6707, otherwise continue
          straight on."
     Plus self-modifying / program-making instructions (Section 6.5, p. 14: "a machine undoubtedly
     can be its own subject matter. It may be used to help in making up its own programmes, or to
     predict the effect of alterations in its own structure").
  2. Discrete-State Machine (Section 5, pp. 6-7):
     3-state clicking wheel (q1, q2, q3), lever input (i0 = stop, i1 = click 120 deg),
     lamp output (o0, o0, o1), storage capacity log2(N_states), and Universality (Section 5, pp. 7-8):
     programming the Section 4 digital computer to mimic the wheel machine (and any finite-state table)
     so that in the Section 5 imitation game between the wheel (B) and the digital computer (A),
     their input-output traces are 100% indistinguishable.
"""
import itertools
import math
from typing import Dict, List, Tuple


class Turing1950DigitalComputer:
    """
    Section 4 Digital Computer:
      - Store: dictionary mapping positions (0..9999) to 10-digit non-negative integers (0..9999999999).
      - Control: instruction pointer `ip` (initially start_ip); fetches store[ip], decodes AAAABBBBCC:
          A = packet // 1000000
          B = (packet // 100) % 10000
          op = packet % 100
      - Executive unit operations (op):
          00: HALT
          17: store[B] = (store[A] + store[B]) % 10^10   (Turing's exact opcode 17 on p. 4!)
          18: store[B] = max(0, store[B] - store[A])     (Subtract store[A] from store[B])
          19: store[B] = (store[A] * store[B]) % 10^10   (Multiply, p. 4)
          20: store[B] = store[A]                        (Copy store[A] to store[B])
          21: store[B] = store[store[A]]                 (Indirect load: read from address in store[A])
          30: ip = B                                     (Unconditional jump: "Now obey instruction in B", p. 5)
          31: if store[A] == 0: ip = B                   (Conditional jump: "If position A contains 0 obey next B", p. 5)
    """
    MOD = 10_000_000_000

    def __init__(self, store: Dict[int, int] = None, start_ip: int = 1000):
        self.store: Dict[int, int] = dict(store or {})
        self.ip: int = start_ip
        self.steps: int = 0
        self.halted: bool = False

    @staticmethod
    def encode_packet(a: int, b: int, op: int) -> int:
        assert 0 <= a < 10000 and 0 <= b < 10000 and 0 <= op < 100
        return a * 1_000_000 + b * 100 + op

    @staticmethod
    def decode_packet(pkt: int) -> Tuple[int, int, int]:
        a = (pkt // 1_000_000) % 10000
        b = (pkt // 100) % 10000
        op = pkt % 100
        return a, b, op

    def step(self) -> bool:
        if self.halted:
            return False
        pkt = self.store.get(self.ip, 0)
        a, b, op = self.decode_packet(pkt)
        next_ip = self.ip + 1
        if op == 0:
            self.halted = True
            return False
        elif op == 17:
            self.store[b] = (self.store.get(a, 0) + self.store.get(b, 0)) % self.MOD
        elif op == 18:
            self.store[b] = max(0, self.store.get(b, 0) - self.store.get(a, 0))
        elif op == 19:
            self.store[b] = (self.store.get(a, 0) * self.store.get(b, 0)) % self.MOD
        elif op == 20:
            self.store[b] = self.store.get(a, 0)
        elif op == 21:
            addr = self.store.get(a, 0) % 10000
            self.store[b] = self.store.get(addr, 0)
        elif op == 30:
            next_ip = b
        elif op == 31:
            if self.store.get(a, 0) == 0:
                next_ip = b
        else:
            raise ValueError(f"Unknown opcode {op} at ip={self.ip} (packet={pkt:010d})")
        self.ip = next_ip
        self.steps += 1
        return True

    def run(self, max_steps: int = 100_000) -> int:
        while not self.halted and self.steps < max_steps:
            self.step()
        return self.steps


class ClickingWheelMachine:
    """
    Section 5 (pp. 6-7) 3-state discrete-state clicking wheel:
      States: q1 (0), q2 (1), q3 (2)
      Input signal: i0 (0: lever stops wheel) or i1 (1: wheel clicks round 120 deg)
      Transition table:
        i0: q1 -> q1, q2 -> q2, q3 -> q3
        i1: q1 -> q2, q2 -> q3, q3 -> q1
      Output table (lamp):
        q1 -> o0 (0), q2 -> o0 (0), q3 -> o1 (1)
    """
    def __init__(self, init_state: int = 0):
        self.state = init_state  # 0=q1, 1=q2, 2=q3

    def step(self, inp: int) -> Tuple[int, int]:
        if inp == 1:
            self.state = (self.state + 1) % 3
        out = 1 if self.state == 2 else 0
        return self.state, out


def mimic_discrete_machine_on_digital_computer(
    trans_table: Dict[Tuple[int, int], int],
    out_table: Dict[int, int],
    init_state: int,
    inputs: List[int],
    corrupt_table: bool = False,
) -> List[Tuple[int, int]]:
    """
    Programs `Turing1950DigitalComputer` to mimic any finite discrete-state machine
    by storing its transition table and output table in the Store and executing a
    stored instruction table for each input symbol (Section 5 Universality, pp. 7-8).
      - Store layout:
          100..199: transition table at address 100 + 10*state + inp
          200..209: output table at address 200 + state
          300: current state q
          301: current input i
          302: current output o
          303: constant 10
          304: constant 100
          305: constant 200
          306: scratch address pointer
    """
    store = {
        300: init_state,
        303: 10,
        304: 100,
        305: 200,
    }
    for (s, i), next_s in trans_table.items():
        store[100 + 10 * s + i] = (next_s + 1) % 3 if (corrupt_table and s == 0 and i == 1) else next_s
    for s, o in out_table.items():
        store[200 + s] = o

    # Stored program at 1000..1009 for one step of the discrete-state machine:
    #   1000: store[306] = store[300]                  (20: copy state q to scratch)
    #   1001: store[306] = store[303] * store[306]     (19: 10 * q)
    #   1002: store[306] = store[301] + store[306]     (17: 10 * q + i)
    #   1003: store[306] = store[304] + store[306]     (17: 100 + 10 * q + i)
    #   1004: store[300] = store[store[306]]           (21: q_next = store[100 + 10*q + i])
    #   1005: store[306] = store[300]                  (20: copy q_next to scratch)
    #   1006: store[306] = store[305] + store[306]     (17: 200 + q_next)
    #   1007: store[302] = store[store[306]]           (21: o = store[200 + q_next])
    #   1008: HALT                                     (00)
    prog = [
        Turing1950DigitalComputer.encode_packet(300, 306, 20),
        Turing1950DigitalComputer.encode_packet(303, 306, 19),
        Turing1950DigitalComputer.encode_packet(301, 306, 17),
        Turing1950DigitalComputer.encode_packet(304, 306, 17),
        Turing1950DigitalComputer.encode_packet(306, 300, 21),
        Turing1950DigitalComputer.encode_packet(300, 306, 20),
        Turing1950DigitalComputer.encode_packet(305, 306, 17),
        Turing1950DigitalComputer.encode_packet(306, 302, 21),
        Turing1950DigitalComputer.encode_packet(0, 0, 0),
    ]
    for idx, pkt in enumerate(prog):
        store[1000 + idx] = pkt

    comp = Turing1950DigitalComputer(store=store, start_ip=1000)
    trace = []
    for inp in inputs:
        comp.store[301] = inp
        comp.ip = 1000
        comp.halted = False
        comp.run()
        trace.append((comp.store[300], comp.store[302]))
    return trace


def main():
    print("=== E2.1: Section 4 Stored-Program Computer & 10-Digit Packet 6809430217 ===")
    # Test Turing's exact packet 6809430217 on pp. 4-5 with Section 2 numbers 34957 and 70764:
    pkt = 6809430217
    a, b, op = Turing1950DigitalComputer.decode_packet(pkt)
    print(f"Decoded packet {pkt}: pos_A={a}, pos_B={b}, opcode={op}")
    assert (a, b, op) == (6809, 4302, 17)

    comp = Turing1950DigitalComputer(
        store={6809: 34957, 4302: 70764, 1000: 6809430217, 1001: 0},
        start_ip=1000,
    )
    comp.run()
    print(f"After executing packet 6809430217: store[4302] = {comp.store[4302]} (true sum 34957+70764 = {34957+70764}, paper witness said 105621)")
    assert comp.store[4302] == 105721

    # Self-modifying program test (Section 4 & Section 6.5 p. 14: machine modifying its own instruction table):
    # Sum array stored at positions 5000..5004 ([10, 20, 30, 40, 50] = 150) by adding 1_000_000 (address +1 in AAAA)
    # to the instruction packet at position 1000 (`5000430217` -> `5001430217` -> ... -> `5004430217`)!
    sm_store = {
        5000: 10, 5001: 20, 5002: 30, 5003: 40, 5004: 50,
        4302: 0,                  # accumulator
        4000: 1_000_000,          # increment for AAAA field of instruction packet!
        4001: 5,                  # loop counter = 5 elements
        4002: 1,                  # constant 1
        1000: 5000430217,         # SELF-MODIFIED instruction: Add store[5000+k] to store[4302]
        1001: Turing1950DigitalComputer.encode_packet(4000, 1000, 17),  # Add 1_000_000 to instruction at store[1000]!
        1002: Turing1950DigitalComputer.encode_packet(4002, 4001, 18),  # Decrement counter store[4001] -= 1
        1003: Turing1950DigitalComputer.encode_packet(4001, 1005, 31),  # If counter == 0 jump to 1005 (HALT)
        1004: Turing1950DigitalComputer.encode_packet(0, 1000, 30),     # Else jump back to 1000
        1005: 0,                  # HALT
    }
    sm_comp = Turing1950DigitalComputer(store=sm_store, start_ip=1000)
    sm_comp.run()
    print(f"Self-modifying loop result: store[4302]={sm_comp.store[4302]} (expected 150), mutated instruction at store[1000]={sm_comp.store[1000]} (started 5000430217)")
    assert sm_comp.store[4302] == 150 and sm_comp.store[1000] == 5005430217

    print("\n=== E2.2: Section 5 Discrete-State Wheel Machine & Universal Mimicry ===")
    trans_table = {
        (0, 0): 0, (1, 0): 1, (2, 0): 2,
        (0, 1): 1, (1, 1): 2, (2, 1): 0,
    }
    out_table = {0: 0, 1: 0, 2: 1}
    matched_seqs = 0
    corrupted_seqs = 0
    for bits in itertools.product([0, 1], repeat=8):
        wheel = ClickingWheelMachine(init_state=0)
        w_trace = [wheel.step(b) for b in bits]
        u_trace = mimic_discrete_machine_on_digital_computer(trans_table, out_table, 0, list(bits), corrupt_table=False)
        c_trace = mimic_discrete_machine_on_digital_computer(trans_table, out_table, 0, list(bits), corrupt_table=True)
        if w_trace == u_trace:
            matched_seqs += 1
        if w_trace == c_trace:
            corrupted_seqs += 1
    print(f"Universal digital computer vs 3-state wheel across all 256 8-step input streams: {matched_seqs}/256 identical")
    print(f"Negative control (corrupted transition entry (q1,i1)): {corrupted_seqs}/256 identical")
    assert matched_seqs == 256 and corrupted_seqs == 1  # only all-zero input 00000000 avoids (q1, i1)

    print("\n=== E2.3: Section 5 Storage Capacity Calculations (pp. 6-7) ===")
    cap_wheel = math.log2(3)
    cap_paper_sheets = 150_000 * math.log2(10)
    manchester_tracks = 64 * 2560
    manchester_tubes = 8 * 1280
    manchester_misc = 300
    manchester_total = manchester_tracks + manchester_tubes + manchester_misc
    print(f"3-state wheel capacity: log2(3) = {cap_wheel:.5f} bits (Turing p. 7: 'about 1.6')")
    print(f"100 sheets x 50 lines x 30 decimal digits (10^150,000 states): {cap_paper_sheets:.1f} bits = {cap_paper_sheets/165000:.2f}x Manchester's 165,000 (Turing p. 7: 'about three Manchester machines')")
    print(f"Manchester breakdown (p. 7): 64*2560 ({manchester_tracks}) + 8*1280 ({manchester_tubes}) + {manchester_misc} = {manchester_total} bits (Turing p. 7: 'making a total of 174,380')")
    assert manchester_total == 174380


if __name__ == "__main__":
    main()
