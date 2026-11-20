#!/usr/bin/env python3
"""Focused verification suite for Clover Studio's x-cruncher exact math & Pi tool."""
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parent / "clover"
if str(CLOVER) not in sys.path:
    sys.path.insert(0, str(CLOVER))

import lfm3_native
import local_tool_packs
import permissions
import semantics
import xcruncher_bridge


def check(cond, msg):
    if not cond:
        print("FAIL", msg)
        raise AssertionError(msg)
    print("PASS", msg)


class _DummyEng:
    def __init__(self, base):
        self.base_dir = Path(base)
        self.workdir = Path(base)
        self.perms = permissions.PermissionStore(self.base_dir / "config" / "permissions.json")
        self.turn_cancel = None
        self.turn_images = []
        self.user_paths = ()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="clover_xc_tool_"))
    try:
        eng = _DummyEng(tmp)
        info = xcruncher_bridge.active_isa_info()
        check(info["backend"] in ("xcruncher-c11-native", "python-bigint-fallback"), f"active_isa_info: {info}")

        # 1. Exact Sparse NAF & Big-Integer Multiplication
        r_mul = json.loads(local_tool_packs.run_pack_tool(
            "xcruncher_compute",
            {"operation": "mul", "a": "(1 << 200000) - 6", "b": "(1 << 199999) - 3"},
            eng,
        ))
        exp_mul = ((1 << 200000) - 6) * ((1 << 199999) - 3)
        old_lim = getattr(sys, "get_int_max_str_digits", lambda: 4300)()
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(200000)
        exp_dec = str(exp_mul)
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(old_lim)
        check(r_mul["decimal_digits"] == len(exp_dec)
              and r_mul["decimal_sha256"] == hashlib.sha256(exp_dec.encode("ascii")).hexdigest(),
              "200,000-bit exact product SHA-256 matches Python ground truth")

        # 2. Exact Factorial (5000! = 16,326 digits)
        r_fact = json.loads(local_tool_packs.run_pack_tool(
            "xcruncher_compute", {"operation": "factorial", "n": 5000}, eng
        ))
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(50000)
        f5000 = str(math.factorial(5000))
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(old_lim)
        check(r_fact["decimal_digits"] == len(f5000)
              and r_fact["decimal_sha256"] == hashlib.sha256(f5000.encode("ascii")).hexdigest(),
              "5000! exact 16,326-digit factorial matches math.factorial")

        # 3. Zero-Storage N-th Hexadecimal Pi Digits + 2^44 Range Guard
        r_pi0 = json.loads(local_tool_packs.run_pack_tool(
            "xcruncher_compute", {"operation": "pi_hex", "hex_offset": 0, "digits": 20}, eng
        ))
        r_pi1m = json.loads(local_tool_packs.run_pack_tool(
            "xcruncher_compute", {"operation": "pi_hex", "hex_offset": 1_000_000, "digits": 20}, eng
        ))
        r_oob = local_tool_packs.run_pack_tool(
            "xcruncher_compute", {"operation": "pi_hex", "hex_offset": (1 << 44) + 1, "digits": 20}, eng
        )
        check(r_pi0["pi_hex"] == "243f6a8885a308d31319" and r_pi1m["pi_hex"] == "6c65e52cb459350050e4",
              f"Zero-storage Pi hex offset 0 & 1M exact: {r_pi1m['pi_hex']}")
        check(r_oob.startswith("Error:"), f"Out-of-range Pi offset > 2^44 rejected cleanly: {r_oob}")

        # 4. Also verify unified `calculate` tool delegates pi_hex, factorial(5000), and near-2^k shifts to x-cruncher
        c_pi = local_tool_packs.run_pack_tool("calculate", {"expression": "pi_hex(1000000, 20)"}, eng)
        check(c_pi == "6c65e52cb459350050e4", f"calculate('pi_hex(1000000, 20)') exact: {c_pi}")
        c_sh = json.loads(local_tool_packs.run_pack_tool(
            "calculate", {"expression": "((1 << 200000) - 6) * ((1 << 199999) - 3)"}, eng
        ))
        check(c_sh["decimal_sha256"] == r_mul["decimal_sha256"], "calculate near-2^k shift matches xcruncher")

        # 5. Semantic Routing (routes to `calculate` when asked, stays quiet on general chat)
        rt = lfm3_native.RUNTIME
        all_tools = [t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS]
        q_pi = "What are the 20 hexadecimal digits of pi at offset 1000000?"
        only_pi, on_pi = semantics.route(semantics.interpret(q_pi, rt), "ask", True, said=q_pi, offer=all_tools)
        check(on_pi and "calculate" in (only_pi or []), f"Pi hex query routes to calculate: {only_pi}")

        q_fact = "Compute 5000! factorial of 5000"
        only_f, on_f = semantics.route(semantics.interpret(q_fact, rt), "ask", True, said=q_fact, offer=all_tools)
        check(on_f and "calculate" in (only_f or []), f"Huge factorial routes to calculate: {only_f}")

        only_always, _ = semantics.route(semantics.interpret("Explain how photosynthesis works", rt),
                                         "always", True, said="Explain how photosynthesis works", offer=all_tools)
        check("xcruncher_compute" not in (only_always or []), "general chat keeps quiet exact-math routing clean")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
