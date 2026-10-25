"""Live verification of Clover Kimi K3's Generative Compilation (`k3_generative_compiler.py`)
for both Python (`CloverPythonGenerativeCompiler`) and Rust (`CloverRustGenerativeCompiler`)
via the live `clover-k3-chat.service` (`http://127.0.0.1:8765/api/chat`)."""
import json, urllib.request
from k3_generative_compiler import CloverPythonGenerativeCompiler, CloverRustGenerativeCompiler


def ask_clover(prompt: str, sid: str) -> dict:
    body = json.dumps({"session_id": sid, "prompt": prompt, "cands": 2, "remember": False}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # --- TEST 1: Live Python Generative Compilation Rollback on HumanEval/17 (`parse_music`) ---
    prompt_py = (
        "from typing import List\n\n"
        "def parse_music(music_string: str) -> List[int]:\n"
        '    """ Input to this function is a string representing musical notes in a special ASCII format.\n'
        "    Your task is to parse this string and return list of integers corresponding to how many beats does each\n"
        "    not last.\n"
        "    Here is a legend:\n"
        "    'o' - whole note, lasts four beats\n"
        "    'o|' - half note, lasts two beats\n"
        "    '.|' - quater note, lasts one beat\n"
        "    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n"
        "    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
        '    """'
    )
    res_py = ask_clover(prompt_py, "sess_gc_verify_py17")
    print("=== TEST 1: LIVE PYTHON GENERATIVE COMPILATION (HumanEval/17 parse_music) ===")
    print("Reply:\n", res_py["reply"])
    print("Tool Calls:\n", json.dumps(res_py["tool_calls"], indent=2))
    gc_summary = next(t for t in res_py["tool_calls"] if t.get("tool") == "generative_compiler" and t.get("event") == "summary")
    assert gc_summary["rollbacks"] == 1 and gc_summary["solved_by_gc_rollback"] is True and gc_summary["final_ok"] is True

    FENCE = "`" * 3
    py_code = res_py["reply"].split(f"{FENCE}python")[-1].split(FENCE)[0].strip()
    ns = {}
    exec(py_code, ns)
    out_val = ns["parse_music"]("o o| .| o| o| .| .| .| .| o o")
    print("[PY-GC] Executed `parse_music` output:", out_val)
    assert out_val == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]

    # --- TEST 2: Live Rust Generative Compilation via `clover-k3-chat.service` ---
    prompt_rs = (
        "Write a Rust function inside a ```rust code block:\n"
        "```rust\n"
        "pub fn fmt_msg(msg: &mut String) -> String\n"
        "```\n"
        "that extracts the first 5 bytes as `tag` (cloned as an owned String before draining), calls `msg.drain(..5)`, "
        'checks `let prio = tag.starts_with("HIGH");`, and returns `format!("{prio}: {msg}")`.'
    )
    res_rs = ask_clover(prompt_rs, "sess_gc_verify_rs_fmt")
    print("\n=== TEST 2: LIVE RUST GENERATIVE COMPILATION (fmt_msg) ===")
    print("Reply:\n", res_rs["reply"])
    print("Tool Calls:\n", json.dumps(res_rs["tool_calls"], indent=2))
    rs_summary = next(t for t in res_rs["tool_calls"] if t.get("tool") == "generative_compiler" and t.get("event") == "summary")
    assert rs_summary["language"] == "rust" and rs_summary["final_ok"] is True

    # --- TEST 3: Direct Rust Sealor + `rustc` Prefix Exhaustion & Early Borrow/Type Error Detection ---
    rs_gc = CloverRustGenerativeCompiler()
    repaired_rs = (
        "const HEADER_SIZE: usize = 5;\n"
        "pub fn fmt_msg(msg: &mut String) -> String {\n"
        "    let tag = msg[..HEADER_SIZE].to_string();\n"
        "    msg.drain(..HEADER_SIZE);\n"
        '    let prio = tag.starts_with("HIGH");\n'
        '    format!("{prio}: {msg}")\n'
        "}\n"
    )
    r_lines = repaired_rs.splitlines()
    for k in range(1, len(r_lines) + 1):
        sub = "\n".join(r_lines[:k]) + "\n"
        v = rs_gc.check_prefix(sub, end_marker=(k == len(r_lines)))
        assert v.ok, f"False rejection at line {k}: {v}"
    print(f"\n=== TEST 3: RUST LINE-PREFIX COMPLETENESS ({len(r_lines)}/{len(r_lines)} prefixes accepted, 0 false rejections) ===")

    print("\nALL LIVE PYTHON & RUST GENERATIVE COMPILATION VERIFICATION TESTS PASSED!")


if __name__ == "__main__":
    main()
