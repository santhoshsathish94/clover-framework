# routetable (Rust) — notes

Implements `include/routetable.h` (fixed contract; not edited). Behaviour reference: `struct CallSet`, `g_rc`, `g_uses`, `g_use_ptr`, `g_rc_cur`, `rc_nu` and `rc_load` in `src/code/gpu/backend-gpu.cu` (read 2026-10-22, lines 52-66 and 369-387).

Layout: `src/lib.rs` (implementation and C ABI), `src/tests.rs` (unit tests, `#[cfg(test)]`), `smoke_rust.c` (C program against `libroutetable.so`).

## Design

- `rt_table` = `Vec<Call>` (`layer: i32`, `experts: Vec<i32>`, `counts: Vec<i32>`), `uses: HashMap<i32, (Vec<i32>, usize)>` (ascending call numbers and the forward-only pointer, one combined entry per key), `cur: i64` (C `long` on LP64, set from `c_long`). Initial `cur` = -1.
- `rt_load` reads the whole file as bytes, splits it after each `\n` (`split_inclusive`: the last line may lack the newline; empty input has no lines; a final `\n` adds no extra line), parses each line, then builds `uses` by walking the calls in order and pushing the call number (`i as i32`) once per listed expert (duplicates are listed twice).
- `strtol` is re-implemented byte-wise: skips C-locale whitespace (space, `\t \n \v \f \r`), optional sign, base-10 digits, saturates at `i64::MAX` / `i64::MIN`, keeps consuming digits after saturation, consumes nothing (end = start, whitespace and sign included) when there is no digit. The original's `(int)` cast is `as i32` (low 32 bits, what gcc does).
- A line is cut at its first NUL (the C string functions would stop there); a NUL does not end the line (getline does not stop at NUL). Past the end `at()` returns 0, which plays the terminating NUL.
- Open failure: writes `cannot read route cache <path bytes>\n` with `write_all` on `stderr()` (errors ignored, no `eprintln!`), then `std::process::exit(1)`. The path is the raw bytes of the C string (`OsStr::from_bytes`), so non-UTF-8 names work and are echoed unchanged.
- Exported functions that dereference pointers are `unsafe extern "C"`; release profile has `panic = "abort"`. Not thread-safe.

## Behaviour map (original -> Rust)

| Original | Rust |
|---|---|
| `fopen(path,"r")` fails -> message, `exit(1)` | `File::open` fails -> `fatal_cannot_read` |
| `getline(...) > 0` per line, every line pushes a `CallSet` | `parse_data`: one `Call` per line |
| `cs.layer = (int)strtol(p,&end,10); p = end;` | `strtol(s,0)`, `as i32` |
| `strtol(p,&end,10); p = end;` (ntok ignored) | `p = strtol(s,p).1` |
| `while (*p == ' ') p++;` (only spaces) | `while at(s,p) == b' '` |
| `if (*p < '0' \|\| *p > '9') break;` (`char` signed: bytes >= 0x80 also break) | `!at(s,p).is_ascii_digit()` |
| `experts.push_back((int)strtol(p,&end,10)); p = end;` | same |
| `if (*p == ':') { counts.push_back((int)strtol(p+1,&end,10)); p = end; }` (no digits: pushes 0, `end = p+1`) | same; `strtol` returns its start index when nothing converts |
| `g_uses[ekey(layer,e)].push_back((int)i)` | `ekey` with `wrapping_mul` / `wrapping_add`, `HashMap::entry` |
| `rc_nu`: unknown key -> 1000000000, no state created | `get_mut` is `None` -> `EXHAUSTED` |
| `rc_nu`: `while (p < l.size() && l[p] <= g_rc_cur) p++`; `p < size ? l[p] : 1000000000` | same, `cur` compared as `i64` |
| `g_rc_cur = cur` | `rt_set_current` |

## Choices where the original is silent or outside the contract

- `ekey`: `layer * experts + e` is signed `int` arithmetic; overflow is undefined in C, here it wraps (x86-64 gcc without `-ftrapv` does the same in practice; not checked against the original).
- `fopen` on a directory succeeds and `getline` then returns -1, so the original sees zero calls. Here `File::open` succeeds and a read error ends the data (bytes read before the error are kept), so a directory also gives zero calls. Observed with a small C program on the box (`fopen("/tmp","r")` ok, `getline` = -1); a read error in the middle of a real file is not tested.
- `rt_experts` / `rt_counts` return NULL when the list is empty (the contract allows it), non-NULL otherwise; `*n` is written when `n` is non-NULL.
- `rt_layer`, `rt_experts`, `rt_counts` with `i >= rt_calls`: outside the contract; the index check aborts the process.
- `rt_free(NULL)` is a no-op. A NULL `t` elsewhere, a NULL `path`, a second `rt_load` without `rt_free` (tables are independent by construction, not tested), concurrent use: outside the contract, not handled.
- `rt_calls` counts and call numbers use `i as i32` as the original's `(int)i` (only differs above 2^31 calls).
- Header ambiguity: none found that changed the implementation. The text "skip spaces" means only `' '` (the original); this is what was implemented, not tabs.

## Tests (`cargo test`, 26 tests, hand-derived expectations)

- `strtol_*`, `cast_to_int_truncates_to_low_32_bits`: whitespace set, signs, no-digit consumption, saturation at both ends with the number of consumed digits, `4294967297 -> 1`, `LONG_MAX -> -1`, `LONG_MIN -> 0`, `2147483648 -> INT_MIN`.
- `empty_line_*`, `layer_only_*`, `line_ending_at_a_space`, `non_digit_stops_the_expert_loop` (`3 5 7 x 9`), `tab_is_skipped_by_strtol_but_not_by_the_expert_loop`, `signs_*`, `colon_without_digits_*`, `counts_and_experts_are_independent_lists`, `no_trailing_newline_and_crlf`, `empty_input_*`, `nul_ends_the_string_but_not_the_line`, `high_bytes_are_not_digits`.
- Next use: `next_use_is_strictly_greater_*` (lists with duplicates, ascending, strict `>`), `initial_current_is_minus_one`, `pointer_only_moves_forward`, `unknown_key_*` (no state created), `duplicated_expert_in_one_call_is_listed_twice`, `key_uses_wrapping_int_arithmetic`, `set_current_takes_a_long_*`.
- C ABI: `ffi_accessors_lengths_and_null_for_empty_lists`, `empty_file_directory_and_ffi_next_use` (empty file, directory, small trace through `rt_load`).
- `missing_file_exits_1_with_exact_stderr`, `missing_file_with_non_utf8_path_prints_the_raw_bytes`: the test binary is re-run as a child (env `RT_CHILD_PATH`), exit status 1 and stderr byte-for-byte.
- Checked on the box: the strtol / cast expectations of the tests (not the Rust code) were also run against glibc's `strtol` in a small C program (all matched). Negative controls in a scratch copy: `<` instead of `<=` in the pointer skip fails 6 tests; also skipping tabs in the expert loop fails 1 test. A third mutation (the colon consuming nothing when there are no digits) is not observable: after a digitless count the loop stops in either case, so no test can tell them apart (the mutation text was not confirmed to have been applied).

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/routetable`)

    cargo build --release                       # target/release/libroutetable.so
    cargo test                                  # or --release
    gcc -O1 -I include -o smoke_rust smoke_rust.c -L target/release -lroutetable -Wl,-rpath,$PWD/target/release && ./smoke_rust

`smoke_rust` writes a 6-line trace to a temp file (empty line, CRLF line, unterminated last line, colon without digits, duplicated expert), loads it with 10 experts per layer and checks accessors and next-use values, forward-only pointer and unknown key.

## Not covered

- Equality with the C++ original: not claimed; another agent compares them (`ref/`).
- A real route cache from `K3GPU_TRACE`; only hand-written traces. Large files and performance (no measurement).
- A read error halfway through a file; very long lines; more than 2^31 calls; Windows or 32-bit (layout and `long` size only reasoned for x86-64 Linux LP64).
- The fatal path inside the engine (only the child-process tests exercise it).
