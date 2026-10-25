use super::*;
use std::ffi::CString;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::process::Command;

fn call(layer: i32, experts: &[i32], counts: &[i32]) -> Call {
    Call { layer, experts: experts.to_vec(), counts: counts.to_vec() }
}

fn one(line: &[u8]) -> Call {
    parse_call(line)
}

fn tmp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("routetable-test-{}-{}", std::process::id(), name));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn load_path(p: &std::path::Path, experts: i32) -> *mut rt_table {
    let c = CString::new(p.as_os_str().as_bytes()).unwrap();
    unsafe { rt_load(c.as_ptr(), experts) }
}

#[test]
fn strtol_whitespace_sign_digits_and_no_conversion() {
    assert_eq!(strtol(b"  42x", 0), (42, 4));
    assert_eq!(strtol(b"+7", 0), (7, 2));
    assert_eq!(strtol(b"-7", 0), (-7, 2));
    assert_eq!(strtol(b"\t\n\x0b\x0c\r 5", 0), (5, 7));
    assert_eq!(strtol(b"0012", 0), (12, 4));
    // nothing consumed (end = start, whitespace and sign included) when there is no digit
    assert_eq!(strtol(b"- 7", 0), (0, 0));
    assert_eq!(strtol(b"  +", 0), (0, 0));
    assert_eq!(strtol(b"  x", 0), (0, 0));
    assert_eq!(strtol(b"", 0), (0, 0));
    assert_eq!(strtol(b"3 5", 1), (5, 3));
    assert_eq!(strtol(b"3 x", 1), (0, 1));
    // base 10: "0x1f" is 0 followed by 'x'
    assert_eq!(strtol(b"0x1f", 0), (0, 1));
}

#[test]
fn strtol_saturates_at_64_bit_long_and_still_consumes_all_digits() {
    assert_eq!(strtol(b"9223372036854775807", 0), (i64::MAX, 19));
    assert_eq!(strtol(b"9223372036854775808", 0), (i64::MAX, 19));
    assert_eq!(strtol(b"18446744073709551615", 0), (i64::MAX, 20));
    assert_eq!(strtol(b"18446744073709551616", 0), (i64::MAX, 20));
    let nines = format!("{}z", "9".repeat(23));
    assert_eq!(strtol(nines.as_bytes(), 0), (i64::MAX, 23));
    assert_eq!(strtol(b"-9223372036854775808", 0), (i64::MIN, 20));
    assert_eq!(strtol(b"-9223372036854775809", 0), (i64::MIN, 20));
    assert_eq!(strtol(b"-18446744073709551616", 0), (i64::MIN, 21));
    assert_eq!(strtol(b"-9223372036854775807", 0), (-i64::MAX, 20));
    let zeros = format!("{}12", "0".repeat(25));
    assert_eq!(strtol(zeros.as_bytes(), 0), (12, 27));
}

#[test]
fn cast_to_int_truncates_to_low_32_bits() {
    // layer field of "N 0": value after strtol then (int)
    assert_eq!(one(b"4294967297 0\n").layer, 1);
    assert_eq!(one(b"4294967296 0\n").layer, 0);
    assert_eq!(one(b"2147483647 0\n").layer, i32::MAX);
    assert_eq!(one(b"2147483648 0\n").layer, i32::MIN);
    assert_eq!(one(b"-2147483649 0\n").layer, i32::MAX);
    assert_eq!(one(b"9223372036854775807 0\n").layer, -1); // LONG_MAX -> int -1
    assert_eq!(one(b"9223372036854775808 0\n").layer, -1); // saturated, same
    assert_eq!(one(b"-9223372036854775809 0\n").layer, 0); // LONG_MIN -> int 0
    assert_eq!(one(b"1 0 4294967297:4294967297 4294967298:4294967299\n"), call(1, &[1, 2], &[1, 3]));
}

#[test]
fn empty_line_is_layer0_without_experts_and_every_line_is_a_call() {
    assert_eq!(parse_data(b"\n"), vec![call(0, &[], &[])]);
    assert_eq!(parse_data(b"\n\n3 5 7\n\n"), vec![call(0, &[], &[]), call(0, &[], &[]), call(3, &[7], &[]), call(0, &[], &[])]);
    assert_eq!(parse_data(b"   \n"), vec![call(0, &[], &[])]);
    assert_eq!(parse_data(b"garbage line\n"), vec![call(0, &[], &[])]);
}

#[test]
fn layer_only_and_second_number_ignored() {
    assert_eq!(one(b"7\n"), call(7, &[], &[]));
    assert_eq!(one(b"7 99\n"), call(7, &[], &[]));
    assert_eq!(one(b"35 7\n"), call(35, &[], &[])); // 7 is the ignored second number
    assert_eq!(one(b"3 5 7 8 9\n"), call(3, &[7, 8, 9], &[]));
    assert_eq!(one(b"  5 6 7\n"), call(5, &[7], &[]));
    assert_eq!(one(b"x 3 4\n"), call(0, &[], &[])); // no digits for layer or ntok; the loop stops at 'x'
}

#[test]
fn line_ending_at_a_space() {
    assert_eq!(one(b"4 1 2 \n"), call(4, &[2], &[]));
    assert_eq!(one(b"4 1 \n"), call(4, &[], &[]));
    assert_eq!(one(b"4 \n"), call(4, &[], &[]));
    assert_eq!(one(b"4 1 2 "), call(4, &[2], &[])); // no newline and a trailing space
}

#[test]
fn non_digit_stops_the_expert_loop() {
    assert_eq!(one(b"3 5 7 x 9\n"), call(3, &[7], &[]));
    assert_eq!(one(b"3 5 7x9\n"), call(3, &[7], &[]));
    assert_eq!(one(b"3 5 7:5:6\n"), call(3, &[7], &[5])); // second colon is not a digit
}

#[test]
fn tab_is_skipped_by_strtol_but_not_by_the_expert_loop() {
    assert_eq!(one(b"3\t5 7\n"), call(3, &[7], &[]));
    assert_eq!(one(b"3 5\t7\n"), call(3, &[], &[]));
    assert_eq!(one(b"3 5 7\t8\n"), call(3, &[7], &[]));
    assert_eq!(one(b"3 5 7:\t4 8\n"), call(3, &[7, 8], &[4])); // strtol after the colon skips the tab
    assert_eq!(one(b"3 \t5 7\n"), call(3, &[7], &[]));
}

#[test]
fn signs_in_layer_ntok_experts_and_counts() {
    assert_eq!(one(b"+3 +5 7\n"), call(3, &[7], &[]));
    assert_eq!(one(b"-3 -5 7\n"), call(-3, &[7], &[]));
    assert_eq!(one(b"3 5 -7\n"), call(3, &[], &[])); // '-' is not a digit: the loop stops
    assert_eq!(one(b"3 5 +7\n"), call(3, &[], &[]));
    assert_eq!(one(b"3 5 7:-2\n"), call(3, &[7], &[-2]));
    assert_eq!(one(b"3 5 7:+2\n"), call(3, &[7], &[2]));
    assert_eq!(one(b"3 5 7:- 2 8\n"), call(3, &[7], &[0])); // sign without digit: count 0, only the colon consumed, then '-' stops the loop
}

#[test]
fn colon_without_digits_pushes_count_zero_and_consumes_only_the_colon() {
    assert_eq!(one(b"3 5 7:\n"), call(3, &[7], &[0]));
    assert_eq!(one(b"3 5 7:x 8\n"), call(3, &[7], &[0]));
    assert_eq!(one(b"3 5 7:+ 8\n"), call(3, &[7], &[0]));
    assert_eq!(one(b"3 5 7: 8 9\n"), call(3, &[7, 9], &[8])); // whitespace after the colon is skipped by strtol: 8 is the count
    assert_eq!(one(b"3 5 7: x\n"), call(3, &[7], &[0]));
    assert_eq!(one(b"3 5 7 : 4\n"), call(3, &[7], &[])); // a space before the colon: no count, the loop stops at ':'
}

#[test]
fn counts_and_experts_are_independent_lists() {
    assert_eq!(one(b"3 5 1:10 2 3:30\n"), call(3, &[1, 2, 3], &[10, 30]));
    assert_eq!(one(b"3 5 1 2 3\n"), call(3, &[1, 2, 3], &[]));
    assert_eq!(one(b"3 5 1:10:20\n"), call(3, &[1], &[10]));
    assert_eq!(one(b"3 5 1:10  2:20\n"), call(3, &[1, 2], &[10, 20])); // several spaces
}

#[test]
fn no_trailing_newline_and_crlf() {
    assert_eq!(parse_data(b"1 2 3\n4 5 6"), vec![call(1, &[3], &[]), call(4, &[6], &[])]);
    assert_eq!(parse_data(b"7"), vec![call(7, &[], &[])]);
    assert_eq!(parse_data(b"1 2 3:4\r\n5 6 7\r\n"), vec![call(1, &[3], &[4]), call(5, &[7], &[])]);
    assert_eq!(parse_data(b"1 2 3 \r\n"), vec![call(1, &[3], &[])]);
    assert_eq!(parse_data(b"1 2 3\r"), vec![call(1, &[3], &[])]); // a lone CR does not end a line
    assert_eq!(parse_data(b"\r\n"), vec![call(0, &[], &[])]);
    assert_eq!(parse_data(b"1\r2 3 4\n"), vec![call(1, &[3, 4], &[])]); // CR inside the line is whitespace for strtol: layer 1, ntok 2
}

#[test]
fn empty_input_has_no_lines_and_a_final_newline_adds_none() {
    assert_eq!(parse_data(b""), vec![]);
    assert_eq!(parse_data(b"1 2 3\n").len(), 1);
    assert_eq!(parse_data(b"1 2 3\n\n").len(), 2);
}

#[test]
fn nul_ends_the_string_but_not_the_line() {
    assert_eq!(one(b"1 2 3\0 4 5\n"), call(1, &[3], &[]));
    assert_eq!(parse_data(b"1 2 3\0\n4 5 6\n"), vec![call(1, &[3], &[]), call(4, &[6], &[])]);
    assert_eq!(parse_data(b"\0\n1 2 3\n"), vec![call(0, &[], &[]), call(1, &[3], &[])]);
    assert_eq!(one(b"1 2 7:\0 5\n"), call(1, &[7], &[0]));
}

#[test]
fn high_bytes_are_not_digits() {
    assert_eq!(one(b"1 2 \xb2 3\n"), call(1, &[], &[]));
    assert_eq!(one(b"1 2 3\xff4\n"), call(1, &[3], &[]));
}

fn sample() -> rt_table {
    // key = layer * 10 + e:  5 -> calls [0,2,2], 6 -> [0,3], 7 -> [2], 15 -> [1]
    rt_table::from_calls(parse_data(b"0 0 5 6\n1 0 5\n0 0 5 5 7\n0 0 6\n"), 10)
}

#[test]
fn next_use_is_strictly_greater_than_cur_and_lists_are_ascending_with_duplicates() {
    let mut t = sample();
    assert_eq!(t.uses[&5].0, vec![0, 2, 2]);
    assert_eq!(t.uses[&6].0, vec![0, 3]);
    assert_eq!(t.uses[&7].0, vec![2]);
    assert_eq!(t.uses[&15].0, vec![1]);
    assert_eq!(t.uses.len(), 4);
    t.cur = -1; // initial value
    assert_eq!(t.next_use(5), 0);
    t.cur = 0;
    assert_eq!(t.next_use(5), 2); // 0 is not > 0
    assert_eq!(t.next_use(6), 3);
    assert_eq!(t.next_use(7), 2);
    assert_eq!(t.next_use(15), 1);
    t.cur = 1;
    assert_eq!(t.next_use(15), EXHAUSTED); // 1 is not > 1
    t.cur = 2;
    assert_eq!(t.next_use(5), EXHAUSTED); // both listings of call 2 are skipped
    assert_eq!(t.next_use(7), EXHAUSTED);
    assert_eq!(t.next_use(6), 3);
    t.cur = 3;
    assert_eq!(t.next_use(6), EXHAUSTED);
}

#[test]
fn initial_current_is_minus_one() {
    let mut t = sample();
    assert_eq!(t.cur, -1);
    assert_eq!(t.next_use(5), 0);
    assert_eq!(t.next_use(15), 1);
}

#[test]
fn pointer_only_moves_forward() {
    let mut t = sample();
    t.cur = 0;
    assert_eq!(t.next_use(6), 3); // pointer of key 6 now at index 1
    t.cur = -1;
    assert_eq!(t.next_use(6), 3); // a fresh scan would give 0; the pointer does not go back
    assert_eq!(t.next_use(5), 0); // key 5 was never asked: its pointer is still at 0
    t.cur = 2;
    assert_eq!(t.next_use(5), EXHAUSTED);
    t.cur = 0;
    assert_eq!(t.next_use(5), EXHAUSTED); // stays exhausted
    assert_eq!(t.next_use(15), 1);
}

#[test]
fn unknown_key_returns_exhausted_and_creates_no_state() {
    let mut t = sample();
    assert_eq!(t.next_use(99), EXHAUSTED);
    assert_eq!(t.next_use(-3), EXHAUSTED);
    assert_eq!(t.uses.len(), 4);
    // asking before the key exists in any call must not disturb the known ones either
    t.cur = 0;
    assert_eq!(t.next_use(5), 2);
}

#[test]
fn duplicated_expert_in_one_call_is_listed_twice() {
    let t = rt_table::from_calls(parse_data(b"2 0 4 4 4\n"), 10);
    assert_eq!(t.uses[&24].0, vec![0, 0, 0]);
    let mut t = t;
    assert_eq!(t.next_use(24), 0);
    t.cur = 0;
    assert_eq!(t.next_use(24), EXHAUSTED);
}

#[test]
fn key_uses_wrapping_int_arithmetic() {
    // 65536 * 65536 = 2^32 wraps to 0 in a C int (signed overflow is UB in C; here it wraps)
    let mut t = rt_table::from_calls(parse_data(b"65536 0 3\n-1 0 5\n"), 65536);
    assert_eq!(t.next_use(3), 0);
    assert_eq!(t.next_use(-65536 + 5), 1);
    assert_eq!(ekey(i32::MAX, 2, 0), -2);
}

#[test]
fn set_current_takes_a_long_and_compares_without_truncation() {
    let c = tmp_file("cur", b"0 0 1\n1 0 1\n");
    unsafe {
        let t = load_path(&c, 10); // keys: call 0 -> 1, call 1 -> 11
        rt_set_current(t, -5);
        assert_eq!(rt_next_use(t, 1), 0);
        rt_set_current(t, 0);
        assert_eq!(rt_next_use(t, 1), EXHAUSTED);
        rt_set_current(t, c_long::MIN);
        assert_eq!(rt_next_use(t, 11), 1);
        rt_set_current(t, 4294967296); // would be 0 if truncated to int
        assert_eq!(rt_next_use(t, 11), EXHAUSTED);
        rt_free(t);
    }
}

#[test]
fn ffi_accessors_lengths_and_null_for_empty_lists() {
    let p = tmp_file("acc", b"3 5 7:2 8\n\n9 1\n4 0 1 2 3:6:7\n");
    unsafe {
        let t = load_path(&p, 10);
        assert_eq!(rt_calls(t), 4);
        assert_eq!(rt_layer(t, 0), 3);
        assert_eq!(rt_layer(t, 1), 0);
        assert_eq!(rt_layer(t, 2), 9);
        assert_eq!(rt_layer(t, 3), 4);
        let mut n = 99usize;
        let e = rt_experts(t, 0, &mut n);
        assert_eq!(n, 2);
        assert!(!e.is_null());
        assert_eq!(std::slice::from_raw_parts(e, n), &[7, 8]);
        let c = rt_counts(t, 0, &mut n);
        assert_eq!(n, 1);
        assert!(!c.is_null());
        assert_eq!(std::slice::from_raw_parts(c, n), &[2]);
        // empty call: n = 0 written, pointers NULL (allowed by the contract)
        n = 99;
        assert!(rt_experts(t, 1, &mut n).is_null());
        assert_eq!(n, 0);
        n = 99;
        assert!(rt_counts(t, 1, &mut n).is_null());
        assert_eq!(n, 0);
        // experts but no counts
        n = 99;
        assert!(!rt_experts(t, 3, &mut n).is_null());
        assert_eq!(n, 3);
        n = 99;
        assert!(!rt_counts(t, 3, &mut n).is_null());
        assert_eq!(n, 1);
        n = 99;
        assert!(rt_counts(t, 2, &mut n).is_null());
        assert_eq!(n, 0);
        rt_free(t);
        rt_free(std::ptr::null_mut());
    }
}

#[test]
fn empty_file_directory_and_ffi_next_use() {
    unsafe {
        let p = tmp_file("empty", b"");
        let t = load_path(&p, 10);
        assert_eq!(rt_calls(t), 0);
        assert_eq!(rt_next_use(t, 0), EXHAUSTED);
        rt_free(t);
        // a directory opens (fopen "r" does) and then reads as nothing
        let t = load_path(&std::env::temp_dir(), 10);
        assert_eq!(rt_calls(t), 0);
        rt_free(t);
        // file read path with a real trace
        let p = tmp_file("trace", b"0 3 1:1 2:1\n1 3 1:1\n0 3 2:1\n");
        let t = load_path(&p, 8);
        assert_eq!(rt_calls(t), 3);
        assert_eq!(rt_next_use(t, 1), 0);
        rt_set_current(t, 0);
        assert_eq!(rt_next_use(t, 2), 2);
        assert_eq!(rt_next_use(t, 9), 1);
        rt_set_current(t, 2);
        assert_eq!(rt_next_use(t, 2), EXHAUSTED);
        rt_free(t);
    }
}

fn child_missing(env_path: &[u8], expect_stderr: &[u8], test: &str) {
    if let Some(v) = std::env::var_os("RT_CHILD_PATH") {
        let c = CString::new(v.into_vec()).unwrap();
        unsafe { rt_load(c.as_ptr(), 10) };
        std::process::exit(99); // not reached: rt_load must exit(1)
    }
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe)
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env("RT_CHILD_PATH", OsStr::from_bytes(env_path))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "stdout: {:?}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(out.stderr, expect_stderr);
}

#[test]
fn missing_file_exits_1_with_exact_stderr() {
    child_missing(b"/nonexistent-dir/route.cache", b"cannot read route cache /nonexistent-dir/route.cache\n", "tests::missing_file_exits_1_with_exact_stderr");
}

#[test]
fn missing_file_with_non_utf8_path_prints_the_raw_bytes() {
    child_missing(b"/nonexistent-dir/\xff\xfe.cache", b"cannot read route cache /nonexistent-dir/\xff\xfe.cache\n", "tests::missing_file_with_non_utf8_path_prints_the_raw_bytes");
}
