#!/bin/bash
# Counts the events of each probe from its .ev file. The driver's workers append concurrently, so lines can run together; the tags are therefore counted as substrings (grep -o), not as lines.
cd ~/rustmig/engcfg/verify/out/probe || exit 1
TAGS="neg_exact_2^63 pos_exact_i64max digits_left_after_overflow layer-1_overflows layer*cards_overflows trap_input_reached top_negative fill_negative n_negative_reaches_dealing ms_expression_evaluated more_counts_than_experts fits_false_total_ok sum_outside_i32 ntop_outside_i32 fill_reaches_255 fill_reaches_65535 s_reaches_2^31 pend_cap_product_wraps emit_called value_differs print_differs evaluated"
for f in $(ls P*.ev 2>/dev/null | sort -V); do
  n=${f%.ev}; printf "%s:" "$n"
  for t in $TAGS; do
    [ "$t" = evaluated ] && [ "$n" = P114 ] && continue
    c=$(tr -d '\n' < "$f" | grep -o -F "$t" | wc -l)
    [ "$c" -gt 0 ] && printf " %s=%d" "$t" "$c"
  done
  echo
done
