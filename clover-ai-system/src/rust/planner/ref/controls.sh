#!/bin/bash
# Controls for the smoke test, run on the box: (1) mutants of the extracted text in temp copies (each must make smoke_ref FAIL), (2) the unmodified sources under ASan+UBSan.
# Works in /tmp/planner-ref-controls; the shipped files in ~/rustmig/planner/ref are only read.
S=$HOME/rustmig/planner/ref
INC="-I$HOME/rustmig/planner/include -I$HOME/rustmig/copyq/include -I$HOME/rustmig/slotcache/include -I$HOME/rustmig/routetable/include"
W=/tmp/planner-ref-controls; rm -rf $W; mkdir -p $W; cd $W
md5sum $S/planner_ref.cpp $S/planner_extracted.inc $S/smoke_ref.c $S/libplanner_ref.so
mutant() {   # name, sed expression
    d=$W/$1; mkdir -p $d; cp $S/planner_ref.cpp $S/smoke_ref.c $d/; sed -e "$2" $S/planner_extracted.inc > $d/planner_extracted.inc
    nd=$(diff <(tr -d '\r' < $S/planner_extracted.inc) <(tr -d '\r' < $d/planner_extracted.inc) | grep -c '^>')
    (cd $d && g++ -O2 -std=c++17 -fPIC -shared $INC planner_ref.cpp -o libplanner_ref.so && gcc -O1 -std=gnu11 -pthread $INC -o smoke_ref smoke_ref.c -L. -lplanner_ref -Wl,-rpath,'$ORIGIN' && ./smoke_ref > out.txt 2>&1; ec=$?; echo "$1: lines changed $nd; exit $ec; $(tail -1 out.txt); FAIL lines printed: $(grep -c '^FAIL line' out.txt)")
}
mutant M1 's/g_pend.size() >= g_pend_cap) { g_rc_planned/g_pend.size() > g_pend_cap) { g_rc_planned/'
mutant M2 's/== 1) { g_host.c.release/== 0) { g_host.c.release/'
mutant M3 's/g_host.cap_slots \* 8 \/ 10/g_host.cap_slots * 9 \/ 10/'
mutant M4 's/(long)g_rc_pos >= g_vq_from/(long)g_rc_pos > g_vq_from/'
mutant M5 's/g_host.c.remove(key); j.kind = 2;/g_host.c.remove(key); j.kind = 1;/'
mutant M6 's/i == pos ? (int)pos : rc_nu_sel(key)/i == pos ? (int)pos + 1 : rc_nu_sel(key)/'
mutant M7 's/g_rc_ok = false; slotcache_set_next_use(nullptr);/g_rc_ok = false;/'
mutant M8 's/g_pf_used++;/g_pf_used += 2;/'
mutant M9 's/g_pf_skipped++; continue;/continue;/'
mutant M10 's/if (!g_prefill \&\& g_host.cap_slots/if (g_host.cap_slots/'
mutant M11 's/g_host.cap_slots \&\& g_pend.size()/g_pend.size()/'
mutant M12 's/ \&\& g_host.c.worth(vk))/)/'
mutant M13 's/if (!hd || hd->fd < 0) continue;/(void)0;/'
mutant M14 's/if (pool.ready \&\& pool.c.contains(key)) continue;/if (pool.c.contains(key)) continue;/'
mutant M15 's/(pool.ready \&\& pool.c.contains(key)) || g_host/(pool.c.contains(key)) || g_host/'
mutant M16 's/a = f->second; g_arr.erase(f);/a = f->second;/'
mutant M17 's/g_wb_keys.push_back(vk);//'
mutant M18 's/g_arr\[key\] = j.arr;//'
mutant M19 's/state.load(std::memory_order_acquire) == 0) out++/state.load(std::memory_order_acquire) == 1) out++/'
mutant M20 's/out < g_vq_cap/out <= g_vq_cap/'
mutant M21 's/pos + (size_t)g_lookahead + 1/pos + (size_t)g_lookahead/g'
mutant M22 's/std::max(g_rc_planned, pos)/pos/'
mutant M23 's/g_vq \&\& (long)g_rc_pos >= g_vq_from/(long)g_rc_pos >= g_vq_from/'
echo "--- control: an unchanged copy through the same mutant() path (must PASS)"
mutant M0 's/XXXXXXXXXXXX/YYYYYYYYYYYY/'
echo "--- ASan + UBSan on the unmodified sources"
d=$W/asan; mkdir -p $d; cp $S/planner_ref.cpp $S/planner_extracted.inc $S/smoke_ref.c $d/; cd $d
g++ -O1 -g -std=c++17 -fPIC -shared -fsanitize=address,undefined -fno-sanitize-recover=undefined $INC planner_ref.cpp -o libplanner_ref.so && \
gcc -O1 -g -std=gnu11 -pthread -fsanitize=address,undefined $INC -o smoke_ref smoke_ref.c -L. -lplanner_ref -Wl,-rpath,'$ORIGIN' && \
ASAN_OPTIONS=detect_leaks=0 ./smoke_ref > out.txt 2>&1; echo "asan: exit $?; $(tail -3 out.txt | tr '\n' '|')"
