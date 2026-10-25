#!/usr/bin/env python3
"""Writes mutated COPIES of ../ref/copyq_ref.cu into mutants/ (the ref file itself is only read). Every edit must match exactly the expected number of times."""
import sys, difflib
src = open('../ref/copyq_ref.cu', 'rb').read().decode('utf-8')

def mut(name, edits):
    t = src
    for old, new, n in edits:
        found = t.count(old)
        if found != n:
            sys.exit(f'{name}: {old!r} matched {found} times, expected {n}')
        t = t.replace(old, new)
    open(f'mutants/copyq_ref_{name}.cu', 'wb').write(t.encode('utf-8'))
    d = list(difflib.unified_diff(src.splitlines(), t.splitlines(), 'ref/copyq_ref.cu', f'mutants/copyq_ref_{name}.cu', lineterm='', n=0))
    open(f'mutants/{name}.diff', 'w').write('\n'.join(d) + '\n')
    print(f'{name}: written, {sum(1 for l in d if l[:1] in "+-" and l[:3] not in ("+++", "---"))} changed lines')

ARR = '                jb.arr->state.store(1, std::memory_order_release);\r\n'
SYNC = '                CHECK(cudaStreamSynchronize(s));\r\n'
# M1: arr_state is set BEFORE the final stream synchronize (hence also before rel_state)
mut('M1', [(ARR, '', 1), (SYNC + '                if (jb.rel)', ARR + SYNC + '                if (jb.rel)', 1)])
# M2: both host->device copies move EXPERT_RAW-1 bytes
mut('M2', [('EXPERT_RAW, cudaMemcpyHostToDevice', 'EXPERT_RAW - 1, cudaMemcpyHostToDevice', 2)])
# M2v: the victim copy (card -> victim_host) moves EXPERT_RAW-1 bytes
mut('M2v', [('jb.dst, EXPERT_RAW, cudaMemcpyDeviceToHost', 'jb.dst, EXPERT_RAW - 1, cudaMemcpyDeviceToHost', 1)])
# M3: act_gate does nothing
G = 'static inline void act_gate() { while (g_act_hold.load(std::memory_order_acquire)) std::this_thread::yield(); }'
mut('M3', [(G, 'static inline void act_gate() { }', 1)])
# M4: the drive reads start one byte late
mut('M4', [('jb.off + (off_t)got', 'jb.off + 1 + (off_t)got', 2)])
