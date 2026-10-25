#!/usr/bin/env python3
# PROPOSAL, not applied to the real driver: builds a scratch copy of verify/pi_driver.cpp (work/proto/pi_driver_proto.cpp) with three extra checks and compiles it, to show that the checks kill
# the mutants the real driver missed (M03, M11-M15). The real pi_driver.cpp is only read.
#   1. every interposed mmap of a host step sleeps 60 ms; the library's reported seconds (printed with one decimal) must lie in [round1(ceil(n_chunks/16) * 0.060), wall time of the call + 0.06]  (kills M11 M12 M13 M15)
#   2. clock_gettime is interposed: each CLOCK_REALTIME reading while armed is shifted by +1000 s * (number of readings so far), CLOCK_MONOTONIC is real  (kills M14, CLOCK_REALTIME instead of CLOCK_MONOTONIC)
#   3. with 60 ms sleeps and n_chunks >= 16 every one of the 16 workers gets its first chunk, so the calling thread must have called mmap at least once  (kills M03)
# usage (box, ~/rustmig/poolinit/verify): python3 mutants/proto_driver.py     -> work/proto/pi_driver_proto
import os, subprocess

V = os.path.expanduser("~/rustmig/poolinit/verify")
src = open(V + "/pi_driver.cpp", newline="").read()
os.makedirs(V + "/work/proto", exist_ok=True)

def rep(old, new, cnt=1):
    global src
    assert src.count(old) == cnt, (old[:70], src.count(old))
    src = src.replace(old, new)

rep("#include <algorithm>\n", "#include <algorithm>\n#include <cmath>\n")
rep("static std::atomic<uint64_t> g_first_op_seq{ ~0ULL };\n",
    "static std::atomic<uint64_t> g_first_op_seq{ ~0ULL };\n"
    "static std::atomic<int> g_delay_ms{ 0 }, g_rt_calls{ 0 };\n"
    "static std::atomic<long> g_caller_tid{ 0 };\n"
    "static std::atomic<unsigned> g_caller_maps{ 0 };\n"
    "static double g_tcall = 0; static int g_delay_used = 0;\n"
    "static double mono_now() { struct timespec t; syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }\n")
rep("    const unsigned call = g_mm_calls.fetch_add(1) + 1;\n",
    "    const unsigned call = g_mm_calls.fetch_add(1) + 1;\n"
    "    if (syscall(SYS_gettid) == g_caller_tid.load()) g_caller_maps.fetch_add(1);\n"
    "    if (g_delay_ms.load()) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = (long)g_delay_ms.load() * 1000000L; nanosleep(&ts, nullptr); }\n")
rep("// ------------------------------------------------------------------------------------------------ unit configuration",
    "extern \"C\" int clock_gettime(clockid_t c, struct timespec *t) noexcept\n"
    "{\n"
    "    const int r = (int)syscall(SYS_clock_gettime, c, t);\n"
    "    if (r == 0 && c == CLOCK_REALTIME && g_armed.load(std::memory_order_acquire)) t->tv_sec += 1000L * (g_rt_calls.fetch_add(1) + 1);\n"
    "    return r;\n"
    "}\n\n"
    "// ------------------------------------------------------------------------------------------------ unit configuration")
rep("        PI.host(&OPS, &cfg, &G_HO);\n",
    "        g_delay_used = s.mmap_fail_at == 0 ? 60 : 0; g_delay_ms.store(g_delay_used); g_rt_calls.store(0); g_caller_maps.store(0); g_caller_tid.store(syscall(SYS_gettid));\n"
    "        { const double tc0 = mono_now(); PI.host(&OPS, &cfg, &G_HO); g_tcall = mono_now() - tc0; }\n"
    "        g_delay_ms.store(0);\n")
rep("    const std::string got = mask_time(new_stderr(), &shape);\n",
    "    const std::string raw_err = new_stderr();\n"
    "    const std::string got = mask_time(raw_err, &shape);\n"
    "    if (shape && X.s.kind == 'H' && !g_model.fatal && g_delay_used > 0) {\n"
    "        const size_t n = (X.s.bytes / X.s.raw) / 32, p = raw_err.find(\"pinned in \");\n"
    "        if (p != std::string::npos) {\n"
    "            const double secs = atof(raw_err.c_str() + p + 10), lo = std::floor((double)((n + 15) / 16) * g_delay_used / 100.0 + 0.5) / 10.0 - 0.001, hi = g_tcall + 0.06;\n"
    "            if (secs < lo || secs > hi) viol(sfmt(\"PROTO: reported %.1f s outside [%.3f, %.3f] (n_chunks %zu, wall of the call %.3f)\", secs, lo, hi, n, g_tcall));\n"
    "        }\n"
    "    }\n")
rep("    for (uintptr_t a : regs) syscall(SYS_munmap, a, chunk);\n",
    "    if (g_delay_used > 0 && n >= 16 && g_caller_maps.load() == 0) viol(sfmt(\"PROTO: n_chunks %zu >= 16 with 60 ms mmaps but the calling thread never called mmap\", n));\n"
    "    for (uintptr_t a : regs) syscall(SYS_munmap, a, chunk);\n")
open(V + "/work/proto/pi_driver_proto.cpp", "w", newline="").write(src)
inc = V + "/../include"
r = subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-pthread", "-rdynamic", "-I" + V + "/../include", "-I" + V + "/../../slotcache/include",
                    V + "/work/proto/pi_driver_proto.cpp", "-o", V + "/work/proto/pi_driver_proto", "-ldl"], capture_output=True, text=True)
print(r.returncode, (r.stdout + r.stderr)[-1500:])
