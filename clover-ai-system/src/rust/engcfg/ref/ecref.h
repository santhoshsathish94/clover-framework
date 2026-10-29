/* Extras of the C++ reference that are NOT part of the contract (engcfg.h): the composite run of the engine's whole configure() text and a snapshot of what it left in the globals.
   Only libengcfg_ref.so exports these (names start with ecref_). The differential driver compares them with a sequence of contract calls. */
#ifndef ECREF_H
#define ECREF_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ecref_snap {
    int configured, cards, readers, verbose;
    size_t reserve;
    int policy, chunks, trace_open;
    size_t host_bytes, host_slots, rc_calls;
    int lookahead;
    size_t pend_cap;
    int rc_ok, nu_set, vq;
    size_t vq_cap;
    long vq_from;
    int hook_set, layout, prefill, force_set;
} ecref_snap;

/* Runs the original configure() with stand-ins: devices = what cudaGetDeviceCount returns; built[0..2] = copyq / slotcache / routetable header present in the build.
   rc_load_sel gives a table of strlen(path) % 3 calls; g_host.init derives cap_slots = bytes / 17547264 / 32 * 32 and prints "K3GPU: host expert cache of %zu slots (stand-in)\n". */
void ecref_configure(int devices, const int built[3]);
void ecref_snapshot(ecref_snap *out);

#ifdef __cplusplus
}
#endif
#endif
