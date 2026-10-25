// Probe for the driver blind spot "set_policy only ever receives 0, 1 or 2". Usage: policy_probe LIB POLICY
// Prints a transcript; run on the reference and on a candidate and diff the two. Not part of the driver; it only uses the public contract in slotcache.h.
#include "slotcache.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nu(int key) { return 100 + key * 3 + (key & 1); }
static unsigned char *S(int n) { return (unsigned char *)(0x100000 + 64 * (uintptr_t)n); }

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    typedef void (*t_pol)(int); typedef void (*t_snu)(sc_next_use_fn); typedef sc_cache *(*t_new)(); typedef void (*t_add)(sc_cache *, unsigned char *);
    typedef unsigned char *(*t_acq)(sc_cache *, int, const int *, size_t, float, int *, float *); typedef int (*t_worth)(const sc_cache *, int); typedef void (*t_touch)(sc_cache *, int, float);
    t_pol set_policy_ = (t_pol)dlsym(h, "sc_set_policy"); t_snu set_next_use_ = (t_snu)dlsym(h, "sc_set_next_use"); t_new new_ = (t_new)dlsym(h, "sc_new");
    t_add add_spare_ = (t_add)dlsym(h, "sc_add_spare"); t_acq acquire_ = (t_acq)dlsym(h, "sc_acquire"); t_worth worth_ = (t_worth)dlsym(h, "sc_worth"); t_touch touch_ = (t_touch)dlsym(h, "sc_touch");
    set_policy_(atoi(argv[2])); set_next_use_(nu);
    sc_cache *c = new_();
    for (int i = 3; i >= 1; i--) add_spare_(c, S(i));
    int vk; float vh = 0;
    acquire_(c, 1, nullptr, 0, 5.0f, &vk, &vh); acquire_(c, 2, nullptr, 0, 1.0f, &vk, &vh); acquire_(c, 3, nullptr, 0, 9.0f, &vk, &vh);
    touch_(c, 1, 1.0f);
    unsigned char *p = acquire_(c, 9, nullptr, 0, 1.0f, &vk, &vh);
    printf("policy %s, function set: acquire(9) -> slot #%d victim key %d victim heat %.9g; worth(77)=%d\n", argv[2], (int)(((uintptr_t)p - 0x100000) / 64), vk, vh, worth_(c, 77));
    return 0;
}
