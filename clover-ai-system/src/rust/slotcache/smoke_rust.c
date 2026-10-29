/* Smoke test for libslotcache.so (Rust). 3 slots, policy 0: acquire keys 1,2,3, find 1, acquire 4, the evicted key must be 2 with slot B. Run twice, once per spare order
   (spares are last-in first-out, so the order decides which slot key 1 gets; slot B goes to key 2 in both).
   build: gcc -O1 -I include -o smoke_rust smoke_rust.c -L target/release -lslotcache -Wl,-rpath,$PWD/target/release
   run:   ./smoke_rust */
#include <stdio.h>
#include <string.h>
#include "slotcache.h"

static int failures = 0;
static void check(int ok, const char *what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); if (!ok) failures++; }

static void scenario(int add_c_first)
{
    unsigned char A[16], B[16], C[16];
    printf("--- spare order: %s\n", add_c_first ? "C,B,A (key1->A key2->B key3->C)" : "A,B,C (key1->C key2->B key3->A)");
    unsigned char *k1 = add_c_first ? A : C, *k3 = add_c_first ? C : A;
    sc_cache *c = sc_new();
    if (add_c_first) { sc_add_spare(c, C); sc_add_spare(c, B); sc_add_spare(c, A); }
    else { sc_add_spare(c, A); sc_add_spare(c, B); sc_add_spare(c, C); }
    int vkey = 123; float vheat = -1.0f;
    check(sc_acquire(c, 1, NULL, 0, 1.0f, &vkey, &vheat) == k1 && vkey == -1, "acquire 1 takes a spare, no victim");
    check(sc_acquire(c, 2, NULL, 0, 1.0f, &vkey, &vheat) == B && vkey == -1, "acquire 2 takes slot B");
    check(sc_acquire(c, 3, NULL, 0, 1.0f, &vkey, &vheat) == k3 && vkey == -1, "acquire 3 takes a spare");
    check(sc_size(c) == 3 && sc_contains(c, 1) && sc_contains(c, 2) && sc_contains(c, 3), "3 entries held");
    check(sc_find(c, 1) == k1, "find 1 returns its slot");
    vkey = 123; vheat = -1.0f;
    unsigned char *p = sc_acquire(c, 4, NULL, 0, 1.0f, &vkey, &vheat);
    check(vkey == 2 && p == B, "acquire 4 evicts key 2 with slot B");
    check(vheat == 1.0f, "victim heat reported");
    check(!sc_contains(c, 2) && sc_contains(c, 4) && sc_size(c) == 3, "key 2 gone, key 4 held, size 3");
    sc_entry e[4];
    size_t n = sc_snapshot_lru(c, e, 4);
    check(n == 3 && e[0].key == 4 && e[1].key == 1 && e[2].key == 3, "lru order 4,1,3");
    sc_free(c);
}

int main(void)
{
    sc_set_policy(0);
    sc_set_next_use(NULL);
    check(sizeof(sc_entry) == 24 && offsetof(sc_entry, slot) == 8 && offsetof(sc_entry, heat) == 16 && offsetof(sc_entry, nu) == 20, "sc_entry is 24 bytes, slot at 8, heat at 16, nu at 20 (LP64)");
    scenario(1);
    scenario(0);
    puts(failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
