/* Smoke test of the slotcache contract against libslotcache_ref.so. Expected values were derived by hand from the contract text in ../include/slotcache.h
   (derivations in the comments); they were not taken from running the library. Build: see build.sh. Run: ./smoke_ref */
#include "../include/slotcache.h"
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks = 0, fails = 0;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

static void lru_is(const sc_cache *c, size_t n, const int *keys, unsigned char *const *slots, const float *heat, const int *nu, int line)
{
    sc_entry e[8];
    size_t tot = sc_snapshot_lru(c, e, 8);
    checks++;
    int ok = (tot == n);
    for (size_t i = 0; ok && i < n; i++) ok = e[i].key == keys[i] && e[i].slot == slots[i] && e[i].heat == heat[i] && e[i].nu == nu[i];
    if (!ok) { fails++; printf("FAIL line %d: lru snapshot (total %zu, expected %zu)\n", line, tot, n); }
}
#define LRU_IS(c, n, keys, slots, heat, nu) lru_is(c, n, keys, slots, heat, nu, __LINE__)

static int nu_val[16];
static int nu_fn(int key) { return nu_val[key]; }

int main(void)
{
    int vk; float vh;

    /* ---- 1. policy 0, three slots.
       spares added b0,b1,b2 are handed out last-in first-out: b2, b1, b0.
       acquire 10 -> b2, 20 -> b1, 30 -> b0; recency (most recent first) 30,20,10.
       find(10) makes 10 the most recent: 10,30,20.  acquire 40 (no spare): policy 0 replaces the least recent = 20 (slot b1, heat 1.0).  Now 40,10,30.
       acquire 50 with 30 protected: the least recent (30) is protected, so the next one, 10 (slot b2), goes.  Now 50,40,30.
       remove(40) returns b1 (not made spare); absent key -> NULL.  release(b1) then acquire 60 takes b1 as spare, vkey = -1. */
    unsigned char b[3];
    sc_set_policy(0); sc_set_next_use(NULL);
    sc_cache *c = sc_new();
    CHECK(sc_size(c) == 0);
    sc_add_spare(c, &b[0]); sc_add_spare(c, &b[1]); sc_release(c, &b[2]);
    unsigned char *sp[4];
    CHECK(sc_snapshot_spare(c, sp, 4) == 3 && sp[0] == &b[0] && sp[1] == &b[1] && sp[2] == &b[2]);
    vk = 7; vh = -1.0f;
    CHECK(sc_acquire(c, 10, NULL, 0, 1.0f, &vk, &vh) == &b[2] && vk == -1 && vh == -1.0f);
    CHECK(sc_acquire(c, 20, NULL, 0, 1.0f, &vk, &vh) == &b[1] && vk == -1);
    CHECK(sc_acquire(c, 30, NULL, 0, 1.0f, &vk, &vh) == &b[0] && vk == -1);
    CHECK(sc_size(c) == 3 && sc_contains(c, 10) == 1 && sc_contains(c, 99) == 0);
    { int k[3] = {30, 20, 10}; unsigned char *s[3] = {&b[0], &b[1], &b[2]}; float h[3] = {1, 1, 1}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    CHECK(sc_find(c, 10) == &b[2]);
    CHECK(sc_find(c, 99) == NULL);
    { int k[3] = {10, 30, 20}; unsigned char *s[3] = {&b[2], &b[0], &b[1]}; float h[3] = {1, 1, 1}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    vk = 0; vh = 0.0f;
    CHECK(sc_acquire(c, 40, NULL, 0, 1.0f, &vk, &vh) == &b[1] && vk == 20 && vh == 1.0f);
    CHECK(sc_contains(c, 20) == 0 && sc_contains(c, 40) == 1 && sc_size(c) == 3);
    { int k[3] = {40, 10, 30}; unsigned char *s[3] = {&b[1], &b[2], &b[0]}; float h[3] = {1, 1, 1}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    { int prot[1] = {30};
      CHECK(sc_acquire(c, 50, prot, 1, 1.0f, &vk, &vh) == &b[2] && vk == 10 && vh == 1.0f); }
    { int k[3] = {50, 40, 30}; unsigned char *s[3] = {&b[2], &b[1], &b[0]}; float h[3] = {1, 1, 1}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    CHECK(sc_remove(c, 40) == &b[1] && sc_size(c) == 2 && sc_contains(c, 40) == 0);
    CHECK(sc_remove(c, 40) == NULL);
    CHECK(sc_snapshot_spare(c, sp, 4) == 0);
    sc_release(c, &b[1]);
    CHECK(sc_snapshot_spare(c, sp, 4) == 1 && sp[0] == &b[1]);
    CHECK(sc_acquire(c, 60, NULL, 0, 1.0f, &vk, &vh) == &b[1] && vk == -1);
    CHECK(sc_size(c) == 3);
    { int k[3] = {60, 50, 30}; unsigned char *s[3] = {&b[1], &b[2], &b[0]}; float h[3] = {1, 1, 1}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    sc_free(c);

    /* ---- 2. policy 1 (lowest heat among the 48 least recent not in prot; ties keep the earlier one in the scan from the least recent end).
       Spares d0,d1,d2: key 1 -> d2, key 2 -> d1, key 3 -> d0, all heat0 5.0.  Recency 3,2,1.
       touch(1,3): 5*0.8f = 4.00000006 rounds to 4.0f (float step near 4 is 4.8e-7), +3 = 7.0.  touch(3,0): 4.0.  touch of an absent key: no effect.
       acquire 4: scan 1(7.0), 2(5.0), 3(4.0) -> victim 3 (the most recent!), vheat 4.0, slot d0.  Recency 4,2,1; heats 4:5.0 2:5.0 1:7.0.
       acquire 5 (heat0 6.0): scan 1(7.0), 2(5.0), 4(5.0, not lower) -> victim 2, vheat 5.0, slot d1.  Recency 5,4,1.
       acquire 6 with 4 protected: scan 1(7.0), skip 4, 5(6.0) -> victim 5, vheat 6.0, slot d1 (inherited from 2). Recency 6,4,1. */
    unsigned char d[3];
    sc_set_policy(1);
    c = sc_new();
    sc_add_spare(c, &d[0]); sc_add_spare(c, &d[1]); sc_add_spare(c, &d[2]);
    CHECK(sc_acquire(c, 1, NULL, 0, 5.0f, &vk, &vh) == &d[2]);
    CHECK(sc_acquire(c, 2, NULL, 0, 5.0f, &vk, &vh) == &d[1]);
    CHECK(sc_acquire(c, 3, NULL, 0, 5.0f, &vk, &vh) == &d[0]);
    sc_touch(c, 1, 3.0f); sc_touch(c, 3, 0.0f); sc_touch(c, 77, 9.0f);
    CHECK(sc_size(c) == 3);
    { int k[3] = {3, 2, 1}; unsigned char *s[3] = {&d[0], &d[1], &d[2]}; float h[3] = {4.0f, 5.0f, 7.0f}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    CHECK(sc_acquire(c, 4, NULL, 0, 5.0f, &vk, &vh) == &d[0] && vk == 3 && vh == 4.0f);
    CHECK(sc_acquire(c, 5, NULL, 0, 6.0f, &vk, &vh) == &d[1] && vk == 2 && vh == 5.0f);
    { int prot[1] = {4};
      CHECK(sc_acquire(c, 6, prot, 1, 5.0f, &vk, &vh) == &d[1] && vk == 5 && vh == 6.0f); }
    { int k[3] = {6, 4, 1}; unsigned char *s[3] = {&d[1], &d[0], &d[2]}; float h[3] = {5.0f, 5.0f, 7.0f}; int n[3] = {0, 0, 0}; LRU_IS(c, 3, k, s, h, n); }
    sc_free(c);

    /* ---- 3. policy 2 with a next-use function (belady). next use: 1->100 2->50 3->200 4->70 5->300 6->10 7->5.
       acquire 1,2,3 into e2,e1,e0: ordered set (50,2) (100,1) (200,3); recency 3,2,1.  worth(4): 70 < 200 -> 1;  worth(5): 300 < 200 false -> 0.
       acquire 4: victim = largest (nu,key) = (200,3), slot e0, vheat 1.5; key 4 stored with nu 70: set (50,2) (70,4) (100,1).
       acquire 6 with 1 protected: largest (100,1) is protected, next (70,4) -> victim 4 (slot e0); set (10,6) (50,2) (100,1).
       try_acquire(7, cand_nu 60, prot {1}): skip (100,1); (50,2) is not needed later than 60 -> NULL, *vkey = -1, nothing changes.
       try_acquire(7, cand_nu 40, prot {1}): (50,2) is needed later than 40 -> victim 2 (slot e1); set (5,7) (10,6) (100,1); recency 7,6,1.
       nu(6) changes to 90, touch(6): stored value follows: set (5,7) (90,6) (100,1).
       Unset the function (policy 2 stays): policy-1 rule. All heats 1.5 (tie) -> earliest in the scan from the least recent end = key 1 (slot e2); its pair (100,1)
       leaves the set although the set is not used for the new key: set (5,7) (90,6); the new key 8 has stored value 0. */
    unsigned char e[3];
    nu_val[1] = 100; nu_val[2] = 50; nu_val[3] = 200; nu_val[4] = 70; nu_val[5] = 300; nu_val[6] = 10; nu_val[7] = 5;
    sc_set_policy(2); sc_set_next_use(nu_fn);
    c = sc_new();
    CHECK(sc_worth(c, 1) == 1);
    sc_add_spare(c, &e[0]); sc_add_spare(c, &e[1]); sc_add_spare(c, &e[2]);
    CHECK(sc_worth(c, 4) == 1);
    CHECK(sc_acquire(c, 1, NULL, 0, 1.5f, &vk, &vh) == &e[2]);
    CHECK(sc_acquire(c, 2, NULL, 0, 1.5f, &vk, &vh) == &e[1]);
    CHECK(sc_acquire(c, 3, NULL, 0, 1.5f, &vk, &vh) == &e[0]);
    { int nu[4], key[4]; CHECK(sc_snapshot_next(c, nu, key, 4) == 3 && nu[0] == 50 && key[0] == 2 && nu[1] == 100 && key[1] == 1 && nu[2] == 200 && key[2] == 3); }
    { int k[3] = {3, 2, 1}; unsigned char *s[3] = {&e[0], &e[1], &e[2]}; float h[3] = {1.5f, 1.5f, 1.5f}; int n[3] = {200, 50, 100}; LRU_IS(c, 3, k, s, h, n); }
    CHECK(sc_worth(c, 4) == 1);
    CHECK(sc_worth(c, 5) == 0);
    CHECK(sc_acquire(c, 4, NULL, 0, 1.5f, &vk, &vh) == &e[0] && vk == 3 && vh == 1.5f);
    { int prot[1] = {1};
      CHECK(sc_acquire(c, 6, prot, 1, 1.5f, &vk, &vh) == &e[0] && vk == 4 && vh == 1.5f);
      vk = 123;
      CHECK(sc_try_acquire(c, 7, 60, prot, 1, 1.5f, &vk, &vh) == NULL && vk == -1 && sc_size(c) == 3 && sc_contains(c, 2) == 1);
      vk = 123;
      CHECK(sc_try_acquire(c, 7, 40, prot, 1, 1.5f, &vk, &vh) == &e[1] && vk == 2 && vh == 1.5f); }
    { int nu[4], key[4]; CHECK(sc_snapshot_next(c, nu, key, 4) == 3 && nu[0] == 5 && key[0] == 7 && nu[1] == 10 && key[1] == 6 && nu[2] == 100 && key[2] == 1); }
    nu_val[6] = 90; sc_touch(c, 6, 0.0f);
    { int nu[4], key[4]; CHECK(sc_snapshot_next(c, nu, key, 4) == 3 && nu[0] == 5 && key[0] == 7 && nu[1] == 90 && key[1] == 6 && nu[2] == 100 && key[2] == 1); }
    { int k[3] = {7, 6, 1}; unsigned char *s[3] = {&e[1], &e[0], &e[2]}; float h[3] = {1.5f, 1.5f, 1.5f}; int n[3] = {5, 90, 100}; LRU_IS(c, 3, k, s, h, n); }
    sc_set_next_use(NULL);
    CHECK(sc_worth(c, 5) == 1);
    CHECK(sc_acquire(c, 8, NULL, 0, 1.5f, &vk, &vh) == &e[2] && vk == 1 && vh == 1.5f);
    { int nu[4], key[4]; CHECK(sc_snapshot_next(c, nu, key, 4) == 2 && nu[0] == 5 && key[0] == 7 && nu[1] == 90 && key[1] == 6); }
    { int k[3] = {8, 7, 6}; unsigned char *s[3] = {&e[2], &e[1], &e[0]}; float h[3] = {1.5f, 1.5f, 1.5f}; int n[3] = {0, 5, 90}; LRU_IS(c, 3, k, s, h, n); }
    sc_free(c);

    /* ---- 4. error path: one slot, key 1 held and protected, acquire 2 -> message on stderr, exit status 1 (checked in a child process). */
    {
        int pf[2];
        CHECK(pipe(pf) == 0);
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            unsigned char x; int prot[1] = {1};
            dup2(pf[1], 2); close(pf[0]); close(pf[1]);
            sc_set_policy(0); sc_set_next_use(NULL);
            sc_cache *k = sc_new();
            sc_add_spare(k, &x);
            sc_acquire(k, 1, NULL, 0, 1.0f, &vk, &vh);
            sc_acquire(k, 2, prot, 1, 1.0f, &vk, &vh);
            _exit(99);
        }
        close(pf[1]);
        char msg[256] = {0};
        ssize_t got = read(pf[0], msg, sizeof msg - 1);
        close(pf[0]);
        int st = 0;
        CHECK(waitpid(pid, &st, 0) == pid);
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        CHECK(got > 0 && strcmp(msg, "expert cache: every slot is in use by one call (too few slots)\n") == 0);
    }

    printf("%s (%d checks, %d failed)\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
