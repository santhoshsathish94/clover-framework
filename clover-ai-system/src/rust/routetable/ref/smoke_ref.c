/* Smoke test of the route-table contract against libroutetable_ref.so. Expected values were derived by hand from the contract text in ../include/routetable.h
   (derivations in the comments); they were not taken from running the library. Build: see build.sh. Run: ./smoke_ref */
#include "../include/routetable.h"
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks = 0, fails = 0;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)
#define NONE 1000000000

static void call_is(const rt_table *t, size_t i, int layer, size_t ne, const int *ev, size_t nc, const int *cv, int line)
{
    size_t n1 = 99, n2 = 99;
    const int *e = rt_experts(t, i, &n1), *c = rt_counts(t, i, &n2);
    checks++;
    int ok = rt_layer(t, i) == layer && n1 == ne && n2 == nc && (ne == 0 || (e && memcmp(e, ev, ne * sizeof(int)) == 0)) && (nc == 0 || (c && memcmp(c, cv, nc * sizeof(int)) == 0));
    if (!ok) { fails++; printf("FAIL line %d: call %zu (layer %d, experts %zu, counts %zu)\n", line, i, rt_layer(t, i), n1, n2); }
}
#define CALL_IS(t, i, layer, ne, ev, nc, cv) call_is(t, i, layer, ne, ev, nc, cv, __LINE__)

int main(void)
{
    char path[64];
    snprintf(path, sizeof path, "/tmp/rt_smoke_trace_%d.txt", (int)getpid());

    /* ---- trace file (experts per layer = 896; calls numbered from 0 in file order; the ntok column is ignored):
       0: "1 4 10:3 20:5 30:2"   layer 1, experts 10 20 30, counts 3 5 2
       1: "2 4 10 20 30"         layer 2, experts 10 20 30, no counts
       2: "1 4 20:1 30:7 40"     layer 1, experts 20 30 40, counts 1 7 (fewer counts than experts)
       3: ""                     empty line: layer 0, nothing
       4: "1 4 10:9 20 "         layer 1, experts 10 20, counts 9 (line ends at a space)
       5: "2 6 3 5 7 x 9"        layer 2, experts 3 5 7, then 'x' stops the line (9 is not read)
       6: "1 4 30:4 10 30"       layer 1, experts 30 10 30, counts 4; last line, no trailing newline; expert 30 twice. */
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) return 1;
    fputs("1 4 10:3 20:5 30:2\n" "2 4 10 20 30\n" "1 4 20:1 30:7 40\n" "\n" "1 4 10:9 20 \n" "2 6 3 5 7 x 9\n" "1 4 30:4 10 30", f);
    fclose(f);

    rt_table *t = rt_load(path, 896);
    CHECK(t != NULL);
    CHECK(rt_calls(t) == 7);
    { int e[] = {10, 20, 30}, c[] = {3, 5, 2};  CALL_IS(t, 0, 1, 3, e, 3, c); }
    { int e[] = {10, 20, 30};                   CALL_IS(t, 1, 2, 3, e, 0, NULL); }
    { int e[] = {20, 30, 40}, c[] = {1, 7};     CALL_IS(t, 2, 1, 3, e, 2, c); }
    CALL_IS(t, 3, 0, 0, NULL, 0, NULL);
    { int e[] = {10, 20}, c[] = {9};            CALL_IS(t, 4, 1, 2, e, 1, c); }
    { int e[] = {3, 5, 7};                      CALL_IS(t, 5, 2, 3, e, 0, NULL); }
    { int e[] = {30, 10, 30}, c[] = {4};        CALL_IS(t, 6, 1, 3, e, 1, c); }

    /* ---- next use. key = layer * 896 + expert; lists of call numbers per key:
       906 (L1 e10): 0 4 6      916 (L1 e20): 0 2 4      926 (L1 e30): 0 2 6 6      936 (L1 e40): 2
       1802 / 1812 / 1822 (L2 e10/e20/e30): 1      1795 / 1797 / 1799 (L2 e3/e5/e7): 5      1801 (L2 e9, after the 'x'): unknown.
       Each key's pointer moves forward past entries <= cur and never back.
       cur starts at -1:   nu(906) = 0 (0 > -1);  nu(1802) = 1;  unknown 99999 and 5 -> NONE.
       cur 0:   906: 0 <= 0 skip -> 4;  1802 -> 1;  916 (first query): skip 0 -> 2.
       cur 4:   906: 4 <= 4 skip -> 6;  916: skip 2, 4 -> NONE;  936: skip 2 -> NONE;  1802: skip 1 -> NONE.
       cur 1 (decreasing): 906 stays at its pointer -> 6 (a fresh scan would say 4);  916 -> NONE (fresh would say 2);  926 (first query): skip 0 -> 2;  1822 (first): skip 1 -> NONE.
       cur -1: 926 stays -> 2 (fresh would say 0);  1812 (first query) -> 1.
       cur 5:  926: skip 2 -> 6;  1795 (first): skip 5 -> NONE.
       cur 6:  926: skip 6 and the second 6 -> NONE.
       cur -1: 926 stays at the end -> NONE;  1797 (first query) -> 5;  906 -> 6;  unknown 1801 -> NONE throughout. */
    CHECK(rt_next_use(t, 906) == 0);
    CHECK(rt_next_use(t, 1802) == 1);
    CHECK(rt_next_use(t, 99999) == NONE);
    CHECK(rt_next_use(t, 5) == NONE);
    rt_set_current(t, 0);
    CHECK(rt_next_use(t, 906) == 4);
    CHECK(rt_next_use(t, 1802) == 1);
    CHECK(rt_next_use(t, 916) == 2);
    rt_set_current(t, 4);
    CHECK(rt_next_use(t, 906) == 6);
    CHECK(rt_next_use(t, 916) == NONE);
    CHECK(rt_next_use(t, 936) == NONE);
    CHECK(rt_next_use(t, 1802) == NONE);
    rt_set_current(t, 1);
    CHECK(rt_next_use(t, 906) == 6);
    CHECK(rt_next_use(t, 916) == NONE);
    CHECK(rt_next_use(t, 926) == 2);
    CHECK(rt_next_use(t, 1822) == NONE);
    rt_set_current(t, -1);
    CHECK(rt_next_use(t, 926) == 2);
    CHECK(rt_next_use(t, 1812) == 1);
    rt_set_current(t, 5);
    CHECK(rt_next_use(t, 926) == 6);
    CHECK(rt_next_use(t, 1795) == NONE);
    rt_set_current(t, 6);
    CHECK(rt_next_use(t, 926) == NONE);
    rt_set_current(t, -1);
    CHECK(rt_next_use(t, 926) == NONE);
    CHECK(rt_next_use(t, 1797) == 5);
    CHECK(rt_next_use(t, 906) == 6);
    CHECK(rt_next_use(t, 1801) == NONE);

    /* ---- free, load again: current call number is -1 again and every pointer is fresh, so 906 -> 0 and 926 -> 0 (not NONE as at the end above). */
    rt_free(t);
    t = rt_load(path, 896);
    CHECK(rt_calls(t) == 7);
    CHECK(rt_next_use(t, 906) == 0);
    CHECK(rt_next_use(t, 926) == 0);
    rt_free(t);

    /* ---- a different experts-per-layer changes the keys: with 100, key 1*100+10 = 110 is L1 e10 (calls 0 4 6) and 906 is unknown. */
    t = rt_load(path, 100);
    CHECK(rt_next_use(t, 110) == 0);
    CHECK(rt_next_use(t, 906) == NONE);
    rt_free(t);
    unlink(path);

    /* ---- missing file: message on stderr, exit status 1 (checked in a child process). */
    {
        int pf[2];
        CHECK(pipe(pf) == 0);
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            dup2(pf[1], 2); close(pf[0]); close(pf[1]);
            rt_load("/tmp/rt_smoke_no_such_file.trace", 896);
            _exit(99);
        }
        close(pf[1]);
        char msg[256] = {0};
        ssize_t got = read(pf[0], msg, sizeof msg - 1);
        close(pf[0]);
        int st = 0;
        CHECK(waitpid(pid, &st, 0) == pid);
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        CHECK(got > 0 && strcmp(msg, "cannot read route cache /tmp/rt_smoke_no_such_file.trace\n") == 0);
    }

    printf("%s (%d checks, %d failed)\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
