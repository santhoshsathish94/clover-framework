/* Smoke test for libroutetable.so (Rust): writes a small hand-written trace, loads it with experts_per_layer = 10 and checks accessors and next-use values.
   build: gcc -O1 -I include -o smoke_rust smoke_rust.c -L target/release -lroutetable -Wl,-rpath,$PWD/target/release
   run:   ./smoke_rust */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "routetable.h"

static int failures = 0;
static void check(int ok, const char *what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); if (!ok) failures++; }

int main(void)
{
    /* keys (layer*10+e): 5 -> calls 0,2,2 ; 6 -> 0,3 ; 7 -> 2 ; 15 -> 1. Call 4 is an empty line, call 5 has no trailing newline, call 3 has a CRLF end. */
    const char *trace = "0 3 5:2 6:1\n1 3 5:4\n0 3 5 5 7:\n0 3 6\r\n\n1 9 4:1";
    char path[] = "/tmp/routetable-smoke-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { puts("FAIL: mkstemp"); return 1; }
    if (write(fd, trace, strlen(trace)) != (ssize_t)strlen(trace)) { puts("FAIL: write"); return 1; }
    close(fd);

    rt_table *t = rt_load(path, 10);
    check(t != NULL, "rt_load returns a table");
    check(rt_calls(t) == 6, "6 calls (empty line and unterminated last line count)");
    check(rt_layer(t, 0) == 0 && rt_layer(t, 1) == 1 && rt_layer(t, 4) == 0 && rt_layer(t, 5) == 1, "layers 0 1 . . 0 1");

    size_t n = 7;
    const int *e = rt_experts(t, 0, &n);
    check(n == 2 && e && e[0] == 5 && e[1] == 6, "call 0 experts 5,6");
    const int *c = rt_counts(t, 0, &n);
    check(n == 2 && c && c[0] == 2 && c[1] == 1, "call 0 counts 2,1");
    e = rt_experts(t, 2, &n);
    check(n == 3 && e && e[0] == 5 && e[1] == 5 && e[2] == 7, "call 2 experts 5,5,7");
    c = rt_counts(t, 2, &n);
    check(n == 1 && c && c[0] == 0, "call 2 counts: colon without digits gives 0");
    rt_experts(t, 4, &n);
    check(n == 0, "empty line has no experts");
    rt_counts(t, 4, &n);
    check(n == 0, "empty line has no counts");

    check(rt_next_use(t, 5) == 0, "initial current -1: key 5 next use is call 0");
    rt_set_current(t, 0);
    check(rt_next_use(t, 5) == 2, "current 0: key 5 -> 2 (strictly greater)");
    check(rt_next_use(t, 6) == 3, "current 0: key 6 -> 3");
    check(rt_next_use(t, 7) == 2 && rt_next_use(t, 15) == 1 && rt_next_use(t, 14) == 5, "keys 7, 15 and 14 (layer 1 expert 4)");
    check(rt_next_use(t, 99) == 1000000000, "unknown key -> 1000000000");
    rt_set_current(t, -1);
    check(rt_next_use(t, 6) == 3, "pointer is forward-only: key 6 stays at 3 after current goes back to -1");
    rt_set_current(t, 2);
    check(rt_next_use(t, 5) == 1000000000 && rt_next_use(t, 7) == 1000000000, "current 2: keys 5 and 7 exhausted");
    rt_set_current(t, 3);
    check(rt_next_use(t, 6) == 1000000000, "current 3: key 6 exhausted");
    rt_free(t);
    unlink(path);

    puts(failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
