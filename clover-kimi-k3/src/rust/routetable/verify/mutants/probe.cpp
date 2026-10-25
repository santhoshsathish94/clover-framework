// Probe: loads one library, loads FILE with EXPERTS, prints the table and the results of a scripted operation list.
//   probe LIB FILE EXPERTS [c<cur> | n<key>]...      (c = rt_set_current, n = rt_next_use)
// Own tool of the mutant campaign; independent of diff_driver.cpp. Build: g++ -O2 -std=c++17 -I../../../include -o probe probe.cpp -ldl
#include "routetable.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto load = (rt_table * (*)(const char *, int)) dlsym(h, "rt_load");
    auto fr = (void (*)(rt_table *))dlsym(h, "rt_free");
    auto ncalls = (size_t(*)(const rt_table *))dlsym(h, "rt_calls");
    auto layer = (int (*)(const rt_table *, size_t))dlsym(h, "rt_layer");
    auto experts = (const int *(*)(const rt_table *, size_t, size_t *))dlsym(h, "rt_experts");
    auto counts = (const int *(*)(const rt_table *, size_t, size_t *))dlsym(h, "rt_counts");
    auto setcur = (void (*)(rt_table *, long))dlsym(h, "rt_set_current");
    auto nu = (int (*)(rt_table *, int))dlsym(h, "rt_next_use");
    rt_table *t = load(argv[2], atoi(argv[3]));
    const size_t n = ncalls(t);
    printf("calls=%zu\n", n);
    for (size_t i = 0; i < n; i++) {
        size_t ne = 0, nc = 0; const int *e = experts(t, i, &ne), *c = counts(t, i, &nc);
        printf("call %zu layer=%d experts[%zu]=", i, layer(t, i), ne);
        for (size_t k = 0; k < ne; k++) printf("%d ", e[k]);
        printf("counts[%zu]=", nc);
        for (size_t k = 0; k < nc; k++) printf("%d ", c[k]);
        printf("\n");
    }
    for (int i = 4; i < argc; i++) {
        if (argv[i][0] == 'c') { setcur(t, atol(argv[i] + 1)); printf("set_current(%ld)\n", atol(argv[i] + 1)); }
        else if (argv[i][0] == 'n') printf("next_use(%d)=%d\n", atoi(argv[i] + 1), nu(t, atoi(argv[i] + 1)));
    }
    fr(t);
    return 0;
}
