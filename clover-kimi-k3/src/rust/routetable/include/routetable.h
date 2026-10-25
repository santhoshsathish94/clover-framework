#ifndef ROUTETABLE_H
#define ROUTETABLE_H
/* Contract of the route table: the parsed K3GPU_ROUTECACHE file and the next-use index built from it.
   Behaviour reference: struct CallSet (52), the globals g_rc / g_uses / g_use_ptr / g_rc_cur (53-57), rc_nu (59-66) and rc_load (the function that reads the file and builds
   g_uses) in src/code/gpu/backend-gpu.cu. Both the C++ reference (ref/) and the Rust crate implement exactly this header; neither may change it.

   File format (one call per line, as written by K3GPU_TRACE): `layer ntok e[:count] e[:count] ...`. Parsing follows the original line by line: a line is read with getline;
   the layer is strtol(base 10) of the line start; the second number (ntok) is read with strtol and ignored; then repeatedly: skip spaces; stop at the first character that is
   not an ASCII digit; read an expert with strtol; if the next character is ':' read a count with strtol after it (a colon with no digits after it records a count of 0 and
   consumes the colon; strtol itself accepts leading whitespace and a sign, so `3: -4` records -4). Between numbers only the space character is skipped, never a tab. EVERY line, including an empty one or one that ends early,
   becomes one call (an empty line becomes layer 0 with no experts). Counts may be fewer than experts or absent; they are kept in the order read, independently of the experts.
   Numbers are C ints (strtol results cast to int, as the original). A line without a trailing newline at the end of the file counts as a line.

   Next use: the index maps key = layer * experts_per_layer + expert (experts_per_layer is the `experts` argument of rt_load) to the ascending list of numbers of the calls that
   contain the expert (a call that lists the expert twice contributes its number twice). rt_next_use(key) returns, for the current call number `cur`, the first number in the
   key's list that is greater than cur, or 1000000000 if there is none or the key is unknown. Each key keeps a pointer into its list that only ever moves forward, exactly as
   the original: a later rt_set_current with a smaller value does not move it back. Calling rt_next_use for an unknown key changes nothing.
   Initial current call number: -1. Not thread-safe (one caller thread, as the original). At most one table is live per process (the original has one set of globals): rt_load
   after an earlier rt_load without rt_free is outside the contract. Also outside the contract (the Rust library may abort or crash, the C++ reference may not): a NULL table
   or path, a table used after rt_free, rt_free twice, an index i >= rt_calls, a NULL n. rt_experts / rt_counts may return NULL when *n is 0. The caller must not free the table
   while anything may still read it (the engine reads it from an atexit handler, so it never frees it). */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct rt_table rt_table;

/* Reads the file. If it cannot be opened: writes "cannot read route cache <path>\n" to stderr and exits the process with status 1 (as the original). */
rt_table *rt_load(const char *path, int experts);
void rt_free(rt_table *t);

size_t rt_calls(const rt_table *t);
int rt_layer(const rt_table *t, size_t i);                          /* i < rt_calls */
/* The experts / counts of call i. *n receives the length; the pointer stays valid until rt_free and is never NULL when *n > 0 (it may be NULL when *n == 0). */
const int *rt_experts(const rt_table *t, size_t i, size_t *n);
const int *rt_counts(const rt_table *t, size_t i, size_t *n);

void rt_set_current(rt_table *t, long cur);                         /* g_rc_cur = cur */
int rt_next_use(rt_table *t, int key);                              /* rc_nu(key) */

#ifdef __cplusplus
}
#endif
#endif
