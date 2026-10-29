#ifndef SLOTCACHE_H
#define SLOTCACHE_H
/* Contract of the expert slot cache. Behaviour reference: struct SlotCache (lines 208-285) and the globals g_policy / g_nu_fn (declared in 200-206, with the unused g_chunks) in src/code/gpu/decode.cuh.
   Keys are >= 0 in practice. The original uses -1 as its "no victim" marker: in the belady branch a negative key is never chosen as a victim; in the heat/LRU branch it can be, and
   then *vkey = -1 is indistinguishable from "no victim". Both implementations do the same; callers must not use negative keys.
   Both the C++ reference (ref/) and the Rust crate implement exactly this header; neither may change it.

   What it is: a table key -> slot (an opaque byte pointer the cache never dereferences) with replacement. A cache owns no memory; the caller hands it slots with
   sc_add_spare() and gets them back through acquire / remove / release. Not thread-safe: the engine calls it from one thread only, the caller must keep doing so.
   Nothing here touches CUDA, files or threads.

   Process-wide settings (they steer every cache, as the two globals did):
     policy 0: replace the least recently used entry that is not in `prot`.
     policy != 0: replace the entry with the lowest heat among the first 48 entries not in `prot`, scanning from the least recently used end (heat ties keep the
                  earlier one in that scan).
     policy 2 AND a next-use function set (belady): replace the entry whose (next use, key) pair is the largest and whose key is not in `prot`. With policy 2 and no
                  function the policy-1 rule applies. The function may be unset at any time (the engine does when the route cache stops matching); entries keep the
                  next-use value they were stored with, and sc_remove / the heat-rule victim removal erase the pair {stored value, key} from the ordered set when that
                  set is not empty, exactly as the original does.
   Heat: an entry starts with heat0 given to sc_acquire; sc_touch in a non-belady state does heat = heat * 0.8f + w in single precision, in that operation order, unfused. The
   C++ original is unfused only because the engine's host compile of this code has no FMA target (nvcc -O3 --fmad=false, no -march); a build with -march=native/-mfma could
   fuse it, and the Rust library never does. NaN heat or w is not produced by the engine; the payload bits of a NaN result are unspecified.
   In belady state sc_touch instead replaces the entry's next-use value by next_use(key) and re-inserts it in the ordered set.

   Errors: when no spare exists and no victim can be chosen, sc_acquire writes "expert cache: every slot is in use by one call (too few slots)\n" to stderr and exits the
   process with status 1 (as the original). In belady state the candidates are ONLY the pairs in the ordered set (an entry stored while belady was off is not in it), so the
   exit can happen with unprotected entries held; in the heat/LRU rule the candidates are the entries not in `prot`. Acquiring a key that is already in the cache, or releasing/adding a slot twice, is outside the contract: not tested, behaviour free. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_cache sc_cache;                       /* opaque */
typedef int (*sc_next_use_fn)(int key);                 /* number of the next call after the current one that needs `key`, or a huge number; may have side effects (the engine's does) */

void sc_set_policy(int policy);                         /* default 1 */
void sc_set_next_use(sc_next_use_fn fn);                /* NULL = unset */

sc_cache *sc_new(void);
void sc_free(sc_cache *c);                              /* frees the cache's bookkeeping only, never a slot */

void sc_add_spare(sc_cache *c, unsigned char *slot);    /* spare.push_back: spares are used last-in first-out */
void sc_release(sc_cache *c, unsigned char *slot);      /* same operation as sc_add_spare, kept under the original's two names */
int sc_contains(const sc_cache *c, int key);            /* 1 if key is held (was map.count / map.find != end) */
size_t sc_size(const sc_cache *c);                      /* entries held */

unsigned char *sc_find(sc_cache *c, int key);           /* the slot, or NULL if absent; a hit becomes the most recently used */
void sc_touch(sc_cache *c, int key, float w);           /* no effect if key is absent */
int sc_worth(const sc_cache *c, int key);               /* 1 if a slot may be given to key: not belady, or a spare exists, or the ordered set is empty, or next_use(key) < largest stored next use */

/* A slot for `key`: a spare if there is one (the last added), else the victim's slot; the victim leaves the cache and *vkey / *vheat describe it (its bytes are still in
   the slot), else *vkey = -1 and *vheat is not written. `prot` is an array of nprot keys (membership test only, order irrelevant, duplicates allowed). key becomes the most
   recently used entry with heat heat0 and, in belady state, next-use value next_use(key) (inserted in the ordered set); otherwise the stored value is 0. */
unsigned char *sc_acquire(sc_cache *c, int key, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat);

/* Read-ahead: the original does not look at the policy state here, so this is the same in every state. As sc_acquire, but only if a spare exists or the held entry with the largest (next use, key) not in `prot` is needed later than cand_nu;
   otherwise returns NULL and changes nothing (*vkey = -1 is written in every case). With no spare and no entry in the ordered set it returns NULL. */
unsigned char *sc_try_acquire(sc_cache *c, int key, int cand_nu, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat);

unsigned char *sc_remove(sc_cache *c, int key);         /* takes key out, returns its slot (not made spare) or NULL if absent */

/* Observation for the verifier: the full state, so two implementations can be compared after every operation. Each function writes up to `cap` records and returns the
   TOTAL number that exist. */
typedef struct sc_entry { int key; unsigned char *slot; float heat; int nu; } sc_entry;
size_t sc_snapshot_lru(const sc_cache *c, sc_entry *out, size_t cap);          /* most recently used first */
size_t sc_snapshot_spare(const sc_cache *c, unsigned char **out, size_t cap);  /* in vector order: the LAST record is the one handed out next */
size_t sc_snapshot_next(const sc_cache *c, int *nu_out, int *key_out, size_t cap); /* the ordered set, smallest (nu, key) first */

#ifdef __cplusplus
}
#endif
#endif
