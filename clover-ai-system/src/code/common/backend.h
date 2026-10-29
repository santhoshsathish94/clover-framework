#ifndef CLOVER_BACKEND_H
#define CLOVER_BACKEND_H
/* The boundary between common/ (the driver and everything not tied to a kind of machine) and a backend (cpu/ today, gpu/ later).
   whole-model.c calls only what is declared here, and a backend defines it. The declarations were in whole-model.c before; they are written down once here.
   Known assumption: expert memory and cache slots are plain host pointers (unsigned char *). A backend whose weights live in device memory will need
   handles instead; that is decided when the machine has been read, not now. */
#include <stddef.h>
#include <stdint.h>

/* Memory for experts (the running layer's buffer, the cache slots, the scratch pool): writable, zero-filled, NULL if it cannot be had. */
unsigned char *backend_expert_memory(size_t bytes);

/* Stage 2: layer 0. */
double stage2_setup(const char *dataset, unsigned max_n);
void stage2_run(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
void stage2_run_from(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
void stage2_step(const uint32_t *id, const float *emb, float *out);
void stage2_stepk(const uint32_t *ids, const float *emb, unsigned n, float *out);
void stage2_keep_positions(unsigned count); void stage2_use_position(unsigned position);

/* Stage 3: layers 1-92, one set of functions per attention kind, named by the prefix P: lbk (KDA) and lbm (MLA). A handle is a layer opened with P##_open. */
#define LAYER_KIND(P) \
    void P##_laps(double *out); \
    void P##_guess_stats(unsigned long *issued, unsigned long *used); \
    void P##_set_slots(void *handle, unsigned char *const *slots, unsigned cap); void P##_cache_stats(unsigned long *hit, unsigned long *miss); \
    void P##_route_log(int on); void P##_routes(unsigned short *out); void P##_hits(unsigned long *out); \
    void P##_set_buffer(unsigned char *buffer); double P##_init(unsigned max_n); \
    void *P##_open(const char *dataset, unsigned layer, unsigned k, unsigned max_n); \
    double P##_run(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, float *out); \
    double P##_run_from(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, size_t pos0, float *out); \
    double P##_step(void *handle, const float *resid, const float *const *snaps, size_t pos0, float *out); \
    double P##_stepk(void *handle, const float *resid, const float *const *snaps, unsigned n, size_t pos0, float *out); \
    void P##_set_scratch(unsigned char *const *slots, unsigned n); unsigned long P##_distinct(void); \
    void P##_keep_positions(void *handle, unsigned count); void P##_use_position(void *handle, unsigned position);
LAYER_KIND(lbk)
LAYER_KIND(lbm)

/* The DSpark draft model: it reads the target's hidden states after layers 2, 23, 47, 71 and 89, one row of 7168 floats per position. */
int dspark_open(const char *checkpoint, const char *head_table, unsigned max_positions);
void dspark_draft(const float *const aux[5], uint32_t anchor, unsigned anchor_position, uint32_t *drafts);
void dspark_reset(void);

#endif
