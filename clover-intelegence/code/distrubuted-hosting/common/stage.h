#ifndef CLOVER_STAGE_H
#define CLOVER_STAGE_H
/* The stage number is the route. Given a stage, both the service that owns it and the
   service the object goes to next are arithmetic, so no stage needs a routing table and
   no frame needs a destination field.

   Layout, one stride per service:
     1 .. 120        server, trunk 0 on the way in
     121 .. 11160    layers 1..92, 120 apiece
     11161 .. 11280  server tail: final normalisation, head, next token

   A stage number recurs on every position. It says where in the process an object is,
   never when, so stage 5521 is always "layer 46, first step" whatever the prompt. */

enum {
    CLOVER_STAGE_STRIDE      = 120,
    CLOVER_STAGE_LAYERS      = 92,
    CLOVER_STAGE_SERVER_BASE = 0,
    CLOVER_STAGE_LAYER_BASE  = CLOVER_STAGE_STRIDE,
    CLOVER_STAGE_TAIL_BASE   = CLOVER_STAGE_LAYER_BASE + CLOVER_STAGE_LAYERS * CLOVER_STAGE_STRIDE,
    CLOVER_STAGE_LAST        = CLOVER_STAGE_TAIL_BASE + CLOVER_STAGE_STRIDE
};

/* 0 is the server, 1..92 a layer, 93 the tail. -1 is out of range. */
static inline int clover_stage_owner(int stage)
{
    if (stage < 1 || stage > CLOVER_STAGE_LAST) return -1;
    if (stage <= CLOVER_STAGE_LAYER_BASE) return 0;
    if (stage > CLOVER_STAGE_TAIL_BASE) return CLOVER_STAGE_LAYERS + 1;
    return (stage - CLOVER_STAGE_LAYER_BASE - 1) / CLOVER_STAGE_STRIDE + 1;
}

/* Position within its owner, 1..120. */
static inline int clover_stage_local(int stage)
{
    if (stage < 1 || stage > CLOVER_STAGE_LAST) return 0;
    return (stage - 1) % CLOVER_STAGE_STRIDE + 1;
}

static inline int clover_stage_first(int owner)
{
    if (owner == 0) return CLOVER_STAGE_SERVER_BASE + 1;
    if (owner > CLOVER_STAGE_LAYERS) return CLOVER_STAGE_TAIL_BASE + 1;
    return CLOVER_STAGE_LAYER_BASE + (owner - 1) * CLOVER_STAGE_STRIDE + 1;
}

/* The only forward path. Crossing a stride boundary hands the object to the next
   service, which is why no stage carries a destination. The tail is the one fork:
   another token restarts at the server, otherwise the object is finished. */
static inline int clover_stage_next(int stage, int more_tokens)
{
    if (stage < 1 || stage > CLOVER_STAGE_LAST) return 0;
    if (stage == CLOVER_STAGE_LAST) return more_tokens ? clover_stage_first(0) : 0;
    return stage + 1;
}

/* True when the next stage belongs to a different service, so a transport knows this is
   where a frame has to be sent rather than looped locally. */
static inline int clover_stage_hands_over(int stage)
{
    int next = clover_stage_next(stage, 1);
    return next && clover_stage_owner(next) != clover_stage_owner(stage);
}

#endif
