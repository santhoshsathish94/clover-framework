#ifndef CLOVER_FRAME_H
#define CLOVER_FRAME_H
/* The object that moves between services. It carries its own stage, so a service reads
   one int to know whether the frame is its work, what step to run, and where it goes
   next. Nothing else needs a routing table and no header needs a destination.

   The layout is fixed and self-describing so it can be written to a file, a pipe or a
   socket and read back without parsing: one fwrite, one fread. */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "stage.h"

enum {
    CLOVER_FRAME_MAGIC     = 0x434C5646,   /* "CLVF" */
    CLOVER_FRAME_WIDTH     = 7168,
    CLOVER_FRAME_SNAPSHOTS = 8
};

typedef struct {
    uint32_t magic;
    int32_t  stage;                 /* owner and next are arithmetic on this */
    uint32_t session;               /* which conversation the state belongs to */
    uint32_t position;              /* which token within it */
    uint32_t snapshot_count;
    uint32_t reserved;
    float    residual[CLOVER_FRAME_WIDTH];
    float    snapshots[CLOVER_FRAME_SNAPSHOTS][CLOVER_FRAME_WIDTH];
} CloverFrame;

static inline int clover_frame_owner(const CloverFrame *frame)
{
    return frame ? clover_stage_owner(frame->stage) : -1;
}

/* One integer compare decides whether this service should touch the frame. */
static inline int clover_frame_is_mine(const CloverFrame *frame, int owner)
{
    return frame && frame->magic == (uint32_t)CLOVER_FRAME_MAGIC && clover_stage_owner(frame->stage) == owner;
}

static inline void clover_frame_advance(CloverFrame *frame, int more_tokens)
{
    if (frame) frame->stage = clover_stage_next(frame->stage, more_tokens);
}

static inline void clover_frame_init(CloverFrame *frame, uint32_t session)
{
    memset(frame, 0, sizeof *frame);
    frame->magic = (uint32_t)CLOVER_FRAME_MAGIC;
    frame->stage = clover_stage_first(0);
    frame->session = session;
}

static inline int clover_frame_write(FILE *file, const CloverFrame *frame)
{
    return file && frame && fwrite(frame, sizeof *frame, 1, file) == 1;
}

static inline int clover_frame_read(FILE *file, CloverFrame *frame)
{
    if (!file || !frame) return 0;
    if (fread(frame, sizeof *frame, 1, file) != 1) return 0;
    return frame->magic == (uint32_t)CLOVER_FRAME_MAGIC &&
           clover_stage_owner(frame->stage) >= 0 &&
           frame->snapshot_count <= CLOVER_FRAME_SNAPSHOTS;
}

#endif
