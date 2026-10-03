#ifndef CLOVER_HEAD_CACHE_H
#define CLOVER_HEAD_CACHE_H
#include "numeric-table.h"
#include <omp.h>

typedef struct {
    ClientOutput *output;
    unsigned char *values;
    unsigned loaded;
    size_t allocated;
    int owned;
    uint64_t decoded_blocks, reused_blocks;
} HeadCache;

static void head_cache_close(HeadCache *cache)
{
    if (!cache) return;
    if (cache->owned) free(cache->values);
    memset(cache,0,sizeof *cache);
}

static int head_cache_open(HeadCache *cache, ClientOutput *output, size_t budget)
{
    memset(cache,0,sizeof *cache);
    cache->output=output;
    if (!output) return 0;
    size_t bytes=(size_t)CLIENT_BLOCKS*CLIENT_RAW_BYTES;
    if (output->table->direct) {
        cache->values=(unsigned char *)output->table->direct;
        cache->loaded=CLIENT_BLOCKS;
        return 1;
    }
    if (budget<bytes) return 1;
    cache->values=malloc(bytes);
    if (cache->values) { cache->allocated=bytes; cache->owned=1; }
    return 1;
}

static int head_cache_prepare(HeadCache *cache)
{
    if (!cache->values) return 1;
    while (cache->loaded<CLIENT_BLOCKS) {
        unsigned block=cache->loaded;
        if (!table_load(cache->output->table,block)) return 0;
        memcpy(cache->values+(size_t)block*CLIENT_RAW_BYTES,cache->output->table->raw,CLIENT_RAW_BYTES);
        cache->decoded_blocks++;
        cache->loaded++;
    }
    return 1;
}

static int head_cache_project(HeadCache *cache, const float *input, float *logits, uint32_t *selected)
{
    if (!cache || !cache->output || !input || !selected || !client_numeric_environment()) return 0;
    if (!cache->values) return client_output_project(cache->output,input,logits,selected);
    for (unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++) if (!isfinite(input[coordinate])) return 0;
    unsigned previously_loaded=cache->loaded;
    if (!head_cache_prepare(cache)) return 0;
    float *scores=logits ? logits : malloc(CLIENT_ROWS*sizeof(float));
    if (!scores) return 0;
#pragma omp parallel for schedule(static)
    for (unsigned row=0;row<CLIENT_ROWS;row++)
        scores[row]=client_head_row(cache->values+(size_t)row*CLIENT_WIDTH*2,input);
    unsigned best=0;
    int valid=1;
    for (unsigned row=0;row<CLIENT_ROWS;row++) {
        if (!isfinite(scores[row])) valid=0;
        if (scores[row]>scores[best]) best=row;
    }
    cache->reused_blocks+=previously_loaded;
    if (!logits) free(scores);
    if (!valid) return 0;
    *selected=best;
    return 1;
}
#endif