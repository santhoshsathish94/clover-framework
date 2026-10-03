#ifndef CLOVER_EXPERT_RESULTS_H
#define CLOVER_EXPERT_RESULTS_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { RESULT_WIDTH=3584, RESULT_WAYS=8 };
typedef struct {
    uint64_t hash, used;
    unsigned layer, expert;
    float input[RESULT_WIDTH], output[RESULT_WIDTH];
} ExpertResult;
typedef struct {
    ExpertResult *entries;
    size_t sets, allocated;
    uint64_t clock, hits, misses, stores, evictions;
} ExpertResults;

static void expert_results_close(ExpertResults *cache)
{
    if (!cache) return;
    free(cache->entries);
    memset(cache,0,sizeof *cache);
}

static int expert_results_open(ExpertResults *cache,size_t budget)
{
    memset(cache,0,sizeof *cache);
    size_t possible=budget/(RESULT_WAYS*sizeof(ExpertResult));
    if (!possible) return 1;
    size_t sets=1;
    while (sets<=possible/2) sets*=2;
    cache->entries=calloc(sets*RESULT_WAYS,sizeof *cache->entries);
    if (!cache->entries) return 1;
    cache->sets=sets;
    cache->allocated=sets*RESULT_WAYS*sizeof *cache->entries;
    return 1;
}

static uint64_t expert_results_hash(const float *input)
{
    uint64_t hash=UINT64_C(14695981039346656037);
    for (size_t offset=0;offset<RESULT_WIDTH*sizeof(float);offset+=8) {
        uint64_t word;
        memcpy(&word,(const unsigned char *)input+offset,sizeof word);
        hash^=word;
        hash*=UINT64_C(1099511628211);
        hash^=hash>>32;
    }
    return hash;
}

static size_t expert_results_set(const ExpertResults *cache,unsigned layer,unsigned expert,uint64_t hash)
{
    hash^=(uint64_t)layer*UINT64_C(0x9e3779b97f4a7c15);
    hash^=(uint64_t)expert*UINT64_C(0xbf58476d1ce4e5b9);
    hash^=hash>>30; hash*=UINT64_C(0xbf58476d1ce4e5b9);
    hash^=hash>>27; hash*=UINT64_C(0x94d049bb133111eb);
    hash^=hash>>31;
    return (size_t)hash&(cache->sets-1);
}

static int expert_results_find(ExpertResults *cache,unsigned layer,unsigned expert,
    uint64_t hash,const float *input,float *output)
{
    if (!cache || !input || !output || layer<1 || layer>92 || expert>=896) return 0;
    if (cache->sets) {
        ExpertResult *set=cache->entries+expert_results_set(cache,layer,expert,hash)*RESULT_WAYS;
        for (unsigned way=0;way<RESULT_WAYS;way++) {
            ExpertResult *entry=&set[way];
            if (entry->used && entry->hash==hash && entry->layer==layer && entry->expert==expert &&
                !memcmp(entry->input,input,RESULT_WIDTH*sizeof(float))) {
                memcpy(output,entry->output,RESULT_WIDTH*sizeof(float));
                entry->used=++cache->clock;
                cache->hits++;
                return 1;
            }
        }
    }
    cache->misses++;
    return 0;
}

static void expert_results_store(ExpertResults *cache,unsigned layer,unsigned expert,
    uint64_t hash,const float *input,const float *output)
{
    if (!cache || !cache->sets || !input || !output || layer<1 || layer>92 || expert>=896) return;
    ExpertResult *set=cache->entries+expert_results_set(cache,layer,expert,hash)*RESULT_WAYS;
    ExpertResult *victim=&set[0];
    for (unsigned way=0;way<RESULT_WAYS;way++) {
        ExpertResult *entry=&set[way];
        if (entry->used && entry->hash==hash && entry->layer==layer && entry->expert==expert &&
            !memcmp(entry->input,input,RESULT_WIDTH*sizeof(float))) { victim=entry; break; }
        if (!entry->used) { victim=entry; break; }
        if (entry->used<victim->used) victim=entry;
    }
    if (victim->used) cache->evictions++;
    victim->used=0;
    victim->hash=hash; victim->layer=layer; victim->expert=expert;
    memcpy(victim->input,input,RESULT_WIDTH*sizeof(float));
    memcpy(victim->output,output,RESULT_WIDTH*sizeof(float));
    victim->used=++cache->clock;
    cache->stores++;
}
#endif