#ifndef CLOVER_CACHE_BUDGET_H
#define CLOVER_CACHE_BUDGET_H
#include "generation.h"
typedef struct { size_t head, pipeline, results; } CacheBudget;

static CacheBudget cache_budget(const GenerationConfig *config,uint64_t available,uint64_t fixed_bytes)
{
    const uint64_t mib=1024*1024;
    uint64_t reserved=fixed_bytes+(uint64_t)config->memory_reserve_mib*mib+2*1024*mib;
    uint64_t usable=available>reserved?available-reserved:0;
    CacheBudget result={0};
    uint64_t head=UINT64_C(163840)*7168*2;
    if ((uint64_t)config->head_cache_mib*mib>=head && usable>=head) {
        result.head=(size_t)head;
        usable-=head;
    }
    uint64_t pipeline=(uint64_t)config->expert_pipeline_mib*mib;
    if (pipeline>=72*mib && usable>=pipeline) {
        result.pipeline=(size_t)pipeline;
        usable-=pipeline;
    }
    uint64_t results=(uint64_t)config->expert_result_cache_mib*mib;
    if (results>usable) results=usable;
    result.results=(size_t)results;
    return result;
}
#endif