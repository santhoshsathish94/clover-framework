#include "cache-budget.h"
#include <assert.h>
int main(int argc,char **argv)
{
    assert(argc==2);
    GenerationConfig config;
    assert(generation_config_load(argv[1],&config));
    assert(config.expert_pipeline_mib==72);
    const uint64_t gib=UINT64_C(1024)*1024*1024;
    CacheBudget budget=cache_budget(&config,124*gib,76*gib);
    assert(budget.head==2348810240ULL && budget.pipeline==72*1024*1024);
    budget=cache_budget(&config,12*gib,76*gib);
    assert(!budget.head && !budget.pipeline);
    puts("PASS: pipeline-only expert staging, bounded head/pipeline allocation and low-memory fallback");
    return 0;
}