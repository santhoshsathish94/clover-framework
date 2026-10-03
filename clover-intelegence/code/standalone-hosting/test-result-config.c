#include "cache-budget.h"
#include "expert-results.h"
#include <assert.h>
#include <stdio.h>
int main(int argc,char **argv)
{
    assert(argc==2);
    GenerationConfig config;
    assert(generation_config_load(argv[1],&config));
    assert(config.expert_result_cache_mib==256);
    uint64_t gib=UINT64_C(1024)*1024*1024;
    CacheBudget budget=cache_budget(&config,124*gib,76*gib);
    assert(budget.results==256*1024*1024);
    ExpertResults cache;
    assert(expert_results_open(&cache,budget.results));
    assert(cache.allocated<=budget.results && cache.sets*RESULT_WAYS==8192);
    expert_results_close(&cache);
    budget=cache_budget(&config,12*gib,76*gib);
    assert(!budget.results);
    budget=cache_budget(&config,89*gib,76*gib);
    assert(budget.head+budget.pipeline+budget.results<=3*gib);
    const char *disabled="{\"max_input_tokens\":128,\"max_output_tokens\":128,\"eos_token_id\":163585,\"expert_result_cache_mib\":0}";
    assert(generation_config_parse((const unsigned char *)disabled,strlen(disabled),&config));
    assert(!cache_budget(&config,124*gib,76*gib).results);
    const char *invalid="{\"max_input_tokens\":128,\"max_output_tokens\":128,\"eos_token_id\":163585,\"expert_result_cache_mib\":8193}";
    assert(!generation_config_parse((const unsigned char *)invalid,strlen(invalid),&config));
    puts("PASS: result cache config, actual RAM bound, low-memory budget, explicit disable and invalid size rejection");
    return 0;
}