#include "expert-results.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    ExpertResults cache,other;
    float input[RESULT_WIDTH]={0}, changed[RESULT_WIDTH]={0}, expected[RESULT_WIDTH], actual[RESULT_WIDTH];
    for (unsigned coordinate=0;coordinate<RESULT_WIDTH;coordinate++) expected[coordinate]=(float)coordinate/32;
    size_t budget=RESULT_WAYS*sizeof(ExpertResult);
    assert(expert_results_open(&cache,budget) && cache.allocated==budget && cache.sets==1);
    assert(expert_results_open(&other,budget));
    uint64_t hash=expert_results_hash(input);
    assert(!expert_results_find(&cache,1,76,hash,input,actual));
    expert_results_store(&cache,1,76,hash,input,expected);
    assert(expert_results_find(&cache,1,76,hash,input,actual) && !memcmp(actual,expected,sizeof actual));
    assert(!expert_results_find(&cache,2,76,hash,input,actual));
    assert(!expert_results_find(&cache,1,77,hash,input,actual));
    assert(!expert_results_find(&other,1,76,hash,input,actual));
    changed[100]=1;
    assert(!expert_results_find(&cache,1,76,hash,changed,actual));
    memset(changed,0,sizeof changed);
    uint32_t negative_zero=UINT32_C(0x80000000); memcpy(changed,&negative_zero,4);
    assert(!expert_results_find(&cache,1,76,hash,changed,actual));
    for (unsigned expert=100;expert<100+RESULT_WAYS;expert++) expert_results_store(&cache,1,expert,hash,input,expected);
    assert(!expert_results_find(&cache,1,76,hash,input,actual));
    assert(cache.evictions==1 && cache.allocated<=budget);
    assert(expert_results_find(&cache,1,107,hash,input,actual));
    float copy[RESULT_WIDTH]; memcpy(copy,actual,sizeof copy);
    for (unsigned expert=200;expert<200+RESULT_WAYS;expert++) expert_results_store(&cache,1,expert,hash,input,input);
    assert(!memcmp(copy,expected,sizeof copy));
    expert_results_close(&cache); expert_results_close(&other);
    assert(expert_results_open(&cache,0) && !cache.entries && !cache.allocated);
    expert_results_store(&cache,1,76,hash,input,expected);
    assert(!expert_results_find(&cache,1,76,hash,input,actual));
    expert_results_close(&cache);
    assert(expert_results_open(&cache,budget));
    assert(!expert_results_find(&cache,1,76,hash,input,actual));
    expert_results_close(&cache);
    puts("PASS: exact vectors, layer/expert/model-owner isolation, forced hash collision, signed zero, eviction, copied-hit lifetime and disabled cache");
    return 0;
}