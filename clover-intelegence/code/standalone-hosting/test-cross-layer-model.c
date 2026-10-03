#define CLOVER_CHECK_EXPERT
#define main standalone_program_main
#include "clover-one.c"
#undef main
#include <assert.h>

static unsigned matched_experts;
static void check_live_expert(unsigned layer,unsigned expert,const float *input,const float *output)
{
    recorded_open((int)layer);
    for (unsigned record=0;record<80;record++) {
        const unsigned char *values=recorded_roots+(size_t)record*51204;
        int32_t expected_expert;
        memcpy(&expected_expert,values,sizeof expected_expert);
        if (expected_expert==(int32_t)expert && !memcmp(input,recorded_inputs+(size_t)record*LAT*4,LAT*4)) {
            assert(!memcmp(output,values+4+3*I_*4,LAT*4));
            matched_experts++;
            return;
        }
    }
    fprintf(stderr,"Independent expert mismatch: position=%u layer=%u expert=%u\n",resident_position,layer,expert);
    assert(0);
}

int main(int argc,char **argv)
{
    assert(argc==2);
    assert(generation_config_load("bin/configs/fresh-request.json",&resident_config));
    resident_configure(argv[1]); result_options();
    load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);
    assert(resident_pipeline.initialized && !resident_results.sets);
    assert(!setenv("K3_RESULT_SET","france",1));
    const unsigned tokens[]={1008,10484};
    for (unsigned position=0;position<2;position++) {
        evaluate_token(tokens[position],position,0);
        assert(matched_experts==(position+1)*92*16);
        assert(resident_cross_layer.submitted==(position+1)*92);
        assert(resident_cross_layer.hits+resident_cross_layer.misses==resident_cross_layer.submitted);
        assert(resident_pipeline.reads==matched_experts+resident_cross_layer.misses);
        assert(!resident_cross_layer.armed && resident_expert_calls==4416 && !recorded_expert_hits);
        for (unsigned layer=0;layer<NLAY;layer++) assert(resident_sequences[layer].length==position+1);
        printf("PASS: position%u %u expert inputs/outputs byte-exact; cross-layer hits=%llu misses=%llu\n",position,
            matched_experts,(unsigned long long)resident_cross_layer.hits,(unsigned long long)resident_cross_layer.misses);
        fflush(stdout);
    }
    assert(expert_results_open(&resident_results,256*1024*1024) && resident_results.sets);
    for (unsigned request=0;request<2;request++) {
        resident_sequence_clear();
        ResidentCacheStats before=resident_cache_stats();
        unsigned matched_before=matched_experts;
        evaluate_token(tokens[0],0,0);
        ResidentCacheStats after=resident_cache_stats();
        assert(matched_experts==matched_before+92*16);
        assert(after.cross_submitted-before.cross_submitted==92);
        assert(after.pipeline_reads-before.pipeline_reads==resident_expert_calls/3+after.cross_misses-before.cross_misses);
        assert(!resident_cross_layer.armed);
        if (request) assert(resident_result_hits>0);
        printf("PASS: cached-result request%u remains byte-exact; result hits=%llu live projections=%llu\n",
            request,resident_result_hits,resident_expert_calls);
        fflush(stdout);
    }
    resident_sequence_clear();
    assert(!resident_cross_layer.armed);
    resident_shutdown();
    puts("PASS: cross-layer speculation preserves all 93 layers and incremental state without prior requests");
    return 0;
}