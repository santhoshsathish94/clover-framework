#define CLOVER_CHECK_EXPERT
#define main standalone_program_main
#include "clover-one.c"
#undef main
#include <assert.h>

static unsigned matched_experts;
static int compare_observations=1;
static void check_live_expert(unsigned layer,unsigned expert,const float *input,const float *output)
{
    if (!compare_observations) return;
    recorded_open((int)layer);
    for (unsigned record=0;record<80;record++) {
        const unsigned char *values=recorded_roots+(size_t)record*51204;
        int32_t recorded_expert;
        memcpy(&recorded_expert,values,sizeof recorded_expert);
        if (recorded_expert==(int32_t)expert && !memcmp(input,recorded_inputs+(size_t)record*LAT*4,LAT*4)) {
            assert(!memcmp(output,values+4+3*I_*4,LAT*4));
            matched_experts++;
            return;
        }
    }
    fprintf(stderr,"No independent reference match at position=%u layer=%u expert=%u\n",resident_position,layer,expert);
    assert(0);
}

int main(int argc,char **argv)
{
    assert(argc==2);
    resident_config=(GenerationConfig){128,128,163585};
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);
    resident_sequence_clear();
    assert(!setenv("K3_RESULT_SET","france",1));
    const unsigned tokens[]={1008,10484,318,15383,387};
    unsigned next=0;
    for (unsigned position=0;position<5;position++) {
        next=evaluate_token(tokens[position],position,position==4);
        assert(matched_experts==(position+1)*92*16);
        for (unsigned layer=0;layer<NLAY;layer++) assert(resident_sequences[layer].length==position+1);
        printf("PASS: prefix position%u live inputs/outputs match independent expert observations (%u matches)\n",position,matched_experts);
        fflush(stdout);
    }
    assert(next==17374);
    puts("PASS: full live five-token prefill selects original Paris token17374"); fflush(stdout);
    compare_observations=0;
    next=evaluate_token(next,5,1);
    assert(next<VOCAB && recorded_expert_hits==0 && resident_expert_calls==4416);
    for (unsigned layer=0;layer<NLAY;layer++) assert(resident_sequences[layer].length==6);
    printf("PASS: continuation token%u computed live using cached prefix, zero runtime observation matches\n",next); fflush(stdout);
    resident_sequence_clear();
    for (unsigned layer=0;layer<NLAY;layer++) {
        assert(resident_sequences[layer].length==0);
        if (resident_sequences[layer].state)
            for (unsigned coordinate=0;coordinate<H*D*D;coordinate++) assert(resident_sequences[layer].state[coordinate]==0);
    }
    resident_shutdown();
    puts("PASS: independent-prefix/live-generation check and new-request cache reset; no saved vectors");
    return 0;
}