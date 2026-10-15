/* Routing over a real 64-position generation: each output token is fed back in, so the
   expert set grows the way it would under sustained traffic rather than over one short
   prompt. Answers how much of a pod's 896 experts actually becomes resident. */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define CLOVER_CHECK_EXPERT
#define main standalone_program_main
static void check_live_expert(unsigned layer,unsigned expert,const float *input,const float *output);
#include "clover-one.c"
#undef main

#define POSITIONS 64
static unsigned chosen[NLAY][TOPK];
static unsigned chosen_n[NLAY];
static void check_live_expert(unsigned layer,unsigned expert,const float *input,const float *output)
{
    (void)input;(void)output;
    if (layer<NLAY && chosen_n[layer]<TOPK) chosen[layer][chosen_n[layer]++]=expert;
}

int main(int argc,char **argv)
{
    assert(argc==2);
    assert(resident_read_config()); resident_configure(argv[1]); result_options();
    load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);
    unsigned ids[POSITIONS]={1008,10484,318,15383,387};
    unsigned seeded=5;
    for (unsigned p=0;p<POSITIONS;p++) {
        memset(chosen_n,0,sizeof chosen_n);
        unsigned next=evaluate_token(ids[p],p,p>0);
        for (unsigned l=0;l<NLAY;l++) {
            if (!chosen_n[l]) continue;
            printf("ROUTE {\"position\":%u,\"layer\":%u,\"experts\":[",p,l);
            for (unsigned j=0;j<chosen_n[l];j++) printf("%s%u",j?",":"",chosen[l][j]);
            printf("]}\n");
        }
        if (p+1<POSITIONS && p+1>=seeded) ids[p+1]=next;
        fflush(stdout);
    }
    resident_shutdown();
    return 0;
}
