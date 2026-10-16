/* observe-route-any.c - routing over a real generation, for any prompt.
 *
 * A copy of observe-route64.c generalised to take the prompt from the command line and
 * to run a chosen number of output tokens, so the same observation can be repeated
 * across many prompts of different input lengths without editing and rebuilding.
 *
 * Why this and not a vector dump: a full CLOVER_VECTORS capture is about 267 MB per
 * position, so 128 positions is 34 GB for ONE prompt. Routing is what actually varies
 * -- of roughly 120 stages per pod only four change the carried state, and the
 * expensive one is the MoE whose cost is set entirely by which experts are chosen.
 * Printing the selection per layer per position is a few hundred KB for a whole run.
 *
 * Each output token is fed back as the next input, so the expert sets grow the way they
 * would under sustained generation rather than over one short prompt.
 *
 * build: gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp \
 *        -DNPOS_SLOTS=8 -I. observe-route-any.c -lm -o /tmp/observe-route-any
 * run:   observe-route-any DATASET OUTPUT_TOKENS id id id ...
 */
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

#define MAXPOS 240
static unsigned chosen[NLAY][TOPK];
static unsigned chosen_n[NLAY];
static void check_live_expert(unsigned layer,unsigned expert,const float *input,const float *output)
{
    (void)input;(void)output;
    if (layer<NLAY && chosen_n[layer]<TOPK) chosen[layer][chosen_n[layer]++]=expert;
}

int main(int argc,char **argv)
{
    if (argc < 4) { fprintf(stderr,"usage: %s DATASET OUTPUT_TOKENS id [id ...]\n",argv[0]); return 2; }
    unsigned wanted = (unsigned)strtoul(argv[2],NULL,10);
    unsigned seeded = (unsigned)(argc - 3);
    unsigned total = seeded + wanted;
    if (total > MAXPOS) { fprintf(stderr,"total positions %u exceeds %d\n",total,MAXPOS); return 2; }

    static unsigned ids[MAXPOS];
    for (unsigned i=0;i<seeded;i++) ids[i]=(unsigned)strtoul(argv[3+i],NULL,10);

    assert(resident_read_config()); resident_configure(argv[1]); result_options();
    load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);

    printf("RUN {\"seeded\":%u,\"wanted\":%u,\"total\":%u}\n",seeded,wanted,total);
    for (unsigned p=0;p<total;p++) {
        memset(chosen_n,0,sizeof chosen_n);
        unsigned next=evaluate_token(ids[p],p,p>0);
        for (unsigned l=0;l<NLAY;l++) {
            if (!chosen_n[l]) continue;
            printf("ROUTE {\"position\":%u,\"layer\":%u,\"generated\":%d,\"experts\":[",
                   p,l,p>=seeded?1:0);
            for (unsigned j=0;j<chosen_n[l];j++) printf("%s%u",j?",":"",chosen[l][j]);
            printf("]}\n");
        }
        if (p+1<total && p+1>=seeded) ids[p+1]=next;   /* never overwrite the prompt */
        if (p+1>=seeded) printf("TOKEN {\"position\":%u,\"emitted\":%u}\n",p,next);
        fflush(stdout);
    }
    resident_shutdown();
    return 0;
}
