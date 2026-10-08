#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
static double worker_time[128][4];
#define ROOT_PHASE_BEGIN(name) double name=omp_get_wtime()
#define ROOT_PHASE_END(name,phase) (worker_time[omp_get_thread_num()][phase]+=omp_get_wtime()-(name))
#define CLOVER_PROFILE_STAGES
#define main standalone_program_main
#include "clover-one.c"
#undef main

typedef struct { int layer; const char *name; unsigned count; double seconds; } StageRow;
static StageRow measurements[8192];
static unsigned measurement_count;
static double boundary_time;
static double previous_workers[4], previous_operators[OP_COUNT], previous_wait;
static double previous_stall, previous_drain; static unsigned long long previous_read;
static char operator_names[OP_COUNT][96];

static void profile_duration(int layer,const char *stage,double seconds)
{
    if (!strcmp(stage,"total:layer")) {
        const char *names[]={"worker:read","worker:decode","worker:crc","worker:math"};
        for (unsigned phase=0;phase<4;phase++) {
            double total=0;
            for (unsigned worker=0;worker<128;worker++) total+=worker_time[worker][phase];
            profile_duration(layer,names[phase],total-previous_workers[phase]);
            previous_workers[phase]=total;
        }
        for (unsigned operation=0;operation<OP_COUNT;operation++) {
            if (!operator_names[operation][0]) snprintf(operator_names[operation],96,"op:%s",OPN[operation]);
            if (op_t[operation]>previous_operators[operation])
                profile_duration(layer,operator_names[operation],op_t[operation]-previous_operators[operation]);
            previous_operators[operation]=op_t[operation];
        }
        profile_duration(layer,"detail:read-ahead-wait",resident_pipeline.wait_seconds-previous_wait);
        previous_wait=resident_pipeline.wait_seconds;
        /* the expert reader pipeline: what this layer waited on and what it pulled */
        profile_duration(layer,"detail:expert-stall",root_wait_seconds-previous_stall);
        previous_stall=root_wait_seconds;
        profile_duration(layer,"detail:expert-drain",root_drain_seconds-previous_drain);
        previous_drain=root_drain_seconds;
        profile_duration(layer,"detail:expert-read-gb",
            (double)(root_pull_bytes-previous_read)/1e9);
        previous_read=root_pull_bytes;
    }
    for (unsigned entry=0;entry<measurement_count;entry++)
        if (measurements[entry].layer==layer && !strcmp(measurements[entry].name,stage)) {
            measurements[entry].seconds+=seconds; measurements[entry].count++; return;
        }
    assert(measurement_count<8192);
    measurements[measurement_count++]=(StageRow){layer,stage,1,seconds};
}
static void profile_boundary(int layer,const char *stage)
{
    double tick=now_s();
    if (stage) profile_duration(layer,stage,tick-boundary_time);
    boundary_time=now_s();
}
static void profile_projection(const unsigned char *weights,double seconds)
{
    for (unsigned slot=0;slot<37;slot++)
        if (prepared_slots[slot] && prepared_pointer(prepared_layer,slot)==weights) {
            profile_duration(prepared_layer,SLOTN[slot],seconds);return;
        }
    assert(0);
}
static void report(unsigned step,unsigned positions,double total)
{
    for (unsigned entry=0;entry<measurement_count;entry++) {
        const StageRow *row=&measurements[entry];
        printf("STAGE_JSON {\"step\":%u,\"positions\":%u,\"layer\":%d,\"stage\":\"%s\",\"ms\":%.6f,\"calls\":%u}\n",
            step,positions,row->layer,row->name,row->seconds*1000,row->count);
    }
    for (unsigned phase=0;phase<4;phase++) {
        double seconds=0;
        for (unsigned worker=0;worker<128;worker++) seconds+=worker_time[worker][phase];
        printf("WORKER_JSON {\"step\":%u,\"phase\":%u,\"worker_seconds\":%.9f}\n",step,phase,seconds);
    }
    printf("POSITION_JSON {\"step\":%u,\"positions\":%u,\"total_seconds\":%.9f}\n",step,positions,total);
    fflush(stdout);
}

/* Profile the two shapes the engine actually runs: one prompt pass carrying every
   prompt position through a single weight sweep, then decode steps carrying one. */
int main(int argc,char **argv)
{
    assert(argc==2 && omp_get_max_threads()<=128);
    assert(resident_read_config()); resident_configure(argv[1]); result_options();
    double tick=now_s(); load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);
    printf("STARTUP_JSON {\"seconds\":%.9f}\n",now_s()-tick); fflush(stdout);

    static const unsigned prompt[]={1008,10484,318,15383,387};   /* The capital of France is */
    const unsigned count=(unsigned)(sizeof prompt/sizeof *prompt);
    const unsigned decode_steps=2;
    lane_sequence_clear(0);
    active_lane=0; batch_decode=0;

    unsigned emitted[8]; unsigned produced=0;
    for (unsigned step=0;step<=decode_steps;step++) {
        measurement_count=0; memset(worker_time,0,sizeof worker_time);
        memset(previous_workers,0,sizeof previous_workers);
        memset(previous_operators,0,sizeof previous_operators);
        previous_wait=resident_pipeline.wait_seconds;
        previous_stall=root_wait_seconds; previous_drain=root_drain_seconds;
        previous_read=root_pull_bytes;
        unsigned positions=step?1:count;
        unsigned token=step?emitted[produced-1]:0;
        tick=now_s();
        unsigned next=step?evaluate_tokens(&token,1,count+step-1,1)
                         :evaluate_tokens(prompt,count,0,1);
        double elapsed=now_s()-tick;
        emitted[produced++]=next;
        report(step,positions,elapsed);
    }
    printf("EMITTED");
    for (unsigned i=0;i<produced;i++) printf(" %u",emitted[i]);
    printf("\n");
    assert(emitted[0]==17374);   /* Paris: the prompt pass must still answer the same */
    resident_shutdown();
    puts("PASS: profiled all 93 layers over one prompt pass and two decode steps");
    return 0;
}