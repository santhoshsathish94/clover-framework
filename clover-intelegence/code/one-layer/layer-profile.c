#define _GNU_SOURCE
/* One layer measured in place.
   Runs a real prompt through the whole engine in passes of NPOS_SLOTS tokens, locks the
   experts of chosen layers in RAM, and reports those layers' time per token by stage.
   The stage hooks are the engine's own (CLOVER_PROFILE_STAGES); this file only drives them.
   Also hashes every position's final residual, so two builds or two widths can be compared
   bit for bit.

     layer-profile DATASET IDS_FILE NTOKENS PIN_LAYERS      (PIN_LAYERS: 0 or e.g. 1,3) */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
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

typedef struct { char name[96]; double ms; } Acc;
enum { MAX_PIN = 8, MAX_ACC = 160 };

static double resident_fraction(const unsigned char *base,size_t bytes)
{
    size_t pages=(bytes+4095)/4096, present=0;
    unsigned char *vec=malloc(pages);
    if (!vec || mincore((void *)base,bytes,vec)) { free(vec); return -1.0; }
    for (size_t p=0;p<pages;p++) if (vec[p]&1) present++;
    free(vec);
    return (double)present/(double)pages;
}

int main(int argc,char **argv)
{
    assert(argc==5 && omp_get_max_threads()<=128);
    const unsigned want=(unsigned)atoi(argv[3]);
    assert(resident_read_config()); resident_configure(argv[1]); result_options();
    double tick=now_s(); load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);
    printf("startup %.2f s  NPOS_SLOTS=%d  threads=%d\n",now_s()-tick,NPOS_SLOTS,omp_get_max_threads());

    static unsigned ids[4096]; unsigned n=0;
    FILE *list=fopen(argv[2],"r");
    if (!list) { fprintf(stderr,"cannot read ids\n"); return 1; }
    for (unsigned value; n<want && n<4096 && fscanf(list,"%u",&value)==1; ) {
        ids[n++]=value;
        int c=fgetc(list); if (c!=',' && c!=' ' && c!='\n' && c!=EOF) ungetc(c,list);
    }
    fclose(list);
    if (n<want) { fprintf(stderr,"only %u ids\n",n); return 1; }

    int pinned[MAX_PIN]; unsigned pin_count=0;
    if (strcmp(argv[4],"0")) {
        char *copy=strdup(argv[4]);
        for (char *part=strtok(copy,","); part && pin_count<MAX_PIN; part=strtok(NULL,",")) pinned[pin_count++]=atoi(part);
        free(copy);
    }
    for (unsigned k=0;k<pin_count;k++) {
        Root *root=resident_roots[pinned[k]];
        if (!root || !root->direct) { fprintf(stderr,"layer %d has no direct mapping\n",pinned[k]); return 1; }
        double mark=now_s();
        if (mlock(root->direct,root->direct_bytes)) { perror("mlock"); return 1; }
        printf("layer %d: %.2f GB locked in %.2f s, resident %.1f%%\n",pinned[k],
            (double)root->direct_bytes/1e9,now_s()-mark,100*resident_fraction(root->direct,root->direct_bytes));
    }
    fflush(stdout);

    static double layer_total[NLAY+1];
    static Acc acc[MAX_PIN][MAX_ACC]; unsigned acc_count[MAX_PIN]; memset(acc_count,0,sizeof acc_count);
    uint64_t hash=1469598103934665603ULL;
    lane_sequence_clear(0); active_lane=0; batch_decode=0;
    unsigned next=0, passes=0; double pass_seconds=0;
    for (unsigned position=0; position<n; position+=NPOS_SLOTS) {
        unsigned chunk=n-position<NPOS_SLOTS?n-position:NPOS_SLOTS;
        int last=position+chunk>=n;
        measurement_count=0; memset(worker_time,0,sizeof worker_time);
        memset(previous_workers,0,sizeof previous_workers);
        memset(previous_operators,0,sizeof previous_operators);
        previous_wait=resident_pipeline.wait_seconds;
        previous_stall=root_wait_seconds; previous_drain=root_drain_seconds;
        previous_read=root_pull_bytes;
        double started=now_s();
        next=evaluate_tokens(ids+position,chunk,position,last);
        pass_seconds+=now_s()-started; passes++;
        for (unsigned entry=0;entry<measurement_count;entry++) {
            const StageRow *row=&measurements[entry];
            if (row->layer>=0 && row->layer<=NLAY && !strcmp(row->name,"total:layer")) layer_total[row->layer]+=row->seconds*1000;
            for (unsigned k=0;k<pin_count;k++) if (row->layer==pinned[k]) {
                unsigned slot=0;
                while (slot<acc_count[k] && strcmp(acc[k][slot].name,row->name)) slot++;
                if (slot==acc_count[k]) { assert(slot<MAX_ACC); snprintf(acc[k][slot].name,96,"%s",row->name); acc[k][slot].ms=0; acc_count[k]++; }
                acc[k][slot].ms+=row->seconds*1000;
            }
        }
        for (unsigned t=0;t<chunk;t++) {
            const unsigned char *bytes=(const unsigned char *)resid[t];
            for (size_t b=0;b<sizeof resid[t];b++) { hash^=bytes[b]; hash*=1099511628211ULL; }
        }
    }

    printf("\ntokens %u  passes %u  width %d  whole model %.3f s = %.1f ms/token\n",n,passes,NPOS_SLOTS,pass_seconds,1000*pass_seconds/n);
    double others=0; unsigned other_count=0;
    for (int L=1;L<=92;L++) {
        int skip=0; for (unsigned k=0;k<pin_count;k++) if (pinned[k]==L) skip=1;
        if (!skip) { others+=layer_total[L]; other_count++; }
    }
    if (other_count) printf("average of the %u unpinned layers 1..92: %.2f ms/token (experts read through the normal path)\n",other_count,others/other_count/n);
    for (unsigned k=0;k<pin_count;k++) {
        Root *root=resident_roots[pinned[k]];
        printf("\nlayer %d (%s), experts locked, resident %.1f%%: %.2f ms/token in the layer\n",pinned[k],
            (((pinned[k]%4)==3 && pinned[k]<=91) || pinned[k]==92)?"MLA":"KDA",
            100*resident_fraction(root->direct,root->direct_bytes),layer_total[pinned[k]]/n);
        for (unsigned s=0;s<acc_count[k];s++)
            if (acc[k][s].ms/n>=0.05) printf("    %-42s %8.3f ms/token\n",acc[k][s].name,acc[k][s].ms/n);
    }
    printf("\nlast token %u\nresid hash %016llx\n",next,(unsigned long long)hash);
    fflush(stdout);
    resident_shutdown();
    return 0;
}
