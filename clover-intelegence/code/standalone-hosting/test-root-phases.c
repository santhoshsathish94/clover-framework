#define _GNU_SOURCE
#include <unistd.h>
#include <omp.h>
static double phase_seconds[4];
static void record_phase(unsigned phase,double elapsed)
{
#pragma omp atomic update
    phase_seconds[phase]+=elapsed;
}
#define ROOT_PHASE_BEGIN(name) double name=omp_get_wtime()
#define ROOT_PHASE_END(name,phase) record_phase(phase,omp_get_wtime()-(name))
#include "live-root.h"
#include <assert.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    const unsigned layers[]={1,2,92};
    RootScratch *scratch=calloc((size_t)omp_get_max_threads(),sizeof *scratch); assert(scratch);
    float input[3584],output[3584];
    for (unsigned coordinate=0;coordinate<3584;coordinate++) input[coordinate]=(float)((int)(coordinate%17)-8)/17.0f;
    for (unsigned test=0;test<3;test++) {
        char path[4096]; snprintf(path,sizeof path,"%s/root-%u",argv[1],layers[test]);
        Root *root=root_open(path,layers[test]); assert(root);
        double started=omp_get_wtime(); memset(phase_seconds,0,sizeof phase_seconds);
        for (unsigned expert=0;expert<16;expert++)
            for (unsigned matrix=0;matrix<3;matrix++) {
                assert(root_project(root,scratch,expert,matrix,input,output));
                for (unsigned row=0;row<(matrix==2?3584:3072);row++) assert(isfinite(output[row]));
            }
        printf("PHASES layer=%u projections=48 wall=%.6f worker_read=%.6f worker_decode=%.6f worker_crc=%.6f worker_math=%.6f\n",
            layers[test],omp_get_wtime()-started,phase_seconds[0],phase_seconds[1],phase_seconds[2],phase_seconds[3]);
        fflush(stdout);
        root_close(root);
    }
    free(scratch);
    return 0;
}