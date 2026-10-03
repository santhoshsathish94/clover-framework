#define _GNU_SOURCE
#include <unistd.h>
#include "live-root.h"
#include "expert-pipeline.h"
#include <assert.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    char path[4096]; snprintf(path,sizeof path,"%s/root-1",argv[1]);
    Root *root=root_open(path,1); assert(root);
    ExpertPipeline pipeline;
    assert(expert_pipeline_open(&pipeline,72*1024*1024) && pipeline.initialized);
    RootScratch *scratch=calloc((size_t)omp_get_max_threads(),sizeof *scratch); assert(scratch);
    float input[3584],expected[16][3][3584],actual[3584];
    for (unsigned coordinate=0;coordinate<3584;coordinate++) input[coordinate]=(float)((int)(coordinate%17)-8)/17.0f;
    double begin=pipeline_seconds();
    for (unsigned expert=0;expert<16;expert++)
        for (unsigned matrix=0;matrix<3;matrix++) assert(root_project(root,scratch,expert,matrix,input,expected[expert][matrix]));
    double direct=pipeline_seconds()-begin;
    assert(expert_pipeline_submit(&pipeline,root,0,0));
    begin=pipeline_seconds();
    for (unsigned expert=0;expert<16;expert++) {
        assert(expert_pipeline_acquire(&pipeline,root,expert,expert%2));
        if (expert+1<16) assert(expert_pipeline_submit(&pipeline,root,expert+1,(expert+1)%2));
        for (unsigned matrix=0;matrix<3;matrix++) {
            assert(root_project(root,scratch,expert,matrix,input,actual));
            assert(!memcmp(expected[expert][matrix],actual,(matrix==2?3584:3072)*sizeof(float)));
        }
        expert_pipeline_release(root);
    }
    double staged=pipeline_seconds()-begin;
    assert(pipeline.reads==16 && !root->prefetched);
    printf("PASS: 16experts/48projections exact; direct=%.6fs pipeline=%.6fs waits=%.6fs read_worker=%.6fs bytes=%llu buffers=%zu\n",
        direct,staged,pipeline.wait_seconds,pipeline.read_seconds,(unsigned long long)pipeline.bytes,
        pipeline.buffers[0].capacity+pipeline.buffers[1].capacity);
    expert_pipeline_close(&pipeline); root_close(root); free(scratch);
    return 0;
}