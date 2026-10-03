#define _GNU_SOURCE
#include <assert.h>
#include <unistd.h>
#include "live-root.h"
#include "expert-pipeline.h"
#include "cross-layer-prefetch.h"

int main(int argc,char **argv)
{
    assert(argc==2);
    char path[4096]; snprintf(path,sizeof path,"%s/root-1",argv[1]);
    Root *root=root_open(path,1); assert(root);
    snprintf(path,sizeof path,"%s/root-2",argv[1]);
    Root *other=root_open(path,2); assert(other);
    ExpertPipeline pipeline; CrossLayerPrefetch ahead={0};
    assert(expert_pipeline_open(&pipeline,72*1024*1024));
    RootScratch *scratch=malloc(sizeof *scratch); assert(scratch);
    const int selected[]={1,2,0}; unsigned missing[]={0,1,2};
    assert(cross_layer_submit(&ahead,&pipeline,root,0));
    assert(!cross_layer_submit(&ahead,&pipeline,root,1));
    assert(cross_layer_match(&ahead,&pipeline,root,selected,missing,3));
    assert(missing[0]==2 && missing[1]==1 && missing[2]==0);
    assert(expert_pipeline_acquire(&pipeline,root,0,0));
    assert(root_block(root,scratch,0,0,0));
    expert_pipeline_release(root);
    assert(!cross_layer_match(&ahead,&pipeline,root,selected,missing,3));
    assert(cross_layer_submit(&ahead,&pipeline,other,0));
    assert(!cross_layer_match(&ahead,&pipeline,root,selected,missing,3));
    assert(expert_pipeline_submit(&pipeline,root,2,0));
    assert(expert_pipeline_acquire(&pipeline,root,2,0));
    assert(root_block(root,scratch,2,0,0));
    expert_pipeline_release(root);
    assert(cross_layer_submit(&ahead,&pipeline,root,3));
    assert(!cross_layer_match(&ahead,&pipeline,root,selected,missing,3));
    assert(cross_layer_submit(&ahead,&pipeline,root,0));
    assert(!cross_layer_match(&ahead,&pipeline,root,selected,missing,0));
    assert(ahead.submitted==4 && ahead.hits==1 && ahead.misses==3);
    expert_pipeline_close(&pipeline);
    assert(expert_pipeline_open(&pipeline,0));
    assert(cross_layer_submit(&ahead,&pipeline,root,0));
    assert(!ahead.armed && ahead.submitted==4);
    expert_pipeline_close(&pipeline);
    root_close(other); root_close(root); free(scratch);
    puts("PASS: cross-layer match, miss, wrong-layer rejection, cached-result skip, shutdown and disabled path");
    return 0;
}