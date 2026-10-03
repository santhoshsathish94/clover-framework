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
    ExpertPipeline pipeline; assert(expert_pipeline_open(&pipeline,72*1024*1024));
    RootScratch *scratch=malloc(sizeof *scratch); assert(scratch);
    assert(expert_pipeline_submit(&pipeline,root,0,0));
    assert(expert_pipeline_acquire(&pipeline,root,0,0));
    assert(root_block(root,scratch,0,0,0));
    unsigned offset=ROOT_MATRICES*2+4;
    root->index[offset]^=1;
    assert(!root_block(root,scratch,0,0,0));
    root->index[offset]^=1;
    unsigned char saved=pipeline.buffers[0].data[0];
    pipeline.buffers[0].data[0]=0;
    assert(!root_block(root,scratch,0,0,0));
    pipeline.buffers[0].data[0]=saved;
    assert(root_block(root,scratch,0,0,0));
    expert_pipeline_release(root);
    Root *missing=calloc(1,sizeof *missing); assert(missing);
    missing->offsets=root->offsets;
    missing->file=tmpfile(); assert(missing->file);
    assert(expert_pipeline_submit(&pipeline,missing,0,1));
    assert(!expert_pipeline_acquire(&pipeline,missing,0,1));
    expert_pipeline_close(&pipeline);
    assert(!pipeline.initialized && !pipeline.buffers[0].data);
    fclose(missing->file); free(missing);
    assert(expert_pipeline_open(&pipeline,0) && !pipeline.initialized);
    expert_pipeline_close(&pipeline);
    root_close(root); free(scratch);
    puts("PASS: original CRC and zlib guards reject corrupted staged data; reader EOF fails; shutdown and disabled mode clean");
    return 0;
}