#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "live-root.h"
#include <assert.h>

static size_t block_offset(unsigned expert,unsigned matrix,unsigned block)
{
    return (size_t)expert*ROOT_EXPERT_RAW +
        (matrix<2 ? (size_t)(matrix*48+block)*ROOT_RAW
                  : (size_t)96*ROOT_RAW+(size_t)block*ROOT_DOWN_RAW);
}

int main(int argc,char **argv)
{
    assert(argc==3 && omp_get_max_threads()<=128);
    const char *dataset=argv[1];
    unsigned layer=(unsigned)atoi(argv[2]);
    char source[4096],target[4096],staging[4096];
    assert(snprintf(source,sizeof source,"%s/root-%u",dataset,layer)>0);
    assert(snprintf(target,sizeof target,"%s/experts.direct",source)>0);
    assert(snprintf(staging,sizeof staging,"%s/experts.direct.building",source)>0);

    Root *root=root_open(source,layer);
    assert(root && !root->direct);
    RootScratch *pool=calloc((size_t)omp_get_max_threads(),sizeof *pool);
    assert(pool);

    const size_t bytes=(size_t)896*ROOT_EXPERT_RAW;
    int handle=open(staging,O_WRONLY|O_CREAT|O_EXCL,0644);
    assert(handle>=0 && !ftruncate(handle,(off_t)bytes));

    double started=omp_get_wtime();
    int ok=1;
#pragma omp parallel for schedule(dynamic,4) reduction(&:ok)
    for (unsigned expert=0;expert<896;expert++) {
        if (!ok) continue;
        RootScratch *scratch=pool+omp_get_thread_num();
        for (unsigned matrix=0;matrix<3 && ok;matrix++) {
            unsigned width=matrix==2?3072:3584, blocks=matrix==2?56:48;
            size_t block_bytes=(size_t)64*width/2+(size_t)64*width/32;
            for (unsigned block=0;block<blocks;block++) {
                if (!root_block(root,scratch,expert,matrix,block)) { ok=0; break; }
                ssize_t count=pwrite(handle,scratch->raw,block_bytes,(off_t)block_offset(expert,matrix,block));
                if (count<0 || (size_t)count!=block_bytes) { ok=0; break; }
            }
        }
    }
    assert(ok);
    assert(!fsync(handle) && !close(handle));
    assert(!rename(staging,target));
    double converted=omp_get_wtime()-started;

    /* Re-decode from the compressed source and compare against what was written. */
    root_close(root);
    root=root_open(source,layer);
    assert(root && root->direct && root->direct_bytes==bytes);
    started=omp_get_wtime();
    unsigned long long checked=0;
    int same=1;
#pragma omp parallel for schedule(dynamic,1) reduction(&:same) reduction(+:checked)
    for (unsigned sample=0;sample<16;sample++) {
        if (!same) continue;
        RootScratch *scratch=pool+omp_get_thread_num();
        unsigned expert=(sample*56+layer)%896;
        for (unsigned matrix=0;matrix<3 && same;matrix++) {
            unsigned width=matrix==2?3072:3584, blocks=matrix==2?56:48;
            size_t block_bytes=(size_t)64*width/2+(size_t)64*width/32;
            for (unsigned block=0;block<blocks;block++) {
                if (!root_block(root,scratch,expert,matrix,block)) { same=0; break; }
                if (memcmp(scratch->raw,root->direct+block_offset(expert,matrix,block),block_bytes)) { same=0; break; }
                checked++;
            }
        }
    }
    assert(same && checked==16*152);
    printf("CONVERTED layer=%u file_bytes=%zu convert_seconds=%.3f verified_blocks=%llu verify_seconds=%.3f\n",
        layer,bytes,converted,checked,omp_get_wtime()-started);
    root_close(root);
    free(pool);
    return 0;
}
