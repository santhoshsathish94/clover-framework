#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "head-cache.h"
#include <assert.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    ClientOutput *head=client_output_open(argv[1]); assert(head);
    const int mapped=head->table->direct!=NULL;
    HeadCache cache;
    const size_t bytes=(size_t)CLIENT_BLOCKS*CLIENT_RAW_BYTES;
    assert(head_cache_open(&cache,head,bytes));
    if (mapped) assert(cache.values==head->table->direct && !cache.allocated && !cache.owned && cache.loaded==CLIENT_BLOCKS);
    else assert(cache.allocated==bytes && cache.owned && !cache.loaded);
    float input[CLIENT_WIDTH];
    float *expected=malloc(CLIENT_ROWS*sizeof(float)), *actual=malloc(CLIENT_ROWS*sizeof(float));
    assert(expected && actual);
    for (unsigned test=0;test<2;test++) {
        for (unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)
            input[coordinate]=(float)((int)((coordinate+test*11)%37)-18)/37.0f;
        uint32_t reference,selected;
        double begin=omp_get_wtime();
        assert(client_output_project(head,input,expected,&reference));
        double baseline=omp_get_wtime()-begin;
        begin=omp_get_wtime();
        assert(head_cache_project(&cache,input,actual,&selected));
        double cached=omp_get_wtime()-begin;
        assert(!memcmp(expected,actual,CLIENT_ROWS*sizeof(float)) && reference==selected);
        assert(cache.loaded==CLIENT_BLOCKS);
        assert(cache.decoded_blocks==(mapped?0:CLIENT_BLOCKS));
        assert(cache.reused_blocks==(uint64_t)(mapped?test+1:test)*CLIENT_BLOCKS);
        printf("PASS: input%u all163840scores exact; stream=%.6fs head=%.6fs decodes=%llu reused=%llu RAM=%zu mapped=%d\n",
            test,baseline,cached,(unsigned long long)cache.decoded_blocks,
            (unsigned long long)cache.reused_blocks,cache.allocated,mapped);
        fflush(stdout);
    }
    head_cache_close(&cache);
    assert(!cache.values && !cache.allocated);
    assert(head_cache_open(&cache,head,0));
    if (mapped) assert(cache.values==head->table->direct && !cache.allocated);
    else assert(!cache.values && !cache.allocated);
    head_cache_close(&cache);
    client_output_close(head); free(expected); free(actual);
    puts("PASS: output-head exactness for changed input, no decoding when mapped, and bounded fallback allocation");
    return 0;
}