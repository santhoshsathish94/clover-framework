#define _GNU_SOURCE
#include <unistd.h>
#include "live-root.h"
#include <assert.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    char path[4096];
    RootScratch *scratch=calloc((size_t)omp_get_max_threads(),sizeof *scratch);
    assert(scratch);
    for (unsigned layer=1;layer<=92;layer++) {
        snprintf(path,sizeof path,"%s/root-%u",argv[1],layer);
        Root *root=root_open(path,layer);
        assert(root);
        if (layer==1 || layer==2 || layer==92) {
            snprintf(path,sizeof path,"%s/root-%u/observations/france/inputs.f32",argv[1],layer);
            FILE *input_file=fopen(path,"rb"); assert(input_file);
            float input[3584],gate[3072],up[3072],down[3584];
            assert(fread(input,sizeof(float),3584,input_file)==3584); assert(!fclose(input_file));
            snprintf(path,sizeof path,"%s/root-%u/observations/france/results.bin",argv[1],layer);
            FILE *result_file=fopen(path,"rb"); assert(result_file);
            int32_t expert; float expected_gate[3072],expected_up[3072],activation[3072],expected_down[3584];
            assert(fread(&expert,sizeof expert,1,result_file)==1);
            assert(fread(expected_gate,sizeof(float),3072,result_file)==3072);
            assert(fread(expected_up,sizeof(float),3072,result_file)==3072);
            assert(fread(activation,sizeof(float),3072,result_file)==3072);
            assert(fread(expected_down,sizeof(float),3584,result_file)==3584); assert(!fclose(result_file));
            assert(root_project(root,scratch,(unsigned)expert,0,input,gate));
            assert(root_project(root,scratch,(unsigned)expert,1,input,up));
            assert(root_project(root,scratch,(unsigned)expert,2,activation,down));
            assert(!memcmp(gate,expected_gate,sizeof gate) && !memcmp(up,expected_up,sizeof up) && !memcmp(down,expected_down,sizeof down));
            printf("PASS: layer %u live gate/up/down match independent recorded projections\n",layer); fflush(stdout);
        }
        root_close(root);
    }
    free(scratch);
    puts("PASS: all92metadata-driven root readers; no dataset modifications");
    return 0;
}