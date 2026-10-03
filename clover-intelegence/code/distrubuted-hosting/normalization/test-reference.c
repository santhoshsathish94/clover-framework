#define _GNU_SOURCE
#define NORMALIZATION_NO_MAIN
#include "normalization.c"
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define main original_main
#include "reference-tail.inc"
#undef main

static float captured_input[5][7168], captured_snapshots[5][8][7168];
static unsigned captured_count, captured_routes[5][16], oracle_hits[3];
static unsigned char oracle_records[80][51204];
static float oracle_inputs[80][3584];
static const char *dataset_root, *prompt_name;
static int selected_layers[93];

static void test_read(const char *path, void *output, size_t length)
{
    FILE *file=fopen(path,"rb");
    if(!file){fprintf(stderr,"missing reference data: %s\n",path);abort();}
    assert(fread(output,1,length,file)==length);
    assert(fgetc(file)==EOF && !ferror(file));
    assert(!fclose(file));
}

static void test_capture_input(int layer)
{
    if(!layer)return;
    memcpy(captured_input,resid,sizeof captured_input);
    captured_count=(unsigned)nsnap;
    assert(captured_count>0&&captured_count<=8);
    for(unsigned position=0;position<5;position++) for(unsigned snapshot=0;snapshot<captured_count;snapshot++)
        memcpy(captured_snapshots[position][snapshot],snap[position][snapshot],7168*4);
    memset(oracle_hits,0,sizeof oracle_hits);
    char path[4096];
    int length=snprintf(path,sizeof path,"%s/root-%d/observations/%s/results.bin",dataset_root,layer,prompt_name);
    assert(length>0&&(size_t)length<sizeof path);
    test_read(path,oracle_records,sizeof oracle_records);
    length=snprintf(path,sizeof path,"%s/root-%d/observations/%s/inputs.f32",dataset_root,layer,prompt_name);
    assert(length>0&&(size_t)length<sizeof path);
    test_read(path,oracle_inputs,sizeof oracle_inputs);
}

static void test_capture_route(int layer,int position,const int *ids)
{
    assert(layer>0&&layer<=92&&position>=0&&position<5);
    for(unsigned rank=0;rank<16;rank++)captured_routes[position][rank]=(unsigned)ids[rank];
}

static void test_observed_expert(int layer,int expert,const char *part,float *const *outputs,
    const float *const *inputs,int positions,int width,int rows)
{
    assert(layer>0&&layer<=92&&expert>=0&&expert<896);
    unsigned matrix=!strcmp(part,"gate")?0:!strcmp(part,"up")?1:2;
    assert((matrix==2&&width==3072&&rows==3584)||(matrix<2&&width==3584&&rows==3072));
    for(int position=0;position<positions;position++) {
        int found=-1;
        for(unsigned record=0;record<80;record++) {
            uint32_t id;memcpy(&id,oracle_records[record],4);
            if(id!=(unsigned)expert)continue;
            const void *expected=matrix==2?(void *)(oracle_records[record]+4+2*3072*4):(void *)oracle_inputs[record];
            if(!memcmp(inputs[position],expected,(size_t)width*4)){found=(int)record;break;}
        }
        if(found<0){fprintf(stderr,"reference observation input mismatch L%d E%d %s\n",layer,expert,part);abort();}
        size_t offset=matrix==2?4+3*3072*4:4+matrix*3072*4;
        memcpy(outputs[position],oracle_records[found]+offset,(size_t)rows*4);
        oracle_hits[matrix]++;
    }
}

static const char *test_leaves;

static void test_finish_layer(int layer)
{
    if (!layer) return;
    for (unsigned matrix = 0; matrix < 3; matrix++) assert(oracle_hits[matrix] == 80);
    if (layer != 92) return;
    assert(nsnap == 8);
    Normalization *model = NULL;
    assert(normalization_open(test_leaves,&model));
    float gains[3][7168], fold[7168];
    for (unsigned tensor = 0; tensor < 3; tensor++) {
        const MRec *record = &mrec[tensor + 1];
        assert(record->nbytes == 14336);
        const unsigned char *raw = file_ptr(record->file_id) + record->off;
        for (unsigned coordinate = 0; coordinate < 7168; coordinate++) {
            uint32_t bits = ((uint32_t)raw[coordinate * 2] | ((uint32_t)raw[coordinate * 2 + 1] << 8)) << 16;
            memcpy(gains[tensor] + coordinate,&bits,4);
        }
        assert(!memcmp(gains[tensor],model->gains[tensor],sizeof gains[tensor]));
    }
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) fold[coordinate] = gains[0][coordinate] * gains[1][coordinate];
    assert(!memcmp(fold,model->fold,sizeof fold));
    float aggregate[7168], expected[7168], actual[7168], snapshots[8*7168];
    float *sources[9];
    for (unsigned position = 0; position < 5; position++) {
        for (unsigned snapshot = 0; snapshot < 8; snapshot++) {
            sources[snapshot] = snap[position][snapshot];
            memcpy(snapshots + snapshot * 7168,sources[snapshot],7168*4);
        }
        sources[8] = resid[position];
        AR(aggregate,sources,9,fold);
        rmsnorm(expected,aggregate,gains[2],7168,EPS5);
        assert(normalization_process(model,resid[position],snapshots,8,actual));
        size_t differences = 0;
        for (unsigned coordinate = 0; coordinate < 7168; coordinate++)
            if (memcmp(actual + coordinate,expected + coordinate,4)) differences++;
        if (differences) fprintf(stderr,"normalization differs at position %u: %zu values\n",position,differences);
        assert(!differences);
    }
    normalization_close(model);
    printf("NORMALIZATION_PASS\t%s\t5\t35840\t21504\n",prompt_name);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    dataset_root = argv[1];
    prompt_name = argv[2];
    test_leaves = argv[3];
    char *args[] = {"original-tail-reference","92",NULL};
    return original_main(2,args);
}