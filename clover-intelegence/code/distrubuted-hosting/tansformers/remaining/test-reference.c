#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define main original_main
#include "reference-all.inc"
#undef main

static float captured_input[5][7168], captured_snapshots[5][8][7168];
static unsigned captured_count, captured_routes[5][16], oracle_hits[3];
static unsigned char oracle_records[80][51204];
static float oracle_inputs[80][3584];
static const char *dataset_root, *packages_root, *prompt_name;
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

static void test_finish_layer(int layer)
{
    if(!layer)return;
    for(unsigned matrix=0;matrix<3;matrix++)assert(oracle_hits[matrix]==80);
    if(!selected_layers[layer])return;
    char path[4096];
    int length=snprintf(path,sizeof path,"%s/transformer-%d/bin/test-layer.so",packages_root,layer);
    assert(length>0&&(size_t)length<sizeof path);
    void *library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!library){fprintf(stderr,"%s\n",dlerror());abort();}
    int (*open_model)(const char *,void **);
    void *(*create_sequence)(const void *);
    int (*process)(void *,void *,const float *,const float *,unsigned,float *,float *);
    int (*read_routes)(const void *,unsigned *);
    void (*close_sequence)(void *),(*close_model)(void *),(*reset_sequence)(void *);
    void *symbol=dlsym(library,"transformer_open");memcpy(&open_model,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_sequence_create");memcpy(&create_sequence,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_process");memcpy(&process,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_sequence_close");memcpy(&close_sequence,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_close");memcpy(&close_model,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_sequence_reset");memcpy(&reset_sequence,&symbol,sizeof symbol);assert(symbol);
    symbol=dlsym(library,"transformer_routes");memcpy(&read_routes,&symbol,sizeof symbol);assert(symbol);
    void *model=NULL;assert(open_model(dataset_root,&model));
    void *sequence=create_sequence(model);assert(sequence);
    float output[7168],snapshots[8*7168];
    for(unsigned position=0;position<5;position++) {
        assert(process(model,sequence,captured_input[position],&captured_snapshots[position][0][0],captured_count,output,snapshots));
        unsigned routes[16];assert(read_routes(sequence,routes));
        assert(!memcmp(routes,captured_routes[position],sizeof routes));
        size_t different=0;
        for(unsigned coordinate=0;coordinate<7168;coordinate++)if(memcmp(output+coordinate,resid[position]+coordinate,4))different++;
        if(different){fprintf(stderr,"L%d position%u differs at %zu residual coordinates\n",layer,position,different);abort();}
        for(int snapshot=0;snapshot<nsnap;snapshot++)assert(!memcmp(snapshots+(size_t)snapshot*7168,snap[position][snapshot],7168*4));
    }
    assert(!process(model,sequence,captured_input[0],&captured_snapshots[0][0][0],captured_count-1,output,snapshots));
    if(layer==2||layer==3||layer==12||layer==92) {
        reset_sequence(sequence);
        assert(process(model,sequence,captured_input[0],&captured_snapshots[0][0][0],captured_count,output,snapshots));
        assert(!memcmp(output,resid[0],sizeof output));
    }
    close_sequence(sequence);close_model(model);assert(!dlclose(library));
    printf("VERIFIED\t%d\t%s\t5\t%d\n",layer,prompt_name,nsnap);fflush(stdout);
}

int main(int argc,char **argv)
{
    if(argc!=5)return 2;
    dataset_root=argv[1];packages_root=argv[2];prompt_name=argv[3];
    int last=0;
    if(!strcmp(argv[4],"all")){for(int layer=2;layer<=92;layer++)selected_layers[layer]=1;last=92;}
    else {
        const char *cursor=argv[4];
        while(*cursor){char *end;long layer=strtol(cursor,&end,10);assert(end!=cursor&&layer>=2&&layer<=92);selected_layers[layer]=1;if(layer>last)last=(int)layer;cursor=end;if(*cursor){assert(*cursor==',');cursor++;}}
    }
    char limit[16];snprintf(limit,sizeof limit,"%d",last);
    char *args[]={"original-layer-reference",limit,NULL};
    return original_main(2,args);
}