#define _GNU_SOURCE
#define CLIENT_NO_MAIN
#include "../client/client.c"
#include <dlfcn.h>
#include <time.h>

typedef struct { float residual[7168]; float snapshots[8*7168]; } PipelineToken;
typedef struct {
    void *library,*model,*sequence;
    void (*close_model)(void *);
    void (*close_sequence)(void *);
} PipelineStage;

static double pipeline_clock(void)
{
    struct timespec time;clock_gettime(CLOCK_MONOTONIC,&time);return(double)time.tv_sec+(double)time.tv_nsec/1e9;
}
static int pipeline_path(char *path,size_t capacity,const char *root,const char *relative)
{
    int length=snprintf(path,capacity,"%s/%s",root,relative);return length>0&&(size_t)length<capacity;
}
static int pipeline_symbol(void *library,const char *name,void *destination,size_t bytes)
{
    void *symbol=dlsym(library,name);
    if(!symbol||bytes!=sizeof symbol){fprintf(stderr,"missing stage symbol %s\n",name);return 0;}
    memcpy(destination,&symbol,bytes);return 1;
}
static void pipeline_stage_close(PipelineStage *stage)
{
    if(stage->sequence&&stage->close_sequence)stage->close_sequence(stage->sequence);
    if(stage->model&&stage->close_model)stage->close_model(stage->model);
    if(stage->library)dlclose(stage->library);
    memset(stage,0,sizeof *stage);
}
static int pipeline_stage_open(PipelineStage *stage,const char *directory,const char *prefix)
{
    char path[4096],name[128];memset(stage,0,sizeof *stage);
    if(!pipeline_path(path,sizeof path,directory,"bin/pipeline-stage.so"))return 0;
    stage->library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!stage->library){fprintf(stderr,"stage load: %s\n",dlerror());return 0;}
    snprintf(name,sizeof name,"%s_close",prefix);
    if(!pipeline_symbol(stage->library,name,&stage->close_model,sizeof stage->close_model)){pipeline_stage_close(stage);return 0;}
    return 1;
}
static int pipeline_sequence(PipelineStage *stage,const char *prefix)
{
    char name[128];void *(*create)(const void *);
    snprintf(name,sizeof name,"%s_sequence_create",prefix);
    if(!pipeline_symbol(stage->library,name,&create,sizeof create))return 0;
    snprintf(name,sizeof name,"%s_sequence_close",prefix);
    if(!pipeline_symbol(stage->library,name,&stage->close_sequence,sizeof stage->close_sequence))return 0;
    stage->sequence=create(stage->model);return stage->sequence!=NULL;
}
static int pipeline_server(const char *root,PipelineToken *tokens,size_t count)
{
    char directory[4096],qkv[4096],trunk[4096];PipelineStage stage;
    if(!pipeline_path(directory,sizeof directory,root,"server")||!pipeline_stage_open(&stage,directory,"server"))return 0;
    int (*open_model)(const char *,const char *,void **);
    int (*process)(const void *,void *,const float *,float *,float *);
    int valid=pipeline_symbol(stage.library,"server_open",&open_model,sizeof open_model)&&
        pipeline_symbol(stage.library,"server_process",&process,sizeof process)&&
        pipeline_path(qkv,sizeof qkv,directory,"bin/dataset/trunk-0-qkv/qkv.bin")&&
        pipeline_path(trunk,sizeof trunk,directory,"bin/dataset/trunk-0")&&open_model(qkv,trunk,&stage.model)&&pipeline_sequence(&stage,"server");
    for(size_t position=0;valid&&position<count;position++)valid=process(stage.model,stage.sequence,tokens[position].residual,
        tokens[position].residual,tokens[position].snapshots);
    pipeline_stage_close(&stage);return valid;
}
static int pipeline_transformer(const char *root,unsigned layer,PipelineToken *tokens,size_t count)
{
    char relative[128],directory[4096],dataset[4096];PipelineStage stage;
    snprintf(relative,sizeof relative,"tansformers/transformer-%u",layer);
    if(!pipeline_path(directory,sizeof directory,root,relative)||!pipeline_stage_open(&stage,directory,"transformer"))return 0;
    int (*open_model)(const char *,void **);
    int valid=pipeline_symbol(stage.library,"transformer_open",&open_model,sizeof open_model)&&
        pipeline_path(dataset,sizeof dataset,directory,"bin/dataset")&&open_model(dataset,&stage.model)&&pipeline_sequence(&stage,"transformer");
    if(layer==1){
        int (*process)(void *,void *,const float *,const float *,float *);
        valid=valid&&pipeline_symbol(stage.library,"transformer_process",&process,sizeof process);
        for(size_t position=0;valid&&position<count;position++)valid=process(stage.model,stage.sequence,tokens[position].residual,
            tokens[position].snapshots,tokens[position].residual);
    }else{
        int (*process)(void *,void *,const float *,const float *,unsigned,float *,float *);
        valid=valid&&pipeline_symbol(stage.library,"transformer_process",&process,sizeof process);
        unsigned snapshots=(layer+11)/12;
        for(size_t position=0;valid&&position<count;position++)valid=process(stage.model,stage.sequence,tokens[position].residual,
            tokens[position].snapshots,snapshots,tokens[position].residual,tokens[position].snapshots);
    }
    pipeline_stage_close(&stage);return valid;
}
static int pipeline_normalization(const char *root,const PipelineToken *last,float output[7168])
{
    char directory[4096],leaves[4096];PipelineStage stage;
    if(!pipeline_path(directory,sizeof directory,root,"normalization")||!pipeline_stage_open(&stage,directory,"normalization"))return 0;
    int (*open_model)(const char *,void **);
    int (*process)(const void *,const float *,const float *,unsigned,float *);
    int valid=pipeline_symbol(stage.library,"normalization_open",&open_model,sizeof open_model)&&
        pipeline_symbol(stage.library,"normalization_process",&process,sizeof process)&&
        pipeline_path(leaves,sizeof leaves,directory,"bin/dataset/leaves.json")&&open_model(leaves,&stage.model)&&
        process(stage.model,last->residual,last->snapshots,8,output);
    pipeline_stage_close(&stage);return valid;
}
static int pipeline_predict(const char *root,const uint32_t *ids,size_t count)
{
    char path[4096],vocab[4096],config[4096];Tokenizer *tokenizer=NULL;ClientInput *input=NULL;ClientOutput *head=NULL;
    PipelineToken *tokens=calloc(count,sizeof *tokens);int success=0;
    if(!tokens)goto finish;
    if(!pipeline_path(path,sizeof path,root,"client/bin/dataset/tiktoken.model")||
        !pipeline_path(vocab,sizeof vocab,root,"client/bin/dataset/vocabulary.bin")||
        !pipeline_path(config,sizeof config,root,"client/bin/configs/tokenizer_config.json")||
        tokenizer_open(path,vocab,config,&tokenizer)!=TOKENIZER_OK)goto finish;
    if(!pipeline_path(path,sizeof path,root,"client/bin/dataset/inputs/seed.bin")||!(input=client_input_open(path)))goto finish;
    for(size_t position=0;position<count;position++)if(!client_input(input,ids[position],tokens[position].residual))goto finish;
    client_input_close(input);input=NULL;
    fprintf(stderr,"Pipeline: %zu tokens, one next-token prediction\n",count);fflush(stderr);
    double started=pipeline_clock(),tick=started;
    if(!pipeline_server(root,tokens,count)){fputs("layer0 failed\n",stderr);goto finish;}
    fprintf(stderr,"Layer 0 complete: %.3fs\n",pipeline_clock()-tick);fflush(stderr);
    for(unsigned layer=1;layer<=92;layer++){
        tick=pipeline_clock();
        if(!pipeline_transformer(root,layer,tokens,count)){fprintf(stderr,"layer%u failed\n",layer);goto finish;}
        fprintf(stderr,"Layer %u complete: %.3fs\n",layer,pipeline_clock()-tick);fflush(stderr);
    }
    float normalized[7168];
    if(!pipeline_normalization(root,&tokens[count-1],normalized)){fputs("normalization failed\n",stderr);goto finish;}
    fputs("Normalization complete; evaluating client output head\n",stderr);fflush(stderr);
    if(!pipeline_path(path,sizeof path,root,"client/bin/dataset/outputs/fruit.bin")||!(head=client_output_open(path)))goto finish;
    uint32_t selected;TokenText text;
    if(!client_output(head,tokenizer,normalized,&selected,&text))goto finish;
    fprintf(stderr,"NEXT_TOKEN %u; full pipeline %.3fs\n",selected,pipeline_clock()-started);fflush(stderr);
    if(fwrite(text.bytes,1,text.length,stdout)!=text.length||fflush(stdout)||ferror(stdout))goto finish;
    success=1;
finish:
    client_input_close(input);client_output_close(head);tokenizer_close(tokenizer);free(tokens);
    if(!success)fputs("pipeline failed; no predicted token delivered\n",stderr);
    return success;
}
int main(int argc,char **argv)
{
    if(argc<3||argc>258){fputs("usage: pipeline CODE_ROOT TOKEN_ID... (max 256)\n",stderr);return 2;}
    uint32_t ids[256];
    for(int index=2;index<argc;index++)if(!decimal_id((const unsigned char *)argv[index],strlen(argv[index]),&ids[index-2]))return 2;
    return pipeline_predict(argv[1],ids,(size_t)argc-2)?0:1;
}