#define TRANSFORMER_NO_MAIN
#include CANDIDATE_SOURCE
#include <assert.h>

int main(int argc,char **argv)
{
    if(argc!=2)return 2;
    Transformer *model=NULL;assert(transformer_open(argv[1],&model));
    TransformerSequence *sequence=transformer_sequence_create(model);assert(sequence);
    float input[7168]={0},snapshots[8*7168]={0},output[7168],returned[8*7168];
    assert(!transformer_process(model,sequence,input,snapshots,TRANSFORMER_INPUT_SNAPSHOTS-1,output,returned));
    for(unsigned position=0;position<2;position++) {
        assert(transformer_process(model,sequence,input,snapshots,TRANSFORMER_INPUT_SNAPSHOTS,output,returned));
        for(unsigned coordinate=0;coordinate<7168;coordinate++)assert(output[coordinate]==0.0f);
        assert(!memcmp(snapshots,returned,TRANSFORMER_OUTPUT_SNAPSHOTS*7168*4));
    }
    assert(sequence->positions==2);
    input[0]=NAN;assert(!transformer_process(model,sequence,input,snapshots,TRANSFORMER_INPUT_SNAPSHOTS,output,returned));
    transformer_sequence_reset(sequence);assert(!sequence->positions&&!sequence->failed);
    assert(!sequence->mla_cache&&!sequence->mla_capacity);
    transformer_sequence_close(sequence);transformer_close(model);
    printf("CONTROLS\t%u\tPASS\n",TRANSFORMER_LAYER);
    return 0;
}