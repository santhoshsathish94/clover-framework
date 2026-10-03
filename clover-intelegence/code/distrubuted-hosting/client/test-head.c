#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define CLIENT_NO_MAIN
#include "client.c"
#include <assert.h>

static int test_seek(FILE *file,uint64_t offset)
{
#ifdef _WIN32
    return _fseeki64(file,(int64_t)offset,SEEK_SET)==0;
#else
    return fseeko(file,(off_t)offset,SEEK_SET)==0;
#endif
}
static float reference_row(const unsigned char *raw,const float *input)
{
    double accum[16]={0};
    for(unsigned coordinate=0;coordinate<7168;coordinate+=16)for(unsigned lane=0;lane<16;lane++){
        unsigned offset=(coordinate+lane)*2;
        uint32_t bits=((uint32_t)raw[offset]|(uint32_t)raw[offset+1]<<8)<<16;
        float value;memcpy(&value,&bits,4);
        accum[lane]+=(double)value*(double)input[coordinate+lane];
    }
    double groups[4];
    for(unsigned group=0;group<4;group++)groups[group]=(accum[group]+accum[group+4])+(accum[group+8]+accum[group+12]);
    return(float)((groups[0]+groups[1])+(groups[2]+groups[3]));
}
int main(int argc,char **argv)
{
    if(argc!=4)return 2;
    FILE *source=fopen(argv[3],"rb");assert(source);
    ClientInput *input=client_input_open(argv[1]);ClientOutput *head=client_output_open(argv[2]);assert(input&&head);
    const unsigned rows[]={0,15,16,176,1008,163839};float embedding[7168];unsigned char original[14336];
    for(unsigned index=0;index<sizeof rows/sizeof *rows;index++){
        assert(client_input(input,rows[index],embedding));
        assert(test_seek(source,UINT64_C(2348810824)+(uint64_t)rows[index]*14336));
        assert(fread(original,1,sizeof original,source)==sizeof original);
        for(unsigned coordinate=0;coordinate<7168;coordinate++){
            uint32_t bits=((uint32_t)original[coordinate*2]|(uint32_t)original[coordinate*2+1]<<8)<<16;
            assert(!memcmp(embedding+coordinate,&bits,4));
        }
    }
    float vector[7168];for(unsigned coordinate=0;coordinate<7168;coordinate++)vector[coordinate]=(float)((int)(coordinate%37)-18)/37.0f;
    float *scores=malloc(CLIENT_ROWS*sizeof(float));assert(scores);uint32_t selected;
    assert(client_output_project(head,vector,scores,&selected));
    assert(test_seek(source,584));
    unsigned char *block=malloc(CLIENT_RAW_BYTES);assert(block);uint32_t reference_best=0;float maximum=0;
    for(unsigned ordinal=0;ordinal<CLIENT_BLOCKS;ordinal++){
        assert(fread(block,1,CLIENT_RAW_BYTES,source)==CLIENT_RAW_BYTES);
        assert(table_load(head->table,ordinal));assert(!memcmp(block,head->table->raw,CLIENT_RAW_BYTES));
        for(unsigned local=0;local<16;local++){
            unsigned id=ordinal*16+local;float score=reference_row(block+(size_t)local*14336,vector);
            assert(!memcmp(&score,scores+id,4));
            if(!id||score>maximum){maximum=score;reference_best=id;}
        }
    }
    assert(reference_best==selected);
    free(block);free(scores);assert(!fclose(source));client_input_close(input);client_output_close(head);
    puts("PASS: six embedding rows, all output-table bytes and 163840 head logits/argmax match original shard");
}